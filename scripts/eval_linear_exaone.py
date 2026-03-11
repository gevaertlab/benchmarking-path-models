# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms as pth_transforms
from read_data_patchbag_hdf5_v2 import PatchBagDataset
import pickle
import numpy as np
import utils
import vision_transformerv2 as vits
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_auc_score, balanced_accuracy_score, f1_score
import random
import timm
from PIL import Image
import pandas as pd
#from EXAONEPath.macenko import macenko_normalizer

def frozen_features(arch, model, images, n, avgpool, batch_size):
    with torch.no_grad():
        if "vit" in arch or arch == "exaone":
            print(f"Input images shape: {images.shape}")
            individual_patch = images.view(-1, images.size(2), images.size(3), images.size(4))
            print(f"individual_patch shape: {individual_patch.shape}")
            output = model(individual_patch)  # [batch_size * bag_size, embed_dim]
            print(f"output shape: {output.shape}")
            reshaped_patch = output.reshape(batch_size, -1, output.size(-1))  # [batch_size, bag_size, embed_dim]
            print(f"output shape before mean: {reshaped_patch.shape}")
            output = torch.mean(reshaped_patch, dim=1)  # [batch_size, embed_dim]
            print(f"output shape after mean: {output.shape}")
        else:
            individual_patch = images.view(-1, images.size(2), images.size(3), images.size(4))
            output = model(individual_patch)
            print(f"output shape: {output.shape}")
            reshaped_patch = output.reshape(batch_size, -1, output.size(1))
            print(f"output shape before mean: {reshaped_patch.shape}")
            output = torch.mean(reshaped_patch, dim=1)
        print(f"output shape after stack: {output.shape}")
    return output

@torch.no_grad()
def extract_features(arch, model, loader, n, avgpool, output_dir, name=''):
    filepath = os.path.join(output_dir, f'{arch}_{name}.pth')
    if os.path.exists(filepath):
        features = torch.load(filepath)
        return features
    features = []
    header = f'{name} feature extraction: '
    metric_logger = utils.MetricLogger(delimiter="  ")
    for it, sample in enumerate(metric_logger.log_every(loader, 10)):
        images = sample['image']
        index = sample['label']
        print('label is ...', index)
        images = images.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        print('images shape is ..', images.shape)
        print('index shape is ..', index.shape)
        batch_size = images.shape[0]
        output = frozen_features(arch, model, images, n, avgpool, batch_size)
        features.append([output.cpu(), index])
    torch.save(features, filepath)
    return features

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    elif args.arch == "exaone":
        print("Exaone Path model")
        model = vits.vit_base(patch_size=args.patch_size, num_classes=0)
        if args.pretrained_weights and os.path.isfile(args.pretrained_weights) and os.path.getsize(args.pretrained_weights) > 0:
            print(f"Loading custom pretrained weights from {args.pretrained_weights}")
            checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
            state_dict = checkpoint.get('state_dict', checkpoint)
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights found at {args.pretrained_weights} and loaded with msg: {msg}")
        else:
            print("No valid pretrained weights provided; using random weights")
        model.cuda()
        model.eval()
        embed_dim = 768
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    if 'scratch' in args.train_from_scratch:
        print("training from scratch...")
        model.train()
    elif 'pretrain' in args.train_from_scratch:
        print("training from pretrained weights...")
        model.eval()
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # ============ preparing data ... ============
    #normalizer = macenko_normalizer()
    val_transform = pth_transforms.Compose([
        #pth_transforms.Lambda(lambda img: normalizer(img) if isinstance(img, Image.Image) else img),
        pth_transforms.Resize(256, interpolation=pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    print('embed_dim is..', embed_dim)
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    if args.no_aug:
        train_transform = val_transform
    else:
        train_transform = pth_transforms.Compose([
            #pth_transforms.Lambda(lambda img: normalizer(img) if isinstance(img, Image.Image) else img),
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    train_csv_fold0 = args.train_csv_path
    val_csv_fold0 = args.val_csv_path
    folds = ['fold1', 'fold2', 'fold3', 'fold4']
    dataset_train_list = [train_csv_fold0]
    dataset_val_list = [val_csv_fold0]
    for f in folds:
        dataset_train_list.append(train_csv_fold0.replace('fold0', f))
        dataset_val_list.append(val_csv_fold0.replace('fold0', f))
    print('dataset_train_list is ,', dataset_train_list)
    print('dataset_val_list is ,', dataset_val_list)

    for i, (data_train_fold, data_val_fold) in enumerate(zip(dataset_train_list, dataset_val_list)):
        fold = 'fold' + str(i)
        print('current fold is ..', fold)
        output_dir = os.path.join(args.output_dir, fold)
        os.makedirs(output_dir, exist_ok=True)
        print("Now loading training data...")
        dataset_train = PatchBagDataset(args.patch_data_path, data_train_fold, args.img_size, args.max_patches_total, transform=train_transform, bag_size=args.bag_size)
        print(len(dataset_train))
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("Now loading validation data...")
        dataset_val = PatchBagDataset(args.patch_data_path, data_val_fold, args.img_size, args.max_patches_total, transform=val_transform, bag_size=args.bag_size)
        print(len(dataset_val))
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("len of val loader is ", len(val_loader))
        print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

        print("Now extracting features....")
        if args.no_aug:
            train_features = extract_features(args.arch, model, train_loader, args.n_last_blocks, args.avgpool_patchtokens, output_dir, 'train')
            train_loader = train_features
            val_features = extract_features(args.arch, model, val_loader, args.n_last_blocks, args.avgpool_patchtokens, output_dir, 'val')
            val_loader = val_features
            print("Features are ready!\nStarting to train the linear classifier.")

        if 'scratch' in args.train_from_scratch:
            print("optimizers from scratch...")
            start = time.time()
            params = list(linear_classifier.parameters()) + list(model.parameters())
            optimizer = torch.optim.SGD(
                params,
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
                momentum=0.9,
                weight_decay=0,
            )
        else:
            print("optimizers from pretrained weights...")
            start = time.time()
            optimizer = torch.optim.SGD(
                linear_classifier.parameters(),
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
                momentum=0.9,
                weight_decay=0,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
        to_restore = {"epoch": 0, "best_acc": 0.}
        utils.restart_from_checkpoint(
            os.path.join(output_dir, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = to_restore["epoch"]
        best_acc = to_restore["best_acc"]

        for epoch in range(start_epoch, args.epochs):
            if isinstance(train_loader, DataLoader):
                train_loader.sampler.set_epoch(epoch)
            print("Training model now.....")
            train_stats = train(args.no_aug, args.arch, model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                test_stats = validate_network(args.no_aug, args.arch, val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.batch_size_per_gpu, output_dir)
                print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} val images for {fold}: {test_stats['acc1']:.1f}%")
                print(f"f1 score at epoch {epoch} of the network on the {len(dataset_val)} val images for {fold}: {test_stats['f1_score']:.1f}%")
                print(f"precision at epoch {epoch} of the network on the {len(dataset_val)} val images for {fold}: {test_stats['precision']:.1f}%")
                print(f"recall at epoch {epoch} of the network on the {len(dataset_val)} val images for {fold} : {test_stats['recall']:.1f}%")
                print(f"balanced_accuracy at epoch {epoch} of the network on the {len(dataset_val)} val images for {fold} : {test_stats['balanced_accuracy']:.1f}%")
                best_acc = max(best_acc, test_stats["acc1"])

                if epoch == args.epochs - 1:
                    df = pd.DataFrame({
                        'Accuracy': [round(test_stats['acc1'], 2)],
                        'F1_score': [round(test_stats['f1_score'], 2)],
                        'Precision': [round(test_stats['precision'], 2)],
                        'Recall': [round(test_stats['recall'], 2)],
                        'balanced_accuracy': [round(test_stats['balanced_accuracy'], 2)]
                    })
                    output_file = os.path.join(output_dir, 'metrics.csv')
                    df.to_csv(output_file, encoding='utf-8', index=False)

                print(f'Max accuracy so far: {best_acc:.2f}%')
                log_stats = {**{k: v for k, v in log_stats.items()}, **{f'test_{k}': v for k, v in test_stats.items()}}
            if utils.is_main_process():
                with (Path(output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": linear_classifier.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_acc": best_acc,
                }
                torch.save(save_dict, os.path.join(output_dir, "checkpoint.pth.tar"))
        print("Training of the supervised linear classifier on frozen features completed.\n"
              "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
        end = time.time()
        print('time taken for validation is ', (end - start) / 60, 'minutes')

    if args.evaluate:
        dataset_test = PatchBagDataset(args.patch_data_path, args.test_csv_path, args.img_size, args.test_max_patches_total, transform=val_transform, bag_size=args.test_bag_size)
        print(len(dataset_test))
        print(f"Data loaded with {len(dataset_test)} test imgs.")
        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.test_batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("in args evaluate, size of test loader", len(test_loader.dataset))
        if args.no_aug:
            test_features = extract_features(args.arch, model, test_loader, args.n_last_blocks, args.avgpool_patchtokens, args.output_dir, 'test')
            test_loader = test_features

        test_stats2 = validate_network(args.no_aug, args.arch, test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, args.test_batch_size_per_gpu, args.output_dir)
        print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats2['acc1']:.1f}%")
        print(f"F1 score(0-1) of the network on the {len(dataset_test)} test images: {test_stats2['f1_score']:.2f}")
        print(f"Precision(0-1) of the network on the {len(dataset_test)} test images: {test_stats2['precision']:.2f}")
        print(f"Recall(0-1) of the network on the {len(dataset_test)} test images: {test_stats2['recall']:.2f}")
        print(f"balanced_accuracy(0-1) of the network on the {len(dataset_test)} test images: {test_stats2['balanced_accuracy']:.2f}")
        df = pd.DataFrame({
            'Accuracy': [round(test_stats2['acc1'], 2)],
            'F1_score': [round(test_stats2['f1_score'], 2)],
            'Precision': [round(test_stats2['precision'], 2)],
            'Recall': [round(test_stats2['recall'], 2)],
            'balanced_accuracy': [round(test_stats2['balanced_accuracy'], 2)]
        })
        output_file = os.path.join(args.output_dir, 'test_metrics.csv')
        df.to_csv(output_file, encoding='utf-8', index=False)
        end_time_for_run = time.time()
        print('total_run time is ..', (end_time_for_run - start) / 60, 'minutes....')
        return

def train(has_frozen, arch, model, linear_classifier, optimizer, train_loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    if isinstance(train_loader, DataLoader):
        iterator = metric_logger.log_every(train_loader, 10, header)
    else:
        iterator = metric_logger.log_every(train_loader, 10, header)
    
    for data in iterator:
        if isinstance(train_loader, DataLoader):
            inp, target = data
        else:
            inp, target = data[0], data[1]
        
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if has_frozen:
            output = linear_classifier(inp)
        else:
            output = frozen_features(arch, model, inp, n, avgpool, inp.shape[0])
            output = linear_classifier(output)
        
        print("output shape before cross entropy", output.shape)
        loss = nn.CrossEntropyLoss()(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate_network(has_frozen, arch, val_loader, model, linear_classifier, n, avgpool, batch_size, outputdir):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    output_preds = {'preds': [], 'targets': [], 'outputs': []}
    
    iterator = metric_logger.log_every(val_loader, 20) if isinstance(val_loader, DataLoader) else metric_logger.log_every(val_loader, 20)
    
    for data in iterator:
        if isinstance(val_loader, DataLoader):
            inp, target = data
        else:
            inp, target = data[0], data[1]
        
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if not has_frozen:
            output = frozen_features(arch, model, inp, n, avgpool, batch_size)
            output = linear_classifier(output)
        else:
            output = linear_classifier(inp)
        loss = nn.CrossEntropyLoss()(output, target)
        
        if isinstance(output, torch.Tensor):
            if linear_classifier.module.num_labels >= 5:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
                _, save_pred = output.topk(1, 1, True, True)
                save_pred = save_pred.flatten()
                f1 = f1_score(save_pred.cpu(), target.cpu(), average='weighted')
                precision = precision_score(save_pred.cpu(), target.cpu(), average="weighted")
                recall = recall_score(save_pred.cpu(), target.cpu(), average="weighted")
                balanced_accuracy = balanced_accuracy_score(save_pred.cpu(), target.cpu())
                output_preds['preds'].append(save_pred.cpu().numpy().copy())
                output_preds['targets'].append(target.cpu().numpy().copy())
                output_preds['outputs'].append(output.cpu().numpy().copy())
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
                metric_logger.meters['f1_score'].update(f1, n=batch_size)
                metric_logger.meters['precision'].update(precision, n=batch_size)
                metric_logger.meters['recall'].update(recall, n=batch_size)
                metric_logger.meters['balanced_accuracy'].update(balanced_accuracy, n=batch_size)
            else:
                acc1, = utils.accuracy(output, target, topk=(1,))
                _, save_pred = output.topk(1, 1, True, True)
                save_pred = save_pred.flatten()
                f1 = f1_score(save_pred.cpu(), target.cpu(), average='weighted')
                precision = precision_score(save_pred.cpu(), target.cpu(), average="weighted")
                recall = recall_score(save_pred.cpu(), target.cpu(), average="weighted")
                balanced_accuracy = balanced_accuracy_score(save_pred.cpu(), target.cpu())
                output_preds['preds'].append(save_pred.cpu().numpy().copy())
                output_preds['targets'].append(target.cpu().numpy().copy())
                output_preds['outputs'].append(output.cpu().numpy().copy())
                metric_logger.update(loss=loss.item())
                metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
                metric_logger.meters['f1_score'].update(f1, n=batch_size)
                metric_logger.meters['precision'].update(precision, n=batch_size)
                metric_logger.meters['recall'].update(recall, n=batch_size)
                metric_logger.meters['balanced_accuracy'].update(balanced_accuracy, n=batch_size)
    
    path_save = os.path.join(outputdir, 'test_predictions.pkl')
    output_preds['preds'] = np.concatenate(output_preds['preds']).flatten()
    output_preds['targets'] = np.concatenate(output_preds['targets']).flatten()
    output_preds['outputs'] = np.concatenate(output_preds['outputs'])
    with open(path_save, 'wb') as f:
        pickle.dump(output_preds, f)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether or not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--pretrained_linear_weights', default='', type=str, help="Path to pretrained linear classifier weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--test_batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--train_csv_path', default='/path/to/train/', type=str,
        help='Please specify csv path to the training data.')
    parser.add_argument('--val_csv_path', default='/path/to/train/', type=str,
        help='Please specify csv path to the validation data.')
    parser.add_argument('--test_csv_path', default='/path/to/train/', type=str,
        help='Please specify csv path to the validation data.')
    parser.add_argument('--patch_data_path', default='/path/to/train/', type=str,
        help='Please specify patch data path to the training data.')
    parser.add_argument('--img_size', default=224, type=int,
        help='Please specify image size to the training data.')
    parser.add_argument('--bag_size', default=5, type=int,
        help='Please specify bag of patches size to the training data.')
    parser.add_argument('--max_patches_total', default=300, type=int,
        help='Please specify max patches to the training data.')
    parser.add_argument('--test_bag_size', default=5, type=int,
        help='Please specify bag of patches size to the test data.')
    parser.add_argument('--test_max_patches_total', default=300, type=int,
        help='Please specify max patches to the test data.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--train_from_scratch', default="scratch", help='training from scratch')
    parser.add_argument('--no_aug', action='store_true', help='no augmentation when training the classifier')
    args = parser.parse_args()
    eval_linear(args)


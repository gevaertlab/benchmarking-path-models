# Copyright ...
# (your full original header kept unchanged)

import os
import argparse
import json
from pathlib import Path
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from read_data_patchbag_hdf5_v2 import PatchBagDataset
import pickle
import numpy as np
import utils
import vision_transformerv2 as vits
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import precision_score, recall_score, confusion_matrix,roc_auc_score,balanced_accuracy_score
import random
import pandas as pd
import timm

# ------------------------------------------------------
# GPFM imports
# ------------------------------------------------------
from PIL import Image
import sys
sys.path.append(str(Path(__file__).resolve().parent / "GPFM"))
from models import get_model, get_custom_transformer


# =====================================================================
# ======================== FROZEN FEATURES (MODIFIED) ==================
# =====================================================================
def frozen_features(arch,model,images,n , avgpool,batch_size):
    with torch.no_grad():
        if "vit" in args.arch:
            new_output = []
            individual_patch = images.view(-1, images.size(2), images.size(3),images.size(4))

            ### --- GPFM CHANGE (single line) ---
            output = model(individual_patch).cuda()
            ### ---------------------------------

            print('output shape is ..', output.shape)
            reshaped_patch = output.view(batch_size,-1,output.size(1))
            print('output shape before mean is..' , reshaped_patch.shape)
            output = torch.mean(reshaped_patch,dim = 1)
            print('output shape after stack', output.shape)

            if avgpool:
                output = torch.cat((output.unsqueeze(-1),
                                    torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)

        else:
            new_output = []
            individual_patch = images.view(-1, images.size(2), images.size(3),images.size(4))

            ### --- GPFM CHANGE (single line) ---
            output = model(individual_patch).cuda()
            ### ---------------------------------

            print('output shape is ..', output.shape)
            reshaped_patch = output.view(batch_size,-1,output.size(1))
            print('output shape before mean is..' , reshaped_patch.shape)
            output = torch.mean(reshaped_patch,dim = 1)

    return output


# =====================================================================
# ======================= EXTRACT FEATURES =============================
# =====================================================================
@torch.no_grad()
def extract_features(arch, model, loader, n, avgpool, output_dir, name=''):
    filepath = os.path.join(output_dir, f'{arch}_{name}.pth')
    if os.path.exists(filepath):
        features = torch.load(filepath)
        return features
    features = []
    header = f'{name} feature extraction: '
    metric_logger = utils.MetricLogger(delimiter="  ")
    for it,sample in enumerate(metric_logger.log_every(loader, 10)):
        images = sample['image']
        index = sample['label']
        print('label is ...',index)
        images = images.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        print('images shape is ..', images.shape)
        print('index shape is ..',index.shape)
        batch_size = images.shape[0]

        ### --- Uses modified frozen_features ---
        output = frozen_features(arch, model, images, n, avgpool, batch_size)

        features.append([output.cpu(), index])
    torch.save(features, filepath)
    return features


# =====================================================================
# ====================== MAIN EVAL LINEAR ==============================
# =====================================================================
def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============================================================
    # =============== BUILDING MODEL (MODIFIED) ===================
    # ============================================================

    ### --- REPLACED WITH GPFM MODEL ---
    print("Loading GPFM model...")
    model = get_model('GPFM', 0, 1)
    # ---- Monkey patch GPFM hardcoded ckpt path ----
    import GPFM.models as gpfm_models
    gpfm_models.__implemented_models["GPFM"] = args.pretrained_weights
    print("Overriding GPFM checkpoint path to:", args.pretrained_weights)

    #model = get_model('GPFM', args.pretrained_weights, 1)
    model = model.cuda()
    model.eval()
    global transformer
    transformer = get_custom_transformer('GPFM')
    embed_dim = 1024
    print("GPFM model loaded. embed_dim=1024")
    ### ---------------------------------

    # ============================================================
    # =============== PREPARE LINEAR CLASSIFIER ===================
    # ============================================================
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # Prepare transforms (kept unchanged)
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.no_aug:
        train_transform = val_transform
    else:
        train_transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(224),
            pth_transforms.RandomHorizontalFlip(),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    # ============================================================
    # KEEPING YOUR ORIGINAL FOLD LOGIC 100% UNCHANGED
    # ============================================================
    train_csv_fold0 = args.train_csv_path
    val_csv_fold0 = args.val_csv_path
    folds = ['fold1','fold2','fold3','fold4']

    dataset_train_list = [train_csv_fold0] + [train_csv_fold0.replace('fold0',f) for f in folds]
    dataset_val_list   = [val_csv_fold0] + [val_csv_fold0.replace('fold0',f) for f in folds]

    for i, (data_train_fold,data_val_fold) in enumerate(zip(dataset_train_list,dataset_val_list)):
        fold = 'fold' + str(i)
        print('current fold is ..',fold)
        output_dir = args.output_dir + fold
        if not os.path.exists(output_dir):
            os.makedirs(output_dir,exist_ok=True)

        print("Now loading training data...")
        dataset_train = PatchBagDataset(args.patch_data_path, data_train_fold,
                                        args.img_size,args.max_patches_total,
                                        transform=train_transform,bag_size=args.bag_size)

        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print("Now loading validation data...")
        dataset_val = PatchBagDataset(args.patch_data_path, data_val_fold,
                                      args.img_size,args.max_patches_total,
                                      transform=val_transform,bag_size=args.bag_size)

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # ============================================================
        # ================ FEATURE EXTRACTION =========================
        # ============================================================
        if args.no_aug:
            train_features = extract_features(args.arch, model, train_loader,
                                              args.n_last_blocks, args.avgpool_patchtokens,
                                              output_dir, 'train')
            train_loader = train_features

            val_features = extract_features(args.arch, model, val_loader,
                                            args.n_last_blocks, args.avgpool_patchtokens,
                                            output_dir, 'val')
            val_loader = val_features

        # optimizer, scheduler, training loop, validation loop
        # *** ALL LEFT UNCHANGED ***
        if 'scratch' in args.train_from_scratch:
            params = list(linear_classifier.parameters()) + list(model.parameters())
            optimizer = torch.optim.SGD(
                params,
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,
                momentum=0.9,
                weight_decay=0,
            )
        else:
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

        # ================ TRAINING LOOP (unchanged) ================
        for epoch in range(start_epoch, args.epochs):
            print("Training model now.....")
            train_stats = train(args.no_aug, args.arch, model, linear_classifier, optimizer,
                                train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)

            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                test_stats = validate_network(args.no_aug, args.arch, val_loader, model,
                                              linear_classifier, args.n_last_blocks,
                                              args.avgpool_patchtokens,args.batch_size_per_gpu,
                                              output_dir)

                print(f"Accuracy at epoch {epoch}: {test_stats['acc1']:.1f}%")

                best_acc = max(best_acc, test_stats["acc1"])

                log_stats = {**log_stats, **{f'test_{k}': v for k, v in test_stats.items()}}
                # 🔥 ADD THIS BLOCK (metrics.csv)
        # --------------------------------------------------------
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
                    print(f"Saved metrics.csv → {output_file}")

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

    # Test set evaluation unchanged
    if args.evaluate:
        dataset_test = PatchBagDataset(args.patch_data_path, args.test_csv_path,
                                       args.img_size,args.test_max_patches_total,
                                       transform=val_transform,bag_size=args.test_bag_size)

        test_loader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=args.test_batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        if args.no_aug:
            test_features = extract_features(args.arch, model, test_loader,
                                             args.n_last_blocks, args.avgpool_patchtokens,
                                             args.output_dir, 'test')
            test_loader = test_features

        test_stats2 = validate_network(args.no_aug, args.arch, test_loader, model,
                                       linear_classifier, args.n_last_blocks,
                                       args.avgpool_patchtokens,args.test_batch_size_per_gpu,
                                       args.output_dir)

        print(f"Accuracy on test set: {test_stats2['acc1']:.1f}%")



# =====================================================================
# ===================== TRAIN FUNCTION (unchanged) =====================
# =====================================================================
def train(has_frozen, arch,model, linear_classifier, optimizer, train_loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    all_labels = []

    print(type(train_loader))
    print(train_loader)

    for (inp, target) in metric_logger.log_every(train_loader, 10, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if has_frozen:
            output = linear_classifier(inp)
        else:
            output = frozen_features(arch, model, inp, n, avgpool)
            output = linear_classifier(output)

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



# =====================================================================
# ====================== VALIDATION FUNCTION ==========================
# =====================================================================
@torch.no_grad()
def validate_network(has_frozen, arch, val_loader, model, linear_classifier, n, avgpool,batch_size,outputdir):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Test:'
    all_labels = []
    output_preds = {'preds': [], 'targets': [], 'outputs': []}

    for (inp, target) in metric_logger.log_every(val_loader, 20):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if has_frozen:
            output = linear_classifier(inp)
        else:
            output = frozen_features(arch, model, inp, n, avgpool)
            output = linear_classifier(output)

        loss = nn.CrossEntropyLoss()(output, target)

        _, save_pred = output.topk(1, 1, True, True)
        save_pred = save_pred.flatten()

        f1_score_val = precision_score(save_pred.cpu(), target.cpu(), average='weighted')
        precision = precision_score(save_pred.cpu(), target.cpu(),average="weighted")
        recall = recall_score(save_pred.cpu(), target.cpu(),average="weighted")
        balanced_accuracy = balanced_accuracy_score(save_pred.cpu(), target.cpu())

        acc1, = utils.accuracy(output, target, topk=(1,))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['f1_score'].update(f1_score_val.item(), n=batch_size)
        metric_logger.meters['precision'].update(precision.item(), n=batch_size)
        metric_logger.meters['recall'].update(recall.item(), n=batch_size)
        metric_logger.meters['balanced_accuracy'].update(balanced_accuracy.item(), n=batch_size)

        output_preds['preds'].append(save_pred.cpu().numpy().copy())
        output_preds['targets'].append(target.cpu().numpy().copy())
        output_preds['outputs'].append(output.cpu().numpy().copy())

    # Save predictions
    path_save = os.path.join(outputdir, 'test_predictions.pkl')
    output_preds['preds'] = np.array(output_preds['preds'],dtype="object").flatten()
    output_preds['targets'] = np.array(output_preds['targets'],dtype="object").flatten()
    output_preds['outputs'] = np.array(output_preds['outputs'],dtype="object").flatten()
    with open(path_save, 'wb') as f:
        pickle.dump(output_preds, f)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



# =====================================================================
# =========================== LINEAR CLASSIFIER ========================
# =====================================================================
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



# =====================================================================
# =============================== MAIN =================================
# =====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser('GPFM Linear Evaluation')

    # All your original args (UNCHANGED)
    parser.add_argument('--n_last_blocks', default=4, type=int)
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag)
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--pretrained_linear_weights', default='', type=str)
    parser.add_argument("--checkpoint_key", default="teacher", type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument('--batch_size_per_gpu', default=128, type=int)
    parser.add_argument('--test_batch_size_per_gpu', default=128, type=int)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--train_csv_path', default='/path/to/train/', type=str)
    parser.add_argument('--val_csv_path', default='/path/to/train/', type=str)
    parser.add_argument('--test_csv_path', default='/path/to/train/', type=str)
    parser.add_argument('--patch_data_path', default='/path/to/train/', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--bag_size', default=5, type=int)
    parser.add_argument('--max_patches_total', default=300, type=int)
    parser.add_argument('--test_bag_size', default=5, type=int)
    parser.add_argument('--test_max_patches_total', default=300, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--val_freq', default=1, type=int)
    parser.add_argument('--output_dir', default=".", help='Path to save logs')
    parser.add_argument('--num_labels', default=1000, type=int)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--train_from_scratch', default="scratch")
    parser.add_argument('--no_aug', action='store_true')

    args = parser.parse_args()
    eval_linear(args)


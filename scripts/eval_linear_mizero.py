#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

#from timm.models.layers.helpers import to_2tuple
import timm

from read_data_patchbag_hdf5_v2 import PatchBagDataset
import utils
import vision_transformerv2 as vits  # keep for vit_* support

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
)

# --------------------------------------------------------
# CTransPath backbone definition
# --------------------------------------------------------

def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


class ConvStem(nn.Module):
    """
    Conv stem used in CTransPath, from the original code.

    It replaces the standard patch embedding with a stack of convs.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        **kwargs):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(
                nn.Conv2d(
                    input_dim,
                    output_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # (B, embed_dim, H', W')
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath(pretrained=False):
    """
    CTransPath = Swin-T backbone with ConvStem as patch embedding.
    """
    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        embed_layer=ConvStem,
        pretrained=False,
    )
    # 2. Force replace patch_embed manually (critical!)
    model.patch_embed = ConvStem(
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=model.embed_dim,
        norm_layer=nn.LayerNorm,
        flatten=True,
    )
    # 3. Replace head for feature extraction
    model.head = nn.Identity()

    return model


# --------------------------------------------------------
# Feature extraction helpers (same structure as your code)
# --------------------------------------------------------


def frozen_features(arch, model, images, n, avgpool, batch_size):
    """
    images: [B, bag_size, C, H, W]
    model: backbone (DINO/ViT/ResNet/CTransPath/etc.)
    Returns: [B, embed_dim] (bag-level pooled features)
    """
    with torch.no_grad():
        # flatten bags -> [B * bag_size, C, H, W]
        individual_patch = images.view(
            -1, images.size(2), images.size(3), images.size(4)
        )

        # forward through backbone
        output = model(individual_patch)
        # For CTransPath (Swin), output is already [B*bag, L, C] or [B*bag, C]
        # timm swin_tiny returns [B*bag, C] by default (pooled features)
        # For ViT-like models with CLS output, same: [B*bag, C]

        # If model returns [B*bag, L, C], you might need to CLS-pool here.
        # For safety, handle both cases:
        if output.ndim == 3:
            # assume [B*bag, L, C] -> take CLS token (index 0) or mean over tokens
            # Here we take mean over tokens:
            output = output.mean(dim=1)

        print("output shape is ..", output.shape)

        # reshape back into bags: [B, bag_size, C]
        reshaped_patch = output.view(batch_size, -1, output.size(1))
        print("output shape before mean is..", reshaped_patch.shape)

        # mean pool across patches -> [B, C]
        output = torch.mean(reshaped_patch, dim=1)
        print("output shape after stack", output.shape)

        if avgpool:
            # keeping your original API; if you ever pass avgpool=True
            # you can customize extra pooling here.
            pass

    return output


@torch.no_grad()
def extract_features(arch, model, loader, n, avgpool, output_dir, name=""):
    filepath = os.path.join(output_dir, f"{arch}_{name}.pth")
    if os.path.exists(filepath):
        features = torch.load(filepath)
        return features

    features = []
    header = f"{name} feature extraction: "
    metric_logger = utils.MetricLogger(delimiter="  ")
    for it, sample in enumerate(metric_logger.log_every(loader, 10, header)):
        images = sample["image"]
        index = sample["label"]  # labels
        print("label is ...", index)

        images = images.cuda(non_blocking=True)
        index = index.cuda(non_blocking=True)
        print("images shape is ..", images.shape)
        print("index shape is ..", index.shape)

        batch_size = images.shape[0]

        output = frozen_features(arch, model, images, n, avgpool, batch_size)
        features.append([output.cpu(), index])

    torch.save(features, filepath)
    return features


# --------------------------------------------------------
# Linear eval training & validation
# --------------------------------------------------------


def train(has_frozen, arch, model, linear_classifier, optimizer, train_loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for inp, target in metric_logger.log_every(train_loader, 10, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if has_frozen:
            output = linear_classifier(inp)
        else:
            output = frozen_features(arch, model, inp, n, avgpool, inp.size(0))
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
def validate_network(
    has_frozen,
    arch,
    val_loader,
    model,
    linear_classifier,
    n,
    avgpool,
    batch_size,
    outputdir,
):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"
    output_preds = {"preds": [], "targets": [], "outputs": []}

    for inp, target in metric_logger.log_every(val_loader, 20, header):
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        if has_frozen:
            output = linear_classifier(inp)
        else:
            output = frozen_features(arch, model, inp, n, avgpool, inp.size(0))
            output = linear_classifier(output)

        loss = nn.CrossEntropyLoss()(output, target)

        # accuracy and metrics
        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        _, save_pred = output.topk(1, 1, True, True)
        save_pred = save_pred.flatten()

        f1 = f1_score(save_pred.cpu(), target.cpu(), average="weighted")
        precision = precision_score(save_pred.cpu(), target.cpu(), average="weighted")
        recall = recall_score(save_pred.cpu(), target.cpu(), average="weighted")
        balanced_acc = balanced_accuracy_score(save_pred.cpu(), target.cpu())

        output_preds["preds"].append(save_pred.cpu().numpy().copy())
        output_preds["targets"].append(target.cpu().numpy().copy())
        output_preds["outputs"].append(output.cpu().numpy().copy())

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["f1_score"].update(f1, n=batch_size)
        metric_logger.meters["precision"].update(precision, n=batch_size)
        metric_logger.meters["recall"].update(recall, n=batch_size)
        metric_logger.meters["balanced_accuracy"].update(balanced_acc, n=batch_size)

        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if linear_classifier.module.num_labels >= 5:
        print(
            "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} "
            "f1score {f1_scores.global_avg:.3f} precision_val {precision_scores.global_avg:.3f} "
            "recall_val {recall_scores.global_avg:.3f} balanced_accuracy_val {balanced_accuracy_scores.global_avg:.3f} "
            "loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1,
                top5=metric_logger.acc5,
                f1_scores=metric_logger.f1_score,
                precision_scores=metric_logger.precision,
                recall_scores=metric_logger.recall,
                balanced_accuracy_scores=metric_logger.balanced_accuracy,
                losses=metric_logger.loss,
            )
        )
    else:
        print(
            "* Acc@1 {top1.global_avg:.3f} f1score {f1_scores.global_avg:.3f} "
            "precision_val {precision_scores.global_avg:.3f} recall_val {recall_scores.global_avg:.3f} "
            "balanced_accuracy_val {balanced_accuracy_scores.global_avg:.3f} "
            "loss {losses.global_avg:.3f}".format(
                top1=metric_logger.acc1,
                f1_scores=metric_logger.f1_score,
                precision_scores=metric_logger.precision,
                recall_scores=metric_logger.recall,
                balanced_accuracy_scores=metric_logger.balanced_accuracy,
                losses=metric_logger.loss,
            )
        )

    # save preds
    path_save = os.path.join(outputdir, "test_predictions.pkl")
    output_preds["preds"] = np.array(output_preds["preds"], dtype="object").flatten()
    output_preds["targets"] = np.array(output_preds["targets"], dtype="object").flatten()
    output_preds["outputs"] = np.array(output_preds["outputs"], dtype="object").flatten()
    with open(path_save, "wb") as f:
        pickle.dump(output_preds, f)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super().__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)


# --------------------------------------------------------
# Main eval loop with CTransPath support
# --------------------------------------------------------


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join(f"{k}: {v}" for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ----------------- build backbone -----------------
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (
            args.n_last_blocks + int(args.avgpool_patchtokens)
        )

    elif "xcit" in args.arch:
        model = torch.hub.load(
            "facebookresearch/xcit:main", args.arch, num_classes=0
        )
        embed_dim = model.embed_dim

    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    elif args.arch == "ctranspath":
        model = ctranspath(pretrained=False)

        if hasattr(model, "head"):
            try:
                embed_dim = model.head.in_features
            except AttributeError:
                embed_dim = model.num_features

            print(f"[ctranspath] Removing head: {type(model.head)}")
            model.head = nn.Identity()
        else:
            raise ValueError("CTransPath has no .head")

        model.cuda()
        model.eval()

    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)

    model.cuda()

    # training mode selection
    if "scratch" in args.train_from_scratch:
        print("training from scratch...")
        model.train()

    elif "pretrain" in args.train_from_scratch:
        print("training from pretrained weights...")
        model.eval()

        if args.arch == "ctranspath":
            state = torch.load(args.pretrained_weights, map_location="cpu")["state_dict"]
            state = {k.replace("module.visual.trunk.", ""): v for k, v in state.items() if "visual.trunk." in k and "attn_mask" not in k}
            model.load_state_dict(state, strict=False)
        else:
            utils.load_pretrained_weights(
                model,
                args.pretrained_weights,
                args.checkpoint_key,
                args.arch,
                args.patch_size,
            )
    else:
        print(f"Unsupported train_from_scratch mode: {args.train_from_scratch}")
        sys.exit(1)

    print(f"Model {args.arch} built. embed_dim={embed_dim}")

    # ----------------- data transforms -----------------
    val_transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )

    if args.no_aug:
        train_transform = val_transform
    else:
        train_transform = pth_transforms.Compose(
            [
                pth_transforms.RandomResizedCrop(224),
                pth_transforms.RandomHorizontalFlip(),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    print("embed_dim is..", embed_dim)
    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(
        linear_classifier, device_ids=[args.gpu]
    )

    # ----------------- folds setup (same structure) -----------------
    train_csv_fold0 = args.train_csv_path
    val_csv_fold0 = args.val_csv_path
    folds = ["fold1", "fold2", "fold3", "fold4"]

    dataset_train_list = [train_csv_fold0]
    for f in folds:
        dataset_train_list.append(train_csv_fold0.replace("fold0", f))

    dataset_val_list = [val_csv_fold0]
    for f in folds:
        dataset_val_list.append(val_csv_fold0.replace("fold0", f))

    print("dataset_train_list is ,", dataset_train_list)
    print("dataset_val_list is ,", dataset_val_list)

    for i, (data_train_fold, data_val_fold) in enumerate(
        zip(dataset_train_list, dataset_val_list)
    ):
        fold = "fold" + str(i)
        print("current fold is ..", fold)
        output_dir = args.output_dir + fold
        os.makedirs(output_dir, exist_ok=True)

        print("Now loading training data...")
        dataset_train = PatchBagDataset(
            args.patch_data_path,
            data_train_fold,
            args.img_size,
            args.max_patches_total,
            transform=train_transform,
            bag_size=args.bag_size,
        )
        print(len(dataset_train))

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train
        )
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print("Now loading validation data...")
        dataset_val = PatchBagDataset(
            args.patch_data_path,
            data_val_fold,
            args.img_size,
            args.max_patches_total,
            transform=val_transform,
            bag_size=args.bag_size,
        )
        print(len(dataset_val))

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print("len of val loader is ", len(val_loader))
        print(
            f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs."
        )

        print("Now extracting features....")
        if args.no_aug:
            train_features = extract_features(
                args.arch,
                model,
                train_loader,
                args.n_last_blocks,
                args.avgpool_patchtokens,
                output_dir,
                "train",
            )
            train_loader = train_features

            val_features = extract_features(
                args.arch,
                model,
                val_loader,
                args.n_last_blocks,
                args.avgpool_patchtokens,
                output_dir,
                "val",
            )
            val_loader = val_features

            print("Features are ready!\nStarting to train the linear classifier.")

        # optimizer
        if "scratch" in args.train_from_scratch:
            print("optimizers from scratch...")
            start = time.time()
            params = list(linear_classifier.parameters()) + list(model.parameters())
            optimizer = torch.optim.SGD(
                params,
                args.lr
                * (args.batch_size_per_gpu * utils.get_world_size())
                / 256.0,
                momentum=0.9,
                weight_decay=0,
            )
        else:
            print("optimizers from pretrained weights...")
            start = time.time()
            optimizer = torch.optim.SGD(
                linear_classifier.parameters(),
                args.lr
                * (args.batch_size_per_gpu * utils.get_world_size())
                / 256.0,
                momentum=0.9,
                weight_decay=0,
            )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0
        )

        # resume
        to_restore = {"epoch": 0, "best_acc": 0.0}
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
            print("Training model now.....")
            start_time_for_run = time.time()

            train_stats = train(
                args.no_aug,
                args.arch,
                model,
                linear_classifier,
                optimizer,
                train_loader,
                epoch,
                args.n_last_blocks,
                args.avgpool_patchtokens,
            )
            scheduler.step()

            log_stats = {f"train_{k}": v for k, v in train_stats.items()}
            log_stats["epoch"] = epoch

            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                test_stats = validate_network(
                    args.no_aug,
                    args.arch,
                    val_loader,
                    model,
                    linear_classifier,
                    args.n_last_blocks,
                    args.avgpool_patchtokens,
                    args.batch_size_per_gpu,
                    output_dir,
                )
                print(
                    f"Accuracy at epoch {epoch} on {len(dataset_val)} "
                    f"val images for {fold}: {test_stats['acc1']:.1f}%"
                )
                print(
                    f"f1 score at epoch {epoch} on {len(dataset_val)} "
                    f"val images for {fold}: {test_stats['f1_score']:.1f}%"
                )
                print(
                    f"precision at epoch {epoch} on {len(dataset_val)} "
                    f"val images for {fold}: {test_stats['precision']:.1f}%"
                )
                print(
                    f"recall at epoch {epoch} on {len(dataset_val)} "
                    f"val images for {fold}: {test_stats['recall']:.1f}%"
                )
                print(
                    f"balanced_accuracy at epoch {epoch} on {len(dataset_val)} "
                    f"val images for {fold}: {test_stats['balanced_accuracy']:.1f}%"
                )
                best_acc = max(best_acc, test_stats["acc1"])

                if epoch == args.epochs - 1:
                    df = pd.DataFrame(
                        {
                            "Accuracy": [round(test_stats["acc1"], 2)],
                            "F1_score": [round(test_stats["f1_score"], 2)],
                            "Precision": [round(test_stats["precision"], 2)],
                            "Recall": [round(test_stats["recall"], 2)],
                            "balanced_accuracy": [
                                round(test_stats["balanced_accuracy"], 2)
                            ],
                        }
                    )
                    output_file = os.path.join(output_dir, "metrics.csv")
                    df.to_csv(output_file, encoding="utf-8", index=False)

                print(f"Max accuracy so far: {best_acc:.2f}%")
                log_stats.update({f"test_{k}": v for k, v in test_stats.items()})

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

        print(
            "Training of the supervised linear classifier on frozen features completed.\n"
            f"Top-1 test accuracy: {best_acc:.1f}"
        )
        end = time.time()
        print("time taken for validation is ", (end - start) / 60, "minutes")

    # optional test-set evaluation hook (if you keep args.evaluate etc.)
    #if args.evaluate:
        # ------------------- TEST SET EVALUATION -------------------
    if args.evaluate:
        print("\nRunning FINAL TEST evaluation...")

        output_dir = args.output_dir + "/test"
        os.makedirs(output_dir, exist_ok=True)

        # Load test dataset
        test_dataset = PatchBagDataset(
            args.patch_data_path,
            args.test_csv_path,
            args.img_size,
            args.test_max_patches_total,
            transform=val_transform,
            bag_size=args.test_bag_size,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        print(f"Loaded {len(test_dataset)} test samples.")

        # Extract features if NO AUG
        if args.no_aug:
            test_features = extract_features(
                args.arch,
                model,
                test_loader,
                args.n_last_blocks,
                args.avgpool_patchtokens,
                output_dir,
                "test",
            )
            test_loader = test_features

        # ---- Run eval ----
        test_stats = validate_network(
            args.no_aug,
            args.arch,
            test_loader,
            model,
            linear_classifier,
            args.n_last_blocks,
            args.avgpool_patchtokens,
            args.test_batch_size_per_gpu,
            output_dir,
        )

        # ---- Save csv ----
        df = pd.DataFrame(
            {
                "Accuracy": [round(test_stats["acc1"], 2)],
                "F1_score": [round(test_stats["f1_score"], 2)],
                "Precision": [round(test_stats["precision"], 2)],
                "Recall": [round(test_stats["recall"], 2)],
                "balanced_accuracy": [round(test_stats["balanced_accuracy"], 2)],
            }
        )
        df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

        print("\nSaved test metrics:", os.path.join(output_dir, "test_metrics.csv"))
        print("Saved test predictions:", os.path.join(output_dir, "test_predictions.pkl"))

        # save test feature pth
        torch.save(model.state_dict(), os.path.join(output_dir, f"{args.arch}_test.pth"))
        print("Saved:", os.path.join(output_dir, f"{args.arch}_test.pth"))

        print("\nTEST evaluation done!\n")

        # you can plug in your test csv + loader here similarly
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Evaluation with linear classification on CTransPath / ViT backbones"
    )
    parser.add_argument(
        "--n_last_blocks",
        default=4,
        type=int,
        help="""Concatenate [CLS] tokens for the `n` last blocks.""",
    )
    parser.add_argument(
        "--avgpool_patchtokens",
        default=False,
        type=utils.bool_flag,
        help="Whether to concatenate global average pooled features.",
    )
    parser.add_argument(
        "--arch", default="ctranspath", type=str, help="Backbone architecture"
    )
    parser.add_argument(
        "--patch_size", default=16, type=int, help="Patch resolution of the model."
    )
    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to evaluate.",
    )
    parser.add_argument(
        "--pretrained_linear_weights",
        default="",
        type=str,
        help="Path to pretrained linear classifier weights to evaluate.",
    )
    parser.add_argument(
        "--checkpoint_key",
        default="teacher",
        type=str,
        help='Key in checkpoint (e.g. "teacher")',
    )
    parser.add_argument(
        "--epochs", default=10, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="Base learning rate for batch size 256.",
    )
    parser.add_argument(
        "--batch_size_per_gpu", default=128, type=int, help="Per-GPU batch size"
    )
    parser.add_argument(
        "--test_batch_size_per_gpu", default=128, type=int, help="Per-GPU test batch size"
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="URL used to set up distributed training",
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="Please ignore and do not set."
    )
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/",
        type=str,
    )
    parser.add_argument(
        "--train_csv_path",
        default="/path/to/train_fold0.csv",
        type=str,
        help="CSV path to training data fold0.",
    )
    parser.add_argument(
        "--val_csv_path",
        default="/path/to/val_fold0.csv",
        type=str,
        help="CSV path to validation data fold0.",
    )
    parser.add_argument(
        "--test_csv_path",
        default="/path/to/test.csv",
        type=str,
        help="CSV path to test data.",
    )
    parser.add_argument(
        "--patch_data_path",
        default="/path/to/patches/",
        type=str,
        help="Path to patch HDF5/NPY data.",
    )
    parser.add_argument(
        "--img_size",
        default=256,
        type=int,
        help="Patch image size.",
    )
    parser.add_argument(
        "--bag_size",
        default=5,
        type=int,
        help="Number of patches per bag.",
    )
    parser.add_argument(
        "--max_patches_total",
        default=300,
        type=int,
        help="Max patches for training data.",
    )
    parser.add_argument(
        "--test_bag_size",
        default=5,
        type=int,
        help="Bag size for test data.",
    )
    parser.add_argument(
        "--test_max_patches_total",
        default=300,
        type=int,
        help="Max patches for test data.",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--val_freq", default=1, type=int, help="Epoch frequency for validation."
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument(
        "--num_labels", default=1000, type=int, help="Number of labels for classifier."
    )
    parser.add_argument(
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="Evaluate model on test set.",
    )
    parser.add_argument(
        "--train_from_scratch",
        default="pretrain",
        help='"scratch" or "pretrain"',
    )
    parser.add_argument(
        "--no_aug",
        action="store_true",
        help="No augmentation when training the classifier.",
    )

    args = parser.parse_args()
    eval_linear(args)


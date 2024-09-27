import numpy as np
import torch
import torch.nn as nn
import numpy
import yaml

import wandb
from utils.metrics import calculate_iou, calculate_ap_for_segmentation
from utils.transformation import transform_mask, transform_batch

# augmentacja z artykułu
from utils.better_aug import BetterAugmentation


"""config = {"epochs": 200,
            "batch_size": 6, # zmieniam na 2 z 6, żeby zobaczyć co się stanie
            "lr": 1e-3,
            "annotator": 1, # zmieniam na jeden
            "model": 'smpUNet++',
            "augmentation": False,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "place": "lab",
            "mode": "multiclass", # zmieniam z normal na multiclass
            "aug_type": "BetterAugmentation",
            "k": 5
          } # TODO WPISAĆ DO YAMLA"""
# będzie to updatowane do yamla w mainie i wykorzystywane w calym pliku !!!


def train(model: torch.nn, train_loader, optimizer, scheduler, loss_fn, augmentation, T_aug, epoch_number, device: torch.device):
    with open("train_config.yaml","r") as file:
        config = yaml.safe_load(file)
        file.close()

    model.eval()

    total_loss = 0
    total_loss_oneclass = 0
    total_loss_multiclass = 0

    total_iou_oneclass = 0
    total_iou_multiclass = 0

    for i, data in enumerate(train_loader):
        if len(data) == 2:
            inputs, labels = data
            if (i+1) == 1 or (i+1) % (4 * inputs.size(0)) == 0:
                print(f"training {(i+1)*inputs.size(0)} / {len(train_loader.dataset)}")

        elif len(data) == 6:
            raise NotImplementedError

        if T_aug == True:
            for i in inputs.shape[0]:
                if len(data) == 2:
                    inputs[i], labels[i] = augmentation(inputs[i], labels[i])

        labels = labels.to(device)
        if len(data) == 2:
            if config["loss"] == "CrossEntropyLoss" or loss_fn == nn.CrossEntropyLoss:
                labels = labels.type(torch.LongTensor)
            if config["loss"] == "BCEWithLogitsLoss" or loss_fn == nn.BCEWithLogitsLoss:
                labels = labels.type(torch.FloatTensor)
            labelsBCE = labels[:, [0, -1], :, :]
            labelsBCE = labelsBCE.to(device)


        labels1 = transform_batch(labels.cpu(),1)
        labels2 = transform_batch(labels.cpu(), 2)

        labels1 = torch.from_numpy(labels1).type(torch.LongTensor).to(device)
        labels2 = torch.from_numpy(labels2).type(torch.LongTensor).to(device)

        optimizer.zero_grad()

        inputs = inputs.to(device)

        output = model(inputs)
        output1 = output[:, :3, :, :]
        output2 = output[:, [0, -1], :, :]

        weights1 = torch.tensor([0.2, 1.0, 0.5]).to(device)
        weights2 = torch.tensor([1.0, 1.0]).to(device)
        loss_fn1 = nn.CrossEntropyLoss(weight=weights1)
        loss_fn2 = nn.CrossEntropyLoss(weight=weights2)

        # Loss calculation

        if config["mode"] == "multiclass":
            if config["loss"] == "CrossEntropyLoss":
                l1 = loss_fn1(output1, labels1)
                l2 = loss_fn2(output2, labels2)
                loss = l1 + l2
            if config["loss"] == "BCEWithLogitsLoss":
                loss = loss_fn(output, labels)
        elif config["mode"] == "oneclass":
            raise NotImplementedError
        elif config["mode"] == "intersection_and_union":
            raise NotImplementedError

        if config["mode"] == "oneclass":
            raise NotImplementedError
        elif config["mode"] == "multiclass" or config["mode"] == "intersection_and_union":
            preds1 = torch.argmax(output1, dim=1)
            preds2 = torch.argmax(output2, dim=1)

            mean_iou, IoUs = calculate_iou(labels1.cpu().numpy(), preds1.cpu().numpy(), 3)
            iou_multiclass = 1 - mean_iou
            mean_iou, IoUs = calculate_iou(labels2.cpu().numpy(), preds2.cpu().numpy(), 2)
            iou_oneclass = 1 - mean_iou

            loss.backward()
            optimizer.step()

            total_iou_multiclass += iou_multiclass
            total_iou_oneclass += iou_oneclass
            total_loss += loss.item()

            if config["loss"] == "CrossEntropyLoss":
                total_loss_oneclass += l2.item()
                total_loss_multiclass += l1.item()

    avg_loss = total_loss / len(train_loader)
    avg_loss_oneclass = total_loss_oneclass / len(train_loader)
    avg_loss_multiclass = total_loss_multiclass / len(train_loader)
    avg_iou_multiclass = total_iou_multiclass / len(train_loader)
    avg_iou_oneclass = total_iou_oneclass / len(train_loader)


    if config["scheduler"] != "ReduceLROnPlateau":
        scheduler.step()

    metrics = {"train/train_loss": avg_loss,
               "train/train_loss_oneclass": avg_loss_oneclass,
               "train/train_loss_multiclass": avg_loss_multiclass,
               "train/train_iou_multiclass": avg_iou_multiclass,
               "train/lr": optimizer.param_groups[0]["lr"],
               "train/epoch": epoch_number}

    wandb.log(metrics)

    return avg_loss, avg_iou_multiclass, avg_iou_oneclass

    return

# z tutorialów torcha

"""def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
"""
import numpy as np
import torch
import yaml

import wandb
import random
import matplotlib.pyplot as plt

from train import transform_batch
from utils.metrics import calculate_iou, calculate_ap_for_segmentation

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
          } # TODO WPISAĆ DO YAMLA
"""

def plot_results(X, y, preds, index=None, mode="train", number="1"):
    colors = [[0, 0, 0], [0, 255, 0], [255, 0, 0]]  # tło, wić, główka

    # jeżeli nie sprecyzowano indeksu wybierz losowy
    if index is None:
        index = random.randint(0, len(X))

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))

    ax[0].set_title("Sperm image")
    ax[0].set_axis_off()

    mask = y[index]
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[mask == i] = color

    ax[1].imshow(mask_rgb)
    ax[1].set_title("Sperm mask image")

    pred_mask = preds[index]

    mask_rgb = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        mask_rgb[pred_mask == i] = color

    ax[2].imshow(mask_rgb)
    ax[2].set_title("Prediction image prediction")
    ax[2].set_axis_off()

    if mode == "train":
        wandb.log({"train plot": wandb.Image(fig)})
    if mode == "val":
        if number == "1":
            wandb.log({"val/plot": wandb.Image(fig)})
        else:
            wandb.log({"val/plot" + number: wandb.Image(fig)})
    plt.close()

def val(model: torch.nn, validation_loader, loss_fn, epoch_number, scheduler, device: torch.device):
    model.eval()

    with open("train_config.yaml","r") as file:
        config = yaml.safe_load(file)


    softmask_oneclass_list = []
    softmask_multiclass_list = []
    vids_list = []
    total_iou_oneclass = 0
    total_iou_multiclass = 0

    with torch.no_grad():
        for i, val_data in enumerate(validation_loader):
            if len(val_data) == 2:
                inputs, labels = val_data
                labels_y1 = labels.type(torch.FloatTensor)
                if (i + 1) == 1 or (i + 1) % (4 * inputs.size(0)) == 0:
                    print(f"validation {(i + 1) * inputs.size(0)} / {len(validation_loader.dataset)}")
            elif len(val_data) == 6:
                raise NotImplementedError
            
            inputs = inputs.to(device)
            images = inputs.detach().cpu().numpy().transpose(0, 2, 3, 1)
            if config["annotator"] == 1:
                labels = labels_y1.to(device)
            elif config["annotator"] == 2:
                #vids = vids_y2.to(device)
                raise NotImplementedError
            elif config["annotator"] == 4:
                #vids = vintersections.to(device)
                raise NotImplementedError
            elif config["annotator"] == 5:
                #vids = vunions.to(device)
                raise NotImplementedError

            if config["mode"] == "oneclass":
                raise NotImplementedError
            else:
                vids_list.append(labels.cpu())
                labels1 = transform_batch(labels.cpu(), 1)
                labels2 = transform_batch(labels.cpu(), 2)

                labels1 = torch.from_numpy(labels1).type(torch.LongTensor).to(device)
                labels2 = torch.from_numpy(labels2).type(torch.LongTensor).to(device)

                labels_numpy1 = labels1.detach().cpu().numpy() # dlaczego przenosimy to do cpu ?
                labels_numpy2 = labels2.detach().cpu().numpy()

                inputs = inputs.to(device)
                outputs = model(inputs)
                output1 = outputs[:, :3, :, :]
                output2 = outputs[:, [0, -1], :, :]

                preds1 = torch.argmax(output1, dim=1)
                preds2 = torch.argmax(output2, dim=1)

                # czym są softmaski # TODO

                softs1 = torch.softmax(output1, dim=1)
                softs2 = torch.softmax(output2, dim=1)

                softs1 = softs1.squeeze(0)
                softs2 = softs2.squeeze(0)

                softmask_multiclass_list.append(softs1.cpu())
                softmask_oneclass_list.append(softs2.cpu())

                preds_out_multiclass = preds1.detach().cpu().numpy()
                preds_out_oneclass = preds2.detach().cpu().numpy()

                mean_iou, IoUs = calculate_iou(labels1.cpu().numpy(),
                                               preds1.cpu().numpy(),3)

                iou = 1 - mean_iou
                total_iou_oneclass += iou

    vids_list = np.concatenate(vids_list, axis=0)
    ids1 = transform_batch(vids_list, 1)
    ids2 = transform_batch(vids_list, 2)

    softmasks_oneclass = [mask for batch in softmask_oneclass_list for mask in batch] # spłaszcza listę
    softmasks_oneclass_np = np.array(softmasks_oneclass)
    softmasks_oneclass_np = softmasks_oneclass_np.transpose(0, 2, 3, 1)
    ap_score_oneclass = calculate_ap_for_segmentation(softmasks_oneclass_np[:, :, :, 1], ids2)
    ap_score_head = 0
    ap_score_tail = 0

    if config["mode"] == "multiclass" or config["mode"] == "intersection_and_union":
        softmasks_multiclass = [mask for batch in softmask_multiclass_list for mask in batch]
        softmasks_multiclass_np = np.array(softmasks_multiclass)
        softmasks_multiclass_np  = softmasks_multiclass_np.transpose(0, 2, 3, 1)
        ap_score_head = calculate_ap_for_segmentation(softmasks_multiclass_np[:, :, :, 2],
                                                      vids_list[:, 2, :, :])
        ap_score_tail = calculate_ap_for_segmentation(softmasks_multiclass_np[:, :, :, 1],
                                                      vids_list[:, 1, :, :])

    avg_iou_multiclass = total_iou_multiclass / len(validation_loader)
    avg_iou_oneclass = total_iou_oneclass / len(validation_loader)

    if config["scheduler"] == "RediceLROnPlateau":
        scheduler.step(avg_iou_multiclass)

    val_metrics = {"val/val_iou_multiclass": avg_iou_multiclass,
                   "val/val_iou_oneclass": avg_iou_oneclass,
                   "val/val_ap_oneclass": ap_score_oneclass,
                   "val/val_ap_head": ap_score_head,
                   "val/val_ap_tail": ap_score_tail,
                   "val/epoch": epoch_number}

    wandb.log(val_metrics)

    return avg_iou_multiclass, avg_iou_oneclass, images, labels_numpy1, labels_numpy2, preds_out_multiclass, preds_out_oneclass, ap_score_oneclass, ap_score_head, ap_score_tail

"""
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss
"""
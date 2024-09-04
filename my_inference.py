import os.path

import segmentation_models_pytorch as smp
import torch
import yaml
import numpy as np
import wandb
import cv2
import pandas as pd

from utils.transformation import transform_batch, transform_mask
from utils.metrics import calculate_iou, calculate_iou_per_class, calculate_ap_for_segmentation
from data import BatchMaker


def inference():
    return


def calculate_optimal_threshold(x_images, masks, softmask_multiclass, softmask_oneclass):

    labels1 = transform_batch(masks, 1)
    labels2 = transform_batch(masks, 2)
    print("labels1", np.shape(labels1))
    print("labels2", np.shape(labels2))

    #thresholds = torch.linspace(0, 1, steps=255)

    thresholds = np.linspace(0, 1, 255)

    iou_multiclass = []
    iou_oneclass_array = []
    iou_head_array = []
    iou_tail_array = []
    iou_merge_array = []

    softmask_multiclass_np = np.array(softmask_multiclass)
    softmask_oneclass_np = np.array(softmask_oneclass)

    softmask_multiclass = np.transpose(softmask_multiclass, axes=(0, 3, 1, 2))
    softmask_oneclass = np.transpose(softmask_oneclass, axes=(0, 3, 1, 2))

    print("softmask_multiclass", np.shape(softmask_multiclass))
    print("softmask_oneclass", np.shape(softmask_oneclass))

    # if multiclass xd

    # softmask_tail = np.array(softmask_multiclass_np[:, :, :, 1])
    # softmask_head = np.array(softmask_multiclass_np[:, :, :, 2])
    softmask_tail = np.array(softmask_multiclass_np[:, 1, :, :])
    softmask_head = np.array(softmask_multiclass_np[:, 2, :, :])
    softmask_merge_multiclass = softmask_head + softmask_tail

    print("softmask_tail", np.shape(softmask_tail))
    print("softmask_head", np.shape(softmask_head))
    print("softmask_merge", np.shape(softmask_merge_multiclass))

    # stop

    for i, threshold in enumerate(thresholds):
        pred_merge_multi = (softmask_merge_multiclass > threshold)
        pred_merge_multi = pred_merge_multi.squeeze()
        labels2 = labels2.squeeze()
        print("pred_merge_multi", np.shape(pred_merge_multi))
        print("labels2", np.shape(labels2))
        iou_merge_multiclass, iou_merge_multiclass_list = calculate_iou(pred_merge_multi, labels2, 2)

        pred_mask_multi = (softmask_multiclass > threshold).astype(np.uint8)

        pred_mask_multi_tail = pred_mask_multi[:, 1, :, :]
        pred_mask_multi_head = pred_mask_multi[:, 2, :, :]

        final_mask = np.zeros_like(pred_mask_multi_head)

        conflict = (pred_mask_multi_head == 1) & (pred_mask_multi_head == 1)

        tail_conflict = softmask_multiclass_np[:, :, :, 1][conflict]
        head_conflict = softmask_multiclass_np[:, :, :, 2][conflict]

        final_mask[conflict] = 2 - (tail_conflict > head_conflict).astype(int)

        final_mask[(pred_mask_multi_tail == 1) & ~conflict] = 1  # 1 dla ogona
        final_mask[(pred_mask_multi_head == 1) & ~conflict] = 2  # 2 dla głowki

        final_iou, final_iou_list = calculate_iou(final_mask, labels1, 3)
        iou_head, iou_head_list = calculate_iou(pred_mask_multi_head, masks[:, 2, :, :], 2)
        iou_tail, iou_tail_list = calculate_iou(pred_mask_multi_tail, masks[:, 1, :, :], 2)


        #  poza warunkami
        pred_masks_oneclass = (softmask_oneclass > threshold)
        pred_masks_oneclass = pred_masks_oneclass[:, 1, :, :]

        iou_oneclass, iou_oneclass_list = calculate_iou(pred_masks_oneclass, labels2, 2)

        iou_multiclass.append(final_iou)
        iou_oneclass_array.append(iou_oneclass)
        iou_head_array.append(iou_head)
        iou_tail_array.append(iou_tail)
        iou_merge_array.append(iou_merge_multiclass)

        print(f"IoU {i} calculated")
        print(f"multiclass:{final_iou}, oneclass:{iou_oneclass}, "
              f"head:{iou_head}, tail:{iou_tail}, merge:{iou_merge_multiclass}")
    ap_score = calculate_ap_for_segmentation(softmask_oneclass_np[:, :, :, 1], labels2)
    print(f"ap: {ap_score}")

    # optymalne thresholdy
    # TODO


def predict(model: torch.nn, test_loader, device = torch.device("cpu")):
    model.eval()
    true_masks = []
    input_imgs = []
    predicted_masks_multiclass = []
    predicted_masks_oneclass = []
    predicted_softmasks_multiclass = []
    predicted_softmasks_oneclass = []

    with torch.no_grad():
        for data in test_loader:
            images, labels = data

            images = images.to(device)
            outputs = model(images)

            output1 = outputs[:, :3, :, :]
            output2 = outputs[:, [0, 1], :, :]

            true_masks.append(labels.cpu())
            input_imgs.append(images.cpu())

            preds1 = torch.argmax(output1, dim=1)
            preds2 = torch.argmax(output2, dim=1)

            softs1 = torch.softmax(output1, dim=1).squeeze(0)
            softs2 = torch.softmax(output2, dim=1).squeeze(0)

            predicted_masks_multiclass.append(preds1.cpu())
            predicted_masks_oneclass.append(preds2.cpu())
            predicted_softmasks_multiclass.append(softs1.cpu())
            predicted_softmasks_oneclass.append(softs2.cpu())
        calculate_optimal_threshold(images, labels, predicted_softmasks_multiclass, predicted_softmasks_oneclass)


with open("path_config.yaml", 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

saved_model_name = 'One segment lossA=1 k=5 M=smpUNet++ aug=False mode=multiclass Opt=Adam Sch=CosineAnnealingLR E=2 B_size=6 lr=0.001 Loss=CrossEntropyLoss date=13-08-2024-19-03_best_model_iou_multiclass'
model_path = yaml_config['save_model_path'] + '/' + saved_model_name

model = smp.UnetPlusPlus(in_channels=3, classes=4, encoder_name="resnet18", encoder_weights=None)
model.load_state_dict(torch.load(model_path))

batch_maker = BatchMaker(work_mode="test", batch_size=1)
test_loader = batch_maker.test_loader

predict(model, test_loader)

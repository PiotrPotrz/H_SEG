import os.path

import torch
import yaml
import numpy as np
import wandb
import cv2
import pandas as pd

from utils.transformation import transform_batch, transform_mask
from utils.metrics import calculate_iou, calculate_iou_per_class


def inference(ximages, mask_numpy, mask_plot, pred_numpy, pred_plot, number = "1",
              mask_name = "mask", video_name = "video", class_diff = False, name= "name"):

    with open("path_config.yaml", "r") as file:
        yaml_config = yaml.safe_load(file)
        file.close()


    labels1 = transform_batch(mask_numpy, 1)
    labels2 = transform_batch(mask_numpy, 2)

    unique_values = np.unique(pred_numpy)
    multiclass = len(unique_values[unique_values != 0]) > 0

    ious_multiclass = []
    ious_oneclass = []
    
    soft_mask_path = os.path.join(yaml_config["save_soft_mask_path"], name)
    
    if not os.path.exists(soft_mask_path):
        os.makedirs(soft_mask_path)
    if not os.path.exists(soft_mask_path + '/0.5masks'):
        os.makedirs(soft_mask_path + '/0.5masks')
    if not os.path.exists(soft_mask_path + '/0.5masks/oneclass'):
        os.makedirs(soft_mask_path + '/0.5masks/oneclass')

    if multiclass:
        if not os.path.exists(soft_mask_path + "/0.5masks//multiclass"):
            os.makedirs(soft_mask_path + "/0.5masks//multiclass")
        mean_iou_multiclass, iou_per_class_multiclass = calculate_iou(labels1, pred_numpy, 3)
        for i in range(labels1.shape[0]):
            iou_multiclass, iou_multiclass2 = calculate_iou(labels1[i], pred_numpy[i], 3)
            iou = ious_multiclass
            ious_multiclass.append(iou)
    else:
        mean_iou_multiclass = 0



    return

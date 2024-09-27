import os.path

import torch
import yaml
import numpy as np
import wandb
import cv2
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

import data
from utils.transformation import transform_batch
from utils.metrics import calculate_iou, calculate_ap_for_segmentation


def apply_threshold(mask, threshold = 0.9):
    return np.where(mask>threshold, 1, 0).astype(float)


def inference(ximages, mask_numpy, mask_plot, pred_numpy, pred_plot, number='1', mask_name='mask', video_name='video',
              class_diff=False, name='name'):
    # pred numpy - predykcja multiclass
    # pred plot - predykcja one class

    ids1 = transform_batch(mask_numpy, 1)
    ids2 = transform_batch(mask_numpy, 2)

    unique_values = np.unique(pred_numpy)
    multiclass = len(unique_values[unique_values != 0]) > 1

    ious_multiclass = []
    ious_oneclass = []

    if multiclass:
        mean_iou_multiclass, iou_per_class_multiclass = calculate_iou(ids1, pred_numpy, 3)
        mean_iou_multiclass05, iou_per_class_multiclass05 = calculate_iou(ids1,
                                                                          apply_threshold(pred_numpy, threshold=0.5), 3)

        for i in range(ids1.shape[0]):
            iou_multiclass, iou_multiclasssss = calculate_iou(ids1[i], pred_numpy[i], 3)
            iou = iou_multiclass
            ious_multiclass.append(iou)

    else:
        mean_iou_multiclass = 0
        iou_per_class_multiclass = 0
        ious_multiclass = 0
        mean_iou_multiclass05, iou_per_class_multiclass05 = 0, 0

    mean_iou_oneclass, iou_per_class_oneclass = calculate_iou(ids2, pred_plot, 2)
    mean_iou_oneclass05, iou_per_class_oneclass05 = calculate_iou(ids2, apply_threshold(pred_plot, threshold=0.5), 2)
    for i in range(ids2.shape[0]):
        iou_oneclass, iou_oneclasssss = calculate_iou(ids2[i], pred_plot[i], 2)
        ious_oneclass.append(iou_oneclass)

    # mean_iou, iou_per_class = calculate_iou(mask_numpy, pred_numpy)
    print(f"Mean IoU oneclass ({mask_name}):", mean_iou_oneclass)
    print(f"IoU per class oneclass ({mask_name}):", iou_per_class_oneclass)
    print(f"Mean IoU multiclass ({mask_name}):", mean_iou_multiclass)
    print(f"IoU per class multiclass ({mask_name}):", iou_per_class_multiclass)

    print(f"Mean IoU oneclass 0.5 ({mask_name}):", mean_iou_oneclass05)
    print(f"IoU per class oneclass 0.5 ({mask_name}):", iou_per_class_oneclass05)
    print(f"Mean IoU multiclass 0.5 ({mask_name}):", mean_iou_multiclass05)
    print(f"IoU per class multiclass 0.5 ({mask_name}):", iou_per_class_multiclass05)

    return mean_iou_oneclass, iou_per_class_oneclass, mean_iou_multiclass, iou_per_class_multiclass, ious_multiclass, ious_oneclass


def calculate_optimal_threshold(x_images, mask_numpy, softmask_multiclass, softmask_oneclass, pred_multiclass, name,
                                intersection):
    ids1 = transform_batch(mask_numpy, 1)
    ids2 = transform_batch(mask_numpy, 2)

    inter_multi = transform_batch(intersection, 1)
    inter_one = transform_batch(intersection, 2)

    unique_values = np.unique(pred_multiclass)
    multiclass = len(unique_values[unique_values != 0]) > 1

    softmask_multiclass_np = np.array(softmask_multiclass)
    softmask_oneclass_np = np.array(softmask_oneclass)
    softmask_oneclass_np = softmask_oneclass_np.transpose((0, 2, 3, 1))
    softmask_multiclass_np = softmask_multiclass_np.transpose((0, 2, 3, 1))
    images_np = np.array(x_images)

    folder_path = os.path.join(yaml_config['save_soft_mask_path'], name)

    for i in range(softmask_oneclass_np.shape[0]):
        img = 255 * softmask_oneclass_np[i]
        softmaskOneclass = img[:, :, 1]
        GT = 255 * ids2[i]

        # GT_head = mask_numpy[i,2,:,:]*255
        # GT_tail = mask_numpy[i,1,:,:]*255

        realImage = images_np[i] * 255
        if multiclass:
            softmaskMulticlass = 255 * softmask_multiclass_np[i]
            img_head = softmaskMulticlass[:, :, 2]
            img_tail = softmaskMulticlass[:, :, 1]
            img_multiclass = img_head + img_tail

    thresholds = torch.linspace(0, 1, steps=255)

    iou_scores_multiclass = []
    iou_scores_oneclass = []
    iou_scores_head = []
    iou_scores_tail = []
    iou_scores_merge = []

    if multiclass:
        softmask_tail_np1 = np.array(softmask_multiclass_np[:, :, :, 1])
        softmask_head_np1 = np.array(softmask_multiclass_np[:, :, :, 2])
        softmask_merge_multiclass = softmask_head_np1 + softmask_tail_np1

    i = 0
    for threshold in thresholds:
        if multiclass:
            pred_merge_multiclass = (softmask_merge_multiclass > threshold.numpy()).astype(float)
            # if config.mode == 'intersection_and_union_inference':
            #     pred_merge_multiclass = np.maximum(pred_merge_multiclass, inter_multi)
            iou_merge_multiclass, i_merge_multiclass = calculate_iou(pred_merge_multiclass, ids2, 2)

            pred_masks_multiclass = (softmask_multiclass > threshold.numpy()).astype(float)
            pred_masks_multiclass_tail = pred_masks_multiclass[:, 1, :, :]
            pred_masks_multiclass_head = pred_masks_multiclass[:, 2, :, :]
            final_mask = np.zeros_like(pred_masks_multiclass_head)
            conflict = (pred_masks_multiclass_head == 1) & (pred_masks_multiclass_tail == 1)
            tail_conflict_values = softmask_multiclass_np[:, :, :, 1][conflict]
            head_conflict_values = softmask_multiclass_np[:, :, :, 2][conflict]
            final_mask[conflict] = 2 - (tail_conflict_values > head_conflict_values).astype(
                int)  # 1 dla tail, 2 dla head
            final_mask[(pred_masks_multiclass_tail == 1) & ~conflict] = 1
            final_mask[(pred_masks_multiclass_head == 1) & ~conflict] = 2

            iou_multiclass, i_multiclass = calculate_iou(final_mask, ids1, 3)
            iou_head, i_head = calculate_iou(pred_masks_multiclass_head, mask_numpy[:, 2, :, :], 2)
            iou_tail, i_tail = calculate_iou(pred_masks_multiclass_tail, mask_numpy[:, 1, :, :], 2)
        else:
            iou_multiclass = 0
            i_multiclass = 0
            iou_head = 0
            i_head = 0
            iou_tail = 0
            i_tail = 0
            iou_merge_multiclass = 0
        pred_masks_oneclass = (softmask_oneclass > threshold.numpy()).astype(float)
        pred_masks_oneclass = pred_masks_oneclass[:, 1, :, :]

        # if config.mode == 'intersection_and_union_inference':
        #     pred_masks_oneclass = np.maximum(pred_masks_oneclass, inter_one)
        iou_oneclass, i_oneclass = calculate_iou(pred_masks_oneclass, ids2, 2)

        iou_scores_multiclass.append(iou_multiclass)
        iou_scores_oneclass.append(iou_oneclass)
        iou_scores_head.append(iou_head)
        iou_scores_tail.append(iou_tail)
        iou_scores_merge.append(iou_merge_multiclass)
        i += 1
        print(f"Iou {i} calculated")

    ap_score = calculate_ap_for_segmentation(softmask_oneclass_np[:, :, :, 1], ids2)

    optimal_threshold_multiclass = thresholds[torch.argmax(torch.tensor(iou_scores_multiclass))]
    optimal_iou_multiclass = max(iou_scores_multiclass)
    print(f"Optimal threshold for multiclass: {optimal_threshold_multiclass}, IoU: {optimal_iou_multiclass}")

    optimal_threshold_oneclass = thresholds[torch.argmax(torch.tensor(iou_scores_oneclass))]
    optimal_iou_oneclass = max(iou_scores_oneclass)
    print(f"Optimal threshold for oneclass: {optimal_threshold_oneclass}, IoU: {optimal_iou_oneclass}")

    optimal_iou_head = max(iou_scores_head)
    optimal_iou_head_threshold = thresholds[torch.argmax(torch.tensor(iou_scores_head))]
    print(f"Optimal threshold for head: {optimal_iou_head_threshold}, IoU: {optimal_iou_head}")

    optimal_iou_tail = max(iou_scores_tail)
    optimal_iou_tail_threshold = thresholds[torch.argmax(torch.tensor(iou_scores_tail))]
    print(f"Optimal threshold for tail: {optimal_iou_tail_threshold}, IoU: {optimal_iou_tail}")

    optimal_iou_merge = max(iou_scores_merge)
    optimal_iou_merge_threshold = thresholds[torch.argmax(torch.tensor(iou_scores_merge))]
    print(f"Optimal threshold for merge: {optimal_iou_merge_threshold}, IoU: {optimal_iou_merge}")

    if multiclass:
        softmask_head = softmask_multiclass_np[:, :, :, 2]
        pred_head = (softmask_head > optimal_iou_head_threshold.numpy()).astype(float)
        softmask_tail = softmask_multiclass_np[:, :, :, 1]
        pred_tail = (softmask_tail > optimal_iou_tail_threshold.numpy()).astype(float)
        softmask_merge_multiclass = softmask_tail + softmask_head

        final_mask = np.zeros_like(pred_head)
        conflict = (pred_head == 1) & (pred_tail == 1)
        tail_conflict_values = softmask_multiclass_np[:, :, :, 1][conflict]
        head_conflict_values = softmask_multiclass_np[:, :, :, 2][conflict]
        final_mask[conflict] = 2 - (tail_conflict_values > head_conflict_values).astype(int)  # 1 dla tail, 2 dla head
        final_mask[(pred_tail == 1) & ~conflict] = 1
        final_mask[(pred_head == 1) & ~conflict] = 2

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        for i in range(final_mask.shape[0]):
            final_mask[i] = cv2.morphologyEx(final_mask[i], cv2.MORPH_CLOSE, kernel)

        pred_head2 = (softmask_head > optimal_threshold_multiclass.numpy()).astype(float)
        pred_tail2 = (softmask_tail > optimal_threshold_multiclass.numpy()).astype(float)
        pred_merge_multiclass = (softmask_merge_multiclass > optimal_iou_merge_threshold.numpy()).astype(float)
        final_mask2 = np.zeros_like(pred_head2)
        conflict2 = (pred_head2 == 1) & (pred_tail2 == 1)
        tail_conflict_values2 = softmask_multiclass_np[:, :, :, 1][conflict2]
        head_conflict_values2 = softmask_multiclass_np[:, :, :, 2][conflict2]
        final_mask2[conflict2] = 2 - (tail_conflict_values2 > head_conflict_values2).astype(
            int)  # 1 dla tail, 2 dla head
        final_mask2[(pred_tail2 == 1) & ~conflict2] = 1
        final_mask2[(pred_head2 == 1) & ~conflict2] = 2

        iou_multiclass_best_tresholds, i_multiclass_best_thresholds = calculate_iou(final_mask, ids1, 3)
    else:
        iou_multiclass_best_tresholds = 0
        i_multiclass2_best_thresholds = 0

    softmask_oneclass2 = softmask_oneclass_np[:, :, :, 1]
    pred_oneclass2 = (softmask_oneclass2 > optimal_threshold_oneclass.numpy()).astype(float)
    # if config.mode == 'intersection_and_union_inference':
    #     pred_oneclass2 = np.maximum(pred_oneclass2, inter_one)

    if multiclass:
        full_agreement_mask = np.zeros_like(pred_masks_multiclass_head)
        full_agreement_mask[(pred_head == 1) & (pred_oneclass2 == 1)] = 2
        full_agreement_mask[(pred_tail == 1) & (pred_oneclass2 == 1)] = 1

        print(f"Full agreement mask:")
        iou_full_agreement, i_full_agreement = calculate_iou(full_agreement_mask, ids1, 3)
        print(f"FULL AGREEMENT IOU: {iou_full_agreement}")

    ious_multiclass = []
    ious_oneclass = []
    ious_head = []
    ious_tail = []
    ious_full_agreement = []

    if multiclass:
        for i in range(ids1.shape[0]):
            iou_multiclass, iou_multiclasssss = calculate_iou(ids1[i], final_mask[i], 3)
            iou_head, i_head = calculate_iou(mask_numpy[i, 2, :, :], pred_head[i], 2)
            iou_tail, i_tail = calculate_iou(mask_numpy[i, 1, :, :], pred_tail[i], 2)
            iou = iou_multiclass
            ious_multiclass.append(iou)
            ious_head.append(iou_head)
            ious_tail.append(iou_tail)
    else:
        mean_iou_multiclass = 0
        iou_per_class_multiclass = 0
        ious_multiclass = 0
        ious_head = 0
        ious_tail = 0

    for i in range(ids2.shape[0]):
        iou_oneclass, iou_oneclasssss = calculate_iou(ids2[i], pred_oneclass2[i], 2)
        ious_oneclass.append(iou_oneclass)

    if not multiclass:
        optimal_iou_head = 0
        optimal_iou_tail = 0
        optimal_iou_head_threshold = 0
        optimal_iou_tail_threshold = 0
        iou_multiclass_best_tresholds = 0
        i_multiclass2_best_thresholds = 0

    # # Wykres IoU od threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds.numpy(), iou_scores_multiclass, label='IoU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU Score')
    plt.title('IoU Score as a function of Threshold (multiclass)')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds.numpy(), iou_scores_oneclass, label='IoU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IoU Score')
    plt.title('IoU Score as a function of Threshold (one class)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return optimal_threshold_multiclass, optimal_iou_multiclass, optimal_threshold_oneclass, optimal_iou_oneclass, ap_score, optimal_iou_head, optimal_iou_tail, optimal_iou_head_threshold, optimal_iou_tail_threshold, iou_multiclass_best_tresholds, ious_oneclass, ious_multiclass, ious_head, ious_tail, optimal_iou_merge, optimal_iou_merge_threshold, iou_full_agreement


def predict(model, test_loader):
    model.eval()

    input_images = []
    predicted_masks_oneclass = []
    predicted_masks_multiclass = []
    predicted_softmasks_oneclass = []
    predicted_softmasks_multiclass = []
    true_masks = []

    # ograniczam się do przypadku z data length == 2
    with torch.no_grad():
        for data in test_loader:
            if len(data) == 2:
                inputs, ids = data

            inputs = inputs.to(device)
            outputs = model(inputs)
            output1 = outputs[:, :3, :, :]
            output2 = outputs[:, [0, -1], :, :]

            if len(data) == 2:
                true_masks.append(ids.cpu())

            input_images.append(inputs.cpu())
            preds1 = torch.argmax(output1, dim=1)
            preds2 = torch.argmax(output2, dim=1)
            softs1 = torch.softmax(output1, dim=1)
            softs1 = softs1.squeeze(0)
            softs2 = torch.softmax(output2, dim=1)
            softs2 = softs2.squeeze(0)

            predicted_masks_multiclass.append(preds1.cpu())
            predicted_masks_oneclass.append(preds2.cpu())
            predicted_softmasks_multiclass.append(softs1.cpu())
            predicted_softmasks_oneclass.append(softs2.cpu())

    input_images = np.concatenate(input_images, axis=0)
    true_masks = np.concatenate(true_masks, axis=0)

    predicted_masks_multiclass = np.concatenate(predicted_masks_multiclass, axis=0)
    predicted_masks_oneclass = np.concatenate(predicted_masks_oneclass, axis=0)

    x_images = input_images.transpose((0, 2, 3, 1))
    true = true_masks.transpose((0, 2, 3, 1))
    pred1 = predicted_masks_multiclass  # .transpose((0, 2, 3, 1))
    pred2 = predicted_masks_oneclass  # .transpose((0, 2, 3, 1))
    soft1 = predicted_softmasks_multiclass
    soft2 = predicted_softmasks_oneclass

    mean_iou_ids1_oneclass, iou_ids1_oneclass, mean_iou_ids1_multiclass, iou_ids1_multiclass, ious_multiclass_annotator1, ious_oneclass_annotator1 = inference(
        x_images, true_masks, true, pred1, pred2, number='_annotator1',
        mask_name='annotator1', video_name='ids1', class_diff=False, name=saved_model_name + '_')

    mean_iou_ids2_oneclass, iou_ids2_oneclass, mean_iou_ids2_multiclass, iou_ids2_multiclass, ious_multicass_annotator2, ious_oneclass_annotator2 = 0, 0, 0, 0, 0, 0  # zmieniam liczbę zer

    mean_iou_intersection_oneclass, iou_intersection_oneclass, mean_iou_intersection_multiclass, iou_intersection_multiclass, ious_multiclass_intersection, ious_oneclass_intersection = 0, 0, 0, 0, 0, 0

    mean_iou_union_oneclass, iou_union_oneclass, mean_iou_union_multiclass, iou_union_multiclass, ious_multiclass_union, ious_oneclass_union = 0, 0, 0, 0, 0, 0

    mean_iou_feeling_oneclass, iou_feeling_oneclass, mean_iou_feeling_multiclass, iou_feeling_multiclass, ious_multiclass_lucky, ious_oneclass_lucky = 0, 0, 0, 0, 0, 0

    (optimal_threshold_multiclass, optimal_iou_multiclass, optimal_threshold_oneclass, optimal_iou_oneclass,
     ap_score_oneclass, optimal_iou_head, optimal_iou_tail, optimal_head_treshold, optimal_tail_threshold,
     iou_multiclass_best_tresholds, ious_opt_oneclass_annotator1, ious_opt_multiclass_annotator1,
     ious_opt_head_annotator1, ious_opt_tail_annotator1, optimal_merge_iou, optimal_merge_treshold,
     iou_full_agreement) = calculate_optimal_threshold(
        x_images, true_masks, soft1, soft2, pred1, saved_model_name + '_',
        ids)  # było intersections_numpy ale ich nie ma

    name = saved_model_name + '_'
    folder_path = os.path.join(yaml_config['save_soft_mask_path'], name)

    test_metrics = {"inference/Mean Iou oneclass (annotator1)": mean_iou_ids1_oneclass,
                    "inference/Iou for each class oneclass (annotator1)": iou_ids1_oneclass,
                    "inference/Mean Iou multiclass (annotator1)": mean_iou_ids1_multiclass,
                    "inference/Iou for each class multiclass (annotator1)": iou_ids1_multiclass,
                    "inference/Optimal threshold multiclass": optimal_threshold_multiclass,
                    "inference/Optimal IoU multiclass": optimal_iou_multiclass,
                    "inference/Optimal threshold oneclass": optimal_threshold_oneclass,
                    "inference/Optimal IoU oneclass": optimal_iou_oneclass,
                    "inference/AP oneclass": ap_score_oneclass,
                    "inference/Optimal IoU head": optimal_iou_head,
                    "inference/Optimal IoU tail": optimal_iou_tail,
                    "inference/Optimal threshold head": optimal_head_treshold,
                    "inference/Optimal threshold tail": optimal_tail_threshold,
                    "inference/IoU multiclass best tresholds": iou_multiclass_best_tresholds,
                    "inference/IoU merge treshold": optimal_merge_treshold,
                    "inference/Optimal IoU merge": optimal_merge_iou,
                    "inference/IoU full agreement": iou_full_agreement
                    }

    wandb.log(test_metrics)


num_classes = 4

with open("path_config.yaml", 'r') as config_file:
    yaml_config = yaml.safe_load(config_file)

model_dict = {'smpUNet': smp.Unet(in_channels=3, classes=num_classes),
              'smpUNet++': smp.UnetPlusPlus(in_channels=3, classes=num_classes, encoder_name="resnet18",
                                            encoder_weights=None),
              }

wandb.init(project="project",
           config={
               "model": "smpUNet++",
               "batch_size": 1,
               "annotator": 1,
           })

config = wandb.config

saved_model_name = 'OslNWAUG A=1 k=5 M=smpUNet++ aug=True mode=multiclass date=26-09-2024-21-53_best_model_iou_oneclass.001 Loss=CrossEntropyLoss date=26-09-2024-21-53_best_model_iou_o.001 Loss=CrossEntropyLoss date=26-09-2024-21-53_best_model_iou_oneclass'
model_path = yaml_config['save_model_path'] + '/' + saved_model_name
k = 1
# name = (f'y2_Two_segment_loss alfa = {k}_Inference: Model_name: {saved_model_name}')
name = "last model aug"

wandb.run.name = name

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print("GPU dostępne:", gpu_name)
    device = torch.device("cuda")
else:
    if torch.cpu.is_available():
        device = torch.device("cpu")
        print("WYKORZYSTUJE SAMO CPU")
    else:
        raise Exception("no device aviliable")



model = model_dict[config.model]
model.load_state_dict(torch.load(model_path))
model.to(device)
batch_maker = data.BatchMaker(batch_size=config["batch_size"], annotator=config["annotator"],
                              mode="mixed", work_mode="test")
test_loader = batch_maker.test_loader
predict(model, test_loader)


wandb.finish()

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import segmentation_models_pytorch as smp
import numpy as np
import yaml
import datetime
from utils.better_aug import BetterAugmentation


import train
import validation
import data

if torch.cuda.is_available():
    print(f"GPU dostępne {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda")
else:
    print("GPU NIE JEST DOSTĘPNE")
    print("WYKORZYSTUJE CPU")
    device = torch.device("cpu")

with open("path_config.yaml", "r") as file:
    path_config = yaml.safe_load(file)
    file.close()

config = {"epochs": 2,
            "batch_size": 6,
            "lr": 1e-3,
            "annotator": 1,
            "model": 'smpUNet++',
            "augmentation": True,
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "place": "lab",
            "mode": "multiclass",
            "aug_type": "BetterAugmentation",
            "k": 5
          }



# tworzenie pliku konfiguracyjnego do treningu
# jest on wykorzystywany w innych plikach

with open("train_config.yaml", "w") as yaml_file:
    yaml.dump(config, yaml_file)
    yaml_file.close()

if config["mode"] == "oneclass":
    classes = 2
else:
    classes = 4

model_dict = {
    "smpUnet": smp.Unet(in_channels=3, classes=classes),
    "smpUNet++": smp.UnetPlusPlus(in_channels=3, classes=classes, encoder_name="resnet18", encoder_weights=None)
}

mode_dict = {'normal': 'mixed',
             'intersection': 'intersection',
             'intersection_and_union': 'intersection_and_union_inference',
             'feeling_lucky': 'feeling_lucky',
             'union': 'union',
             "oneclass": 'mixed',
             "multiclass": 'mixed'
}

model = model_dict[config["model"]]
model.to(device)

loss_dict = {
    "CrossEntropyLoss": nn.CrossEntropyLoss()
}

loss_fn = loss_dict[config["loss"]]

optimizer_dict = {
    "SGD": optim.SGD(model.parameters(), lr=config["lr"]),
    "Adam": optim.Adam(model.parameters(), lr=config["lr"])
}

optimizer = optimizer_dict[config["optimizer"]]

scheduler_dict = {'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"]),
                  'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min'),
                  "MultiStepLR": optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3),
                  'None': None}

scheduler = scheduler_dict[config["scheduler"]]

timestamp = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M")


wandb.init(project="project", config=config)

name = (f'A={config["annotator"]} k={config["k"]} M={config["model"]} aug={config["augmentation"]} mode={config["mode"]} Opt={config["optimizer"]} Sch={config["scheduler"]} E={config["epochs"]} B_size={config["batch_size"]} lr={config["lr"]} Loss={config["loss"]} date={timestamp}')
# name = "czy aug dziala"
if config["mode"] == "intersection_and_union":
    description = "Two segment"
else:
    description = "One segment loss"

name = description + name

wandb.run.name = name

best_iou = 1000000
best_iou_multiclass = 1000000
best_iou_opt_oneclass = 0
best_ap_oneclass = 0
best_ap_head = 0
best_ap_tail = 0

batch_maker = data.BatchMaker(batch_size=config["batch_size"], annotator=config["annotator"], 
                              mode=mode_dict[config["mode"]], work_mode="train")

train_loader = batch_maker.train_loader
val_loader = batch_maker.test_loader

aug = BetterAugmentation()

for epoch in range(config["epochs"]):
    epoch_number = epoch + 1
    # train_loss, train_iou_multiclass, train_iou_oneclass = train.train(model, train_loader, optimizer, scheduler,
    #                                                                    loss_fn, config["augmentation"], config["aug_type"], epoch_number, device)
    # train_loss, train_iou_multiclass, train_iou_oneclass = train.train(model, train_loader, optimizer, scheduler,
    #                                                                    loss_fn, aug,
    #                                                                    config["aug_type"], epoch_number, device)
    train_loss, train_iou_multiclass, train_iou_oneclass = train.train(model, train_loader, optimizer, scheduler,
                                                                       loss_fn, aug,
                                                                       True, epoch_number, device)
    validation_iou_multiclass, validation_iou_oneclass, vimages, vlbls_multiclass, vlbls_oneclass, vpreds_multiclass, vpreds_oneclass, ap_score_oneclass, ap_score_head, ap_score_tail = validation.val(model, val_loader, loss_fn, epoch_number, scheduler, device)
    validation.plot_results(vimages, vlbls_multiclass, vpreds_multiclass, index=0, mode="val", number="1")
    validation.plot_results(vimages, vlbls_oneclass, vpreds_oneclass, index=0, mode="val", number="2")

    print(f'Epoch {epoch_number}, Train Loss: {train_loss}, Train Iou Multiclass: {train_iou_multiclass}, Train Iou Oneclass: {train_iou_oneclass}, '
        f'Validation Iou Multiclass: {validation_iou_multiclass}, Validation Iou Oneclass: {validation_iou_oneclass}')
    print(f'Validation AP Oneclass: {ap_score_oneclass}', f'Validation AP Head: {ap_score_head}',
          f'Validation AP Tail: {ap_score_tail}')

    if validation_iou_oneclass < best_iou:
        best_iou = validation_iou_oneclass
        model_path = path_config['save_model_path'] + '/' + name + '_best_model_iou_oneclass'
        torch.save(model.state_dict(), model_path)
        print('Model saved (iou_oneclass)')
    if ap_score_oneclass > best_ap_oneclass:
        best_ap_oneclass = ap_score_oneclass
        model_path = path_config['save_model_path'] + '/' + name + '_best_model_ap_oneclass'
        torch.save(model.state_dict(), model_path)
        print("Model saved (ap_oneclass)")
    if ap_score_head > best_ap_head:
        best_ap_head = ap_score_head
        model_path = path_config['save_model_path'] + '/' + name + '_best_model_ap_head'
        torch.save(model.state_dict(), model_path)
        print("Model saved (ap_head)")
    if ap_score_tail > best_ap_tail:
        best_ap_tail = ap_score_tail
        model_path = path_config['save_model_path'] + '/' + name + '_best_model_ap_tail'
        torch.save(model.state_dict(), model_path)
        print("Model saved (ap_tail)")
    if validation_iou_multiclass < best_iou_multiclass:
        best_iou_multiclass = validation_iou_multiclass
        model_path = path_config['save_model_path'] + '/' + name + '_best_model_iou_multiclass'
        torch.save(model.state_dict(), model_path)
        print('Model saved (iou_multiclass)')
    if epoch_number == config["epochs"]:
        model_path = path_config['save_model_path'] + '/' + name + '_last_model'
        torch.save(model.state_dict(), model_path)
        print('Model saved')

# zapisywanie modelów na wandb

wandb.save(path_config['save_model_path'] + '/' + name + '_best_model_iou_oneclass')
wandb.save(path_config['save_model_path'] + '/' + name + '_best_model_ap_oneclass')
wandb.save(path_config['save_model_path'] + '/' + name + '_best_model_ap_head')
wandb.save(path_config['save_model_path'] + '/' + name + '_best_model_ap_tail')
wandb.save(path_config['save_model_path'] + '/' + name + '_best_model_iou_multiclass')
wandb.save(path_config['save_model_path'] + '/' + name + '_last_model')

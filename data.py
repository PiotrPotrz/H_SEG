import os.path

import matplotlib.pyplot as plt
import skimage.transform
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import numpy as np
import cv2
import glob

def process_mask(rgb_mask, class_colors):
    """
    Funkcja zamienia maskę rgb na maskę z numerami klas.
    :param rgb_mask: Maska kolorowa
    :param class_colors: Lista kolorów klas
    :return: Nowa maska z numerami klas
    """
    # nowa pusta maska
    index_mask = np.zeros(shape=(*rgb_mask.shape[:2],len(class_colors)), dtype=np.float32)
    for class_idx, color in enumerate(class_colors):
        idx = class_idx
        if class_idx == 3:
            idx = 1
            # tworzona jest maska binarna
            # dla koloru odpowiadającego color przypisywane jest 1 a dla innych zero
            mask = (rgb_mask == color).all(axis=2).astype(np.float32)
            index_mask[:, :, -1] = np.logical_or(index_mask[:, :, -1], mask)
        else:
            mask =(rgb_mask == color).all(axis=2).astype(np.float32)
            index_mask[:, :, idx] = mask

        if idx == 1 or idx == 2:
            index_mask[:, :, -1] = np.logical_or(index_mask[:, :, -1], mask)
    return index_mask


class Data:
    def __init__(self, path_config_path="path_config.yaml", mode='full', annotator=1):
        self.mode = mode
        self.annontator = annotator
        with open(path_config_path, "r") as path_config_file:
            path_config = yaml.safe_load(path_config_file)
            self.dataset_path = path_config["dataset_path"]
            self.image_width = path_config["image_width"]
            self.image_height = path_config["image_height"]

    def prepare_dataset(self, dataset_name):
        class_colors = [[0, 0, 0], [0, 255, 0], [0, 0, 255],[0,255,255]]
        # dataset_name may be train or test
        match self.annontator:
            case 1:
                name = "GT1_"
            case 2:
                name = "GT2_"
            case 3:
                name = "GT3_"
            case 4:
                name = "GT4_"
            case 5:
                name = "GT5_"

        if dataset_name != "train" and dataset_name != "test" and dataset_name != "test_small":
            raise ValueError("dataset_name must be train, test or test_small")

        path = self.dataset_path + "/" + dataset_name

        images = sorted(glob.glob(f"{path}/images/*"))

        if np.isin(self.mode, ['full', 'tail', 'head', 'mixed']):
            mask_path = f"{path}/{name}{self.mode}"
            print(mask_path)
            masks = sorted(glob.glob(f"{mask_path}/*.png"))
            print(np.shape(masks))

            X = np.zeros(shape=(len(images), self.image_height, self.image_width, 3), dtype=np.float32)
            y = np.zeros(shape=(len(masks), self.image_height, self.image_width, 4), dtype=np.float32)

            for n, (image, mask_image) in enumerate(zip(images, masks)):
                img = cv2.imread(image)
                img = img.astype(np.float32)
                # w oryginale użyto preserve_range = True, a potem normalizacje TODO
                # są to operacja zbędne

                img = skimage.transform.resize(img, (self.image_height, self.image_width, 3), mode='constant', preserve_range=False)

                mask = cv2.imread(mask_image)
                mask = mask.astype(np.float32)
                mask = skimage.transform.resize(mask, (self.image_height, self.image_width, 3), mode='constant', preserve_range=True)
                mask_id = process_mask(mask, class_colors)

                # fig, ax = plt.subplots(1, 4, figsize=(20, 8))
                # for i in range(4):
                #     plt.subplot(1, 4, i + 1)
                #     plt.imshow(mask_id[:, :, i])
                # plt.show()

                X[n] = img
                y[n] = mask_id

            print(f"X shape: {np.shape(X)}")
            print(f"y shape: {np.shape(y)}")
            return X, y

        elif np.isin(self.mode, ['intersection_and_union', 'intersection', 'intersection_and_union_inference', 'intersection_inference', 'feeling_lucky', 'union']):
            gt_path1 = f"{path}/GT1_mixed"
            gt_path2 = f"{path}/GT2_mixed" # ale przecież w TEST nie ma GT2 !!!!!!! TODO

            masks1 = sorted(glob.glob(f"{gt_path1}*.png"))
            print(np.shape(masks1))
            masks2 = sorted(glob.glob(f"{gt_path2}*.png"))
            print(np.shape(masks2))

            X = np.zeros(shape=(len(images), self.image_height, self.image_width, 3), dtype=np.float32)
            y1 = np.zeros(shape=(len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)
            y2 = np.zeros(shape=(len(masks2), self.image_height, self.image_width, 4), dtype=np.float32)
            intersections = np.zeros((len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)
            unions = np.zeros((len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)
            feelings = np.zeros((len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)
        elif  self.mode == "both":

            print("both")
            gt_path1 = path + '/GT1_' + 'mixed'
            # gt_path2 = dataset_path + '/GT2_' + 'mixed/'
            masks = sorted(glob.glob(f"{gt_path1}*.png"))
            masks2 = sorted(glob.glob(f"{gt_path1}*.png"))  # tutaj zmiana

            X = np.zeros((len(images), self.image_height, self.image_width, 3), dtype=np.float32)
            y1 = np.zeros((len(masks), self.image_height, self.image_width, 4), dtype=np.float32)
            y2 = np.zeros((len(masks), self.image_height, self.image_width, 4), dtype=np.float32)

            for n, (img, mimg, mimg2) in enumerate(zip(images, masks, masks2)):
                # Load images
                img = cv2.imread(img)
                x_img = img.astype(np.float32)
                x_img = skimage.transform.resize(x_img, (self.image_height, self.image_width, 3), mode='constant',
                               preserve_range=True)
                # Normalize images
                min_val = np.min(x_img)
                max_val = np.max(x_img)
                x_img = (x_img - min_val) / (max_val - min_val)

                # Load masks
                mask = cv2.imread(mimg)
                mask = mask.astype(np.float32)
                mask = skimage.transform.resize(mask, (self.image_height, self.image_width, 3), mode='constant',
                              preserve_range=True)
                mask_id = process_mask(mask, class_colors)

                mask2 = cv2.imread(mimg2)
                mask2 = mask2.astype(np.float32)
                mask2 = skimage.transform.resize(mask2, (self.image_height, self.image_width, 3), mode='constant',
                               preserve_range=True)
                mask2_id = process_mask(mask2, class_colors)
                # Save images and masks

                X[n] = x_img
                y1[n] = mask_id
                y2[n] = mask2_id

            return X, y1, y2

        elif self.mode == "IOU":
            gt_path1 = f"{path}/GT1_mixed"
            # nie mam pewności jak podzielić te zbiory, więc tworzę własną wersję zbiorów

            masks = sorted(glob.glob(f"{gt_path1}*.png"))

            masks1 = [mask for mask in masks if os.path.splitext(os.path.basename(mask))[0] % 2 == 0]
            masks2 = [mask for mask in masks if os.path.splitext(os.path.basename(mask))[0] % 2 == 1]

            X = np.zeros(shape=(len(images), self.image_height, self.image_width, 3), dtype=np.float32)
            y1 = np.zeros(shape=(len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)
            y2 = np.zeros(shape=(len(masks2), self.image_height, self.image_width, 4), dtype=np.float32)
            intersections = np.zeros((len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)
            unions = np.zeros((len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)
            feelings = np.zeros((len(masks1), self.image_height, self.image_width, 4), dtype=np.float32)

            for n, (image, mask1, mask2) in enumerate(zip(images, masks1, masks2)):
                pass
            # pass # # # # # # # # # # # # #

        else:
            raise NotImplementedError


class BatchMaker:
    def __init__(self, path_config_path="path_config.yaml", mode='full', annotator=1, work_mode='train', batch_size = 6):
        self.process_data = Data(path_config_path=path_config_path,
                                 mode=mode,
                                 annotator=annotator)
        self.batch_size = batch_size

        if np.isin(mode, ['full', 'tail', 'head', 'mixed']):
            if work_mode == "train":
                x_train, y_train = self.process_data.prepare_dataset("train")
                x_test, y_test = self.process_data.prepare_dataset("test")
                self.train_loader = self.basic_loader(x_train, y_train, shuffle=True)
                self.test_loader = self.basic_loader(x_test, y_test, shuffle=True)

            elif work_mode == "test":
                x_test, y_test = self.process_data.prepare_dataset("test")
                self.test_loader = self.basic_loader(x_test, y_test, shuffle=True)
        elif mode == "both":
            if mode == 'all':
                x_train, int_train, un_train = self.process_data.prepare_dataset('/train')
                x_val, int_val, un_val = self.process_data.prepare_dataset('/test_small')
                x_test, int_test, un_test = self.process_data.prepare_dataset( '/test')
                self.train_loader = self.create_loader2(x_train, int_train, un_train, shuffle=False)
                self.val_loader = self.create_loader2(x_val, int_val, un_val, shuffle=False)
                self.test_loader = self.create_loader2(x_test, int_test, un_test, shuffle=False)
            elif mode == 'train':
                x_train, int_train, un_train = self.process_data.prepare_dataset('/train')
                x_val, int_val, un_val = self.process_data.prepare_dataset('/test_small')
                self.train_loader = self.create_loader2(x_train, int_train, un_train, shuffle=True)
                self.val_loader = self.create_loader2(x_val, int_val, un_val, shuffle=True)
            elif mode == 'test':
                x_test, int_test, un_test = self.process_data.prepare_dataset('/test')
                self.test_loader = self.create_loader2(x_test, int_test, un_test, shuffle=False)




    def basic_loader(self, x, y, shuffle):
        x = np.transpose(x, axes=(0, 3, 1, 2))
        y = np.transpose(y, axes=(0, 3, 1, 2))

        x = torch.from_numpy(x)
        y = torch.from_numpy(y).type(torch.float64)
        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def inference_loader(self, x, intersection,union,shuffle):
        x = np.transpose(x, (0, 3, 1, 2))
        intersection = np.transpose(intersection, (0, 3, 1, 2))
        union = np.transpose(union, (0, 3, 1, 2))
        x_tensor = torch.from_numpy(x)
        intersection_tensor = torch.from_numpy(intersection).type(torch.float64)
        union_tensor = torch.from_numpy(union).type(torch.float64)
        dataset = TensorDataset(x_tensor, intersection_tensor,union_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score
def calculate_iou_per_class(mask1, mask2, num_classes):
    """
    Funkcja obliczająca iou na klasę
    :param mask1: maska 1
    :param mask2: maska 2
    :param num_classes: ilość klas
    :return: iou_scores - tablica obliczonych iou klas
    """
    iou_scores = np.zeros(num_classes)
    # if len(mask1.shape) == 2:
    #     plt.subplots(ncols=2, nrows=1)
    #
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(mask1)
    #     plt.title("mask1")
    #
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask2)
    #     plt.title("mask2")

    # tworzenie masek dla każdej klasy
    for clas in range(num_classes):
        # tworzone maski dla każdej klasy
        # każda wartość maski to klasa (0, 1, 2 itd.)
        mask1_class = (mask1 == clas)
        mask2_class = (mask2 == clas)

        intersection = np.logical_and(mask1_class, mask2_class).sum()
        union = np.logical_or(mask1_class, mask2_class).sum()

        # try:
        #     shell = get_ipython().__class__.__name__
        #     if 'ZMQInteractiveShell' in shell:
        #         # Code specific to Jupyter Lab
        #         fig, ax = plt.subplots(num_classes, 2)
        #         plt.suptitle("IOU")
        #         plt.tight_layout()
        #         for i in range(num_classes):
        #             plt.subplot(num_classes, 2, i+1)
        #             plt.imshow(mask1_class.astype(int))
        #             plt.title(f"mask1 {i+1}")
        #
        #             plt.subplot(num_classes, 2, i + 1 + num_classes)
        #             plt.imshow(mask2_class.astype(int))
        #             plt.title(f"mask2 {i+1}")
        # except NameError:
        #     # Code specific to other environments like PyCharm
        #     pass

        if union == 0:
            iou_scores[clas] = float("nan")
        else:
            iou_scores[clas] = intersection/union
    print(iou_scores)
    return iou_scores

def calculate_iou(mask1, mask2, num_classes):
    """

    :param mask1:
    :param mask2:
    :param num_classes:
    :return:
    """
    # sprawdzenie czy nie ma tylko jednej maski
    # mask1.shape[0] to ilość masek
    #
    # if len(mask1.shape) != (2, 512, 512):
    #     plt.subplots(1, 2)
    #     plt.suptitle("THE FIRST ONES")
    #     plt.subplot(1, 2, 1)
    #     plt.title("mask1")
    #     plt.imshow(mask1[0,:,:])
    #     plt.subplot(1, 2, 2)
    #     plt.title("mask2")
    #     plt.imshow(mask2[0,:,:])

    if len(mask1.shape) == 2:
        mask1 = mask1[np.newaxis, :, :]
        mask2 = mask2[np.newaxis, :, :]
    num_layers = mask1.shape[0]
    iou_scores = []

    # dla każdej maski obliczame iou
    for i in range(num_layers):
        # iou dodawane do listy iou
        iou_scores.append(calculate_iou_per_class(mask1[i], mask2[i], num_classes))

    # średnia wzdłuż pierwszej osi ignorując nan
    iou = np.nanmean(iou_scores, axis=0)
    avg_iou = np.nanmean(iou[1:])
    return avg_iou, iou

def iou(prediction, mask):

    return

def calculate_ap_for_segmentation(predicted_probabilities, true_labels):
    ap = average_precision_score(true_labels.flatten(), predicted_probabilities.flatten())
    print("calculate ap")
    return ap


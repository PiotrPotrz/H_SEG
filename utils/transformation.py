import numpy as np



def transform_mask(mask, number):
    new_mask = np.zeros((512, 512))
    if number == 1:
        for i in range(3):
            new_mask[mask[i] == 1] = i
    elif number == 2:
        for i in range(4):
            if i == 0:
                new_mask[mask[i] == 1] = i
            if i == 3:
                new_mask[mask[i] == 1] = 1
    # c
    return new_mask


def transform_batch(batch, number):
    new_batch = np.zeros((batch.shape[0], 512, 512))

    if np.isin(number, [1, 2]):
        for i in range(batch.shape[0]):
            new_batch[i] = transform_mask(batch[i], number)
        return new_batch
    else:
        raise ValueError("Number has to be 1 or 2 !!!")

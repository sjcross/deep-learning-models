import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import random
import matplotlib.pyplot as plt

import imageio as io

def gen(image_path, mask_path, image_size, image_channels, batch_size, num_classes):
    image_data_gen_args = dict(
                        # brightness_range=[0.3,1.4],
                        #  rotation_range=90,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         horizontal_flip=False,
                         vertical_flip=False,
                         zoom_range=0)
    image_datagen = ImageDataGenerator(**image_data_gen_args)
    mask_data_gen_args = dict(
                        #  rotation_range=90,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         horizontal_flip=False,
                         vertical_flip=False,
                         zoom_range=0)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)

    if image_channels == 1:
        color_mode = 'grayscale'
    elif image_channels == 3:
        color_mode = 'rgb'

    image_generator = image_datagen.flow_from_directory(
        image_path,
        class_mode=None,
        seed=42,
        batch_size=batch_size,
        color_mode=color_mode,
        target_size=image_size)

    mask_generator = mask_datagen.flow_from_directory(
        mask_path,
        class_mode=None,
        seed=42,
        batch_size=batch_size,
        color_mode='grayscale',
        target_size=image_size)


    # combine generators into one which yields image and masks
    while True:
        # Loading next image and mask
        next_im = image_generator.next()
        next_mask = mask_generator.next()

        # # Creating a weight array of the desired size
        # num_slices = next_mask.shape[0];
        # next_weight = np.reshape(np.copy(next_mask)[:,:,:,0],(num_slices,image_size[0]*image_size[1]))

        # # For each slice, count the number of sample for each class, then calculate the corresponding weights
        # for slice in range (0,num_slices):
        #     unique, counts = np.unique(next_mask[slice].astype(np.uint8), return_counts=True)
        #     # weights = dict(zip(unique,1/(1+np.log(counts))))
        #     weights = dict(zip(unique,image_size[0]*image_size[1]/(1+counts)))
            
        #     next_weight[slice] = [weights[val] for val in next_weight[slice].astype(np.uint8)]

        # Adapting the mask size
        if num_classes > 1:
            next_mask = to_categorical(next_mask,num_classes=num_classes,dtype=np.dtype('uint8'))
        # next_mask = np.reshape(next_mask,(num_slices,image_size[0]*image_size[1],num_classes))

        # print(next_im.shape,np.max(next_mask))
        # plt.subplot(1,2,1)
        # plt.imshow(next_im[0,:,:,0])
        # plt.subplot(1,2,2)
        # plt.imshow(next_mask[0,:,:,0])
        # plt.show()

        yield(next_im,next_mask)
        # yield(next_im,next_mask,next_weight)


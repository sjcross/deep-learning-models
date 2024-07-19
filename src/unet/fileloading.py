import numpy as np
from scipy import ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

def gen(image_path, mask_path, image_height, image_width, image_channels, batch_size, num_classes):
    image_data_gen_args = dict(
                        #  brightness_range=[0.5,1.4],
                         rotation_range=180,
                         shear_range=0.2,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=False,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**image_data_gen_args)
    mask_data_gen_args = dict(
                        #  brightness_range=[0.5,1.4],
                         rotation_range=180,
                         shear_range=0.2,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=False,
                         zoom_range=0.2)
    mask_datagen = ImageDataGenerator(**mask_data_gen_args)

    image_size=(image_height,image_width)

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

        # Adapting the mask size
        if num_classes > 1:
            next_mask = to_categorical(next_mask,num_classes=num_classes,dtype=np.dtype('uint8'))

        for i in range(next_mask.shape[0]):
            for j in range(next_mask.shape[3]):
                next_mask[i,:,:,j] = ndimage.median_filter(next_mask[i,:,:,j], size=4)

        yield(next_im,next_mask)

import numpy as np
from tensorflow.keras.preprocessing.image import apply_affine_transform, ImageDataGenerator
from tensorflow.keras.utils import to_categorical, Sequence

import matplotlib.pyplot as plt

class FileLoader(Sequence):
    def __init__(self, image_path, mask_path, image_height, image_width, image_depth, image_channels, num_classes, batch_size, shuffle=True):
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._shuffle = shuffle
        
        color_mode = 'grayscale'
        if image_channels == 3:
            color_mode = 'rgb'
            
        if image_depth <= 1:
            image_size=(image_height,image_width)
        else:
            image_size=(image_depth,image_height,image_width)
            
        image_generator = ImageDataGenerator().flow_from_directory(image_path, color_mode=color_mode, target_size=image_size, class_mode=None, shuffle=False)
        # batches = [next(image_generator) for _ in range(len(image_generator))]
        batches = [next(image_generator) for _ in range(2)]
        self._images = np.concatenate(batches, axis=0)
        
        mask_generator = ImageDataGenerator().flow_from_directory(mask_path, color_mode='grayscale', target_size=image_size, class_mode=None, shuffle=False)
        # batches = [next(mask_generator) for _ in range(len(mask_generator))]
        batches = [next(mask_generator) for _ in range(2)]
        self._masks = np.concatenate(batches, axis=0)
        
        self._indices = np.arange(len(self._images))
        self.on_epoch_end()
        
        self._datagen_args = dict(
                            rotation_range=10,
                            shear_range=0.1,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            horizontal_flip=False,
                            vertical_flip=False,
                            zoom_range=0.1,
                            dtype=np.uint8)
                
    def __len__(self):
        return int(np.ceil(len(self._images)/self._batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self._indices[idx*self._batch_size:(idx+1)*self._batch_size]
        batch_images = self._images[batch_indices]
        batch_masks = self._masks[batch_indices]

        self._datagen = ImageDataGenerator(**self._datagen_args)
        
        if self._shuffle:
            X_aug = np.empty_like(batch_images)
            Y_aug = np.empty_like(batch_masks)

            for i in range(len(batch_images)):
                # seed = np.random.randint(1e6)
                
                params = self._datagen.get_random_transform(batch_images[i].shape)#, seed=seed)
                
                X_aug[i] = apply_affine_transform(
                    batch_images[i],
                    theta=params['theta'],
                    tx=params['tx'],
                    ty=params['ty'],
                    shear=params['shear'],
                    zx=params['zx'],
                    zy=params['zy'],
                    row_axis=0,
                    col_axis=1,
                    channel_axis=2,
                    fill_mode='nearest',
                    order=1
                )
                
                Y_aug[i] = apply_affine_transform(
                    batch_masks[i],
                    theta=params['theta'],
                    tx=params['tx'],
                    ty=params['ty'],
                    shear=params['shear'],
                    zx=params['zx'],
                    zy=params['zy'],
                    row_axis=0,
                    col_axis=1,
                    channel_axis=2,
                    fill_mode='nearest',
                    order=0
                )
            
        else:
            X_aug = batch_images
            Y_aug = batch_masks
        
        Y_aug = np.squeeze(Y_aug, axis=-1)
        Y_aug = Y_aug.astype(np.int32)
        Y_aug = to_categorical(Y_aug, self._num_classes)
        # Y_aug = np.squeeze(Y_aug, axis=-1)
        # Y_aug = to_categorical(Y_aug, self._num_classes,dtype=np.dtype('uint8'))
        
        # plt.imshow(Y_aug[0,:,:,0])
        # plt.savefig("test.png")
        
        return X_aug, Y_aug
    
    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
            
    # def gen(self, num_classes):
    #     image_data_gen_args = dict(
    #                         # brightness_range=[0.9,1.1],
    #                         rotation_range=10,
    #                         shear_range=0.1,
    #                         width_shift_range=0.1,
    #                         height_shift_range=0.1,
    #                         horizontal_flip=False,
    #                         vertical_flip=False,
    #                         zoom_range=0.1)
    #     image_datagen = ImageDataGenerator(**image_data_gen_args)
    #     mask_data_gen_args = dict(
    #                         # Brightness range should stay the same
    #                         rotation_range=10,
    #                         shear_range=0.1,
    #                         width_shift_range=0.1,
    #                         height_shift_range=0.1,
    #                         horizontal_flip=False,
    #                         vertical_flip=False,
    #                         zoom_range=0.1)
    #     mask_datagen = ImageDataGenerator(**mask_data_gen_args)      

    #     image_generator = image_datagen.flow(
    #         self._images,
    #         seed=42,
    #         batch_size=self._batch_size)

    #     mask_generator = mask_datagen.flow(
    #         self._masks,
    #         seed=42,
    #         batch_size=self._batch_size)


    #     # combine generators into one which yields image and masks
    #     while True:
    #         # Loading next image and mask
    #         next_im = next(image_generator)
    #         next_mask = next(mask_generator)
            
    #         # Adapting the mask size
    #         if num_classes > 1:
    #             next_mask = to_categorical(next_mask,num_classes=num_classes,dtype=np.dtype('uint8'))

    #         yield(next_im,next_mask)

from __future__ import print_function, unicode_literals, absolute_import, division
from stardist.models import Config2D, StarDist2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import tensorflow

np.random.seed(42)
path = "C:\\Users\\steph\\Desktop\\test - Copy\\"
batch_size = 1
image_width = 320

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class SDGen(tensorflow.keras.Sequential):
    def __init__(self,image_path,image_size, batch_size):
        data_gen_args = dict(rotation_range=10,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         horizontal_flip=True,
                         vertical_flip=True,
                         zoom_range=0.2)
        
        self._gen = ImageDataGenerator(**data_gen_args).flow_from_directory(image_path,
            class_mode=None,
            seed=42,
            batch_size=batch_size,
            color_mode='grayscale',
            target_size=image_size)
        
    def __len__(self):
        return self._gen.__len__()

    def __getitem__(self,index):
        return self._gen.__getitem__(index)[0,:,:,0].astype(np.int8)

    def on_epoch_end(self):
        return self._gen.on_epoch_end()

train_image_generator = SDGen(path+"Train_raw\\",image_size=(image_width,image_width),batch_size=batch_size)
train_class_generator = SDGen(path+"Train_class\\",image_size=(image_width,image_width),batch_size=batch_size)
valid_image_generator = SDGen(path+"Valid_raw\\",image_size=(image_width,image_width),batch_size=batch_size)
valid_class_generator = SDGen(path+"Valid_class\\",image_size=(image_width,image_width),batch_size=batch_size)

conf = Config2D(n_channel_in=1, train_batch_size=4, train_shape_completion=False)
model = StarDist2D(conf, name='stardist_no_shape_completion', basedir='models')
model.train(train_image_generator,train_class_generator,validation_data=(valid_image_generator,valid_class_generator))
# model.load_weights("C:\\Users\\steph\\Documents\\Programming\\Python Projects\\trainstardist\\models\\stardist_no_shape_completion\\weights_now.h5")
model.export_TF("C:\\Users\\steph\\Desktop\\testmodel2.zip")

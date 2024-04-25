import argparse
import math
import os
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from unet import *
from fileloading import gen

# Initialising the system
num_classes = 1
batch_size = 4
epochs = 200
# image_width = 800
# image_height = 800 
seed = 2023
random.seed = seed
tf.seed = seed

parser = argparse.ArgumentParser()

required = parser.add_argument_group('required arguments')
required.add_argument("-p", "--path", type=str, required=True)
required.add_argument("-iw", "--im_width", type=int, required=True)
required.add_argument("-ih", "--im_height", type=int, required=True)

optional = parser.add_argument_group('optional arguments')
required.add_argument("-ic", "--im_channels", type=int, required=False, default=1)

args = parser.parse_args()

root_path = args.path
image_width = args.im_width
image_height = args.im_height
image_channels = args.im_channels

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Setting main parameters
# root_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\"
# root_path = "Z:\\Stephen\\People\\T\\Nathalie Tarassova\\2023-07-31 Angiogenesis analysis\\Training sets\\2023-08-01\\"
# model_path = "Z:\\Stephen\\People\\T\\Qiao Tong\\2022-10-06 DL scale segmentation\\2023-01-11_UNet_currentBest_E52_Acc0.984_ValLoss0.024.hdf5"
model_path = None

path, dirs, files = next(os.walk(root_path+"train_raw\\class1"))
train_size = math.ceil(len(files)/batch_size)
path, dirs, files = next(os.walk(root_path+"valid_raw\\class1"))
val_size = math.ceil(len(files)/batch_size)

train_generator = gen(root_path+"train_raw\\",root_path+"train_class\\",image_size=(image_width,image_height),image_channels=image_channels,batch_size=batch_size,num_classes=num_classes)
valid_generator = gen(root_path+"valid_raw\\",root_path+"valid_class\\",image_size=(image_width,image_height),image_channels=image_channels,batch_size=batch_size,num_classes=num_classes)
# test_generator = gen(root_path+"Test_raw\\",root_path+"Test_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)

model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_Valacc{val_acc:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_acc',verbose=1, save_best_only=True)
tboard = TensorBoard(log_dir="log",histogram_freq=0, write_graph=True, write_images=False)

model = UNetModel(image_width, image_height,image_channels=image_channels,num_classes=num_classes)
if (num_classes == 1):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=6["acc"], sample_weight_mode="temporal")
else:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"], sample_weight_mode="temporal")

if model_path is not None:
    model.load_weights(model_path)

model.fit_generator(
    generator=train_generator,
    validation_data=valid_generator,
    validation_steps=val_size,
    steps_per_epoch=train_size,
    steps_per_epoch=train_size,
    epochs=epochs,
    callbacks=[model_checkpoint])

model.save_weights(root_path+"UNet_final.hdf5")

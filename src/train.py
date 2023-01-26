import math
import os
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from models.unet import *
from utils.fileloading import gen

# Initialising the system
num_classes = 1
batch_size = 2
epochs = 200
image_width = 640
image_height = 640 
seed = 2023
random.seed = seed
tf.seed = seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Setting main parameters
# root_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\"
root_path = "C:\\Users\\sc13967\\Desktop\\2023-01-13 No TRAP images\\"
# model_path = "Z:\\Stephen\\People\\T\\Qiao Tong\\2022-10-06 DL scale segmentation\\2023-01-11_UNet_currentBest_E52_Acc0.984_ValLoss0.024.hdf5"
model_path = None

path, dirs, files = next(os.walk(root_path+"Train_raw\\Train"))
train_size = math.ceil(len(files)/batch_size)
path, dirs, files = next(os.walk(root_path+"Valid_raw\\Valid"))
val_size = math.ceil(len(files)/batch_size)

train_generator = gen(root_path+"Train_raw\\",root_path+"Train_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)
valid_generator = gen(root_path+"Valid_raw\\",root_path+"Valid_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)
# test_generator = gen(root_path+"Test_raw\\",root_path+"Test_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)

model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_IOU{mean_io_u:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='mean_io_u',verbose=1, save_best_only=True)
tboard = TensorBoard(log_dir="log",histogram_freq=0, write_graph=True, write_images=False)

model = UNetModel(image_width, image_height,num_classes=num_classes)
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
    epochs=epochs,
    callbacks=[model_checkpoint,tboard])

model.save_weights(root_path+"UNet_final.hdf5")

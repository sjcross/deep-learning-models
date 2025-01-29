import math
import os
import random
import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(1, "..\\models")
sys.path.insert(1, "..\\utils")

from RotationModel import *
from FileLoading import gen
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint

# Initialising the system
num_classes = 4
batch_size = 4
epochs = 50
image_width = 512
image_height = 512
seed = 2019
random.seed = seed
tf.seed = seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Setting main parameters
root_path = "C:\\Users\\sc13967\\Desktop\\20190508_deeplearning\\RotationMax\\"

train_count = 0
for path, dirs, files in os.walk(root_path+"Train\\"):
    train_count += len(files)
train_size = math.ceil(train_count/batch_size)

valid_count = 0
for path, dirs, files in os.walk(root_path+"Valid\\"):
    valid_count += len(files)
val_size = math.ceil(valid_count/batch_size)

print(" ")
print("Training size: ",train_size)
print("Validation size: ",val_size)
print(" ")

model = RotModel(image_width,image_height,1,num_classes)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

data_gen_args = dict(rescale = 1.0 / 255,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     horizontal_flip=False,
                     vertical_flip=False,
                     zoom_range=0.1)
train_datagen = ImageDataGenerator(**data_gen_args)
test_datagen = ImageDataGenerator(**data_gen_args)

train_generator = train_datagen.flow_from_directory(
    root_path+"Train\\",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

valid_generator = test_datagen.flow_from_directory(
    root_path+"Valid\\",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

filepath = root_path+"BestRotationModel.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit_generator(
    generator=train_generator,
    validation_data=valid_generator,
    validation_steps=val_size,
    steps_per_epoch=train_size,
    epochs=epochs,
    callbacks=callbacks_list)

model.save_weights(root_path+"RotationModel.h5")

# (image,mask,weight) = next(test_generator)
# result = model.predict(image)
#
# num_slices = mask.shape[0];
# mask = np.reshape(mask,(num_slices,image_width,image_height,num_classes))
# result = np.reshape(result,(num_slices,image_width,image_height,num_classes))
#
# fig = plt.figure()
# fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
# ax = fig.add_subplot(2, 3, 1)
# ax.imshow(np.reshape(mask[0,:,:,0]*255, (image_width,image_height)), cmap="gray")
# ax = fig.add_subplot(2, 3, 2)
# ax.imshow(np.reshape(mask[0,:,:,1]*255, (image_width,image_height)), cmap="gray")
# ax = fig.add_subplot(2, 3, 3)
# ax.imshow(np.reshape(mask[0,:,:,2]*255, (image_width,image_height)), cmap="gray")
# ax = fig.add_subplot(2, 3, 4)
# ax.imshow(np.reshape(result[0,:,:,0]*255, (image_width,image_height)), cmap="gray")
# ax = fig.add_subplot(2, 3, 5)
# ax.imshow(np.reshape(result[0,:,:,1]*255, (image_width,image_height)), cmap="gray")
# ax = fig.add_subplot(2, 3, 6)
# ax.imshow(np.reshape(result[0,:,:,2]*255, (image_width,image_height)), cmap="gray")
#
# plt.show()

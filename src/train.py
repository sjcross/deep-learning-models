import math
import os
import random

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ModelCheckpoint
from models.unet import *
from utils.fileloading import gen

# Initialising the system
num_classes = 2
batch_size = 5
epochs = 2
image_width = 400
image_height = 400
seed = 2019
random.seed = seed
tf.seed = seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Setting main parameters
root_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\"

path, dirs, files = next(os.walk(root_path+"Train_raw\\Train"))
train_size = math.ceil(len(files)/batch_size)
path, dirs, files = next(os.walk(root_path+"Valid_raw\\Valid"))
val_size = math.ceil(len(files)/batch_size)

print(" ")
print("Training size: ",train_size)
print("Validation size: ",val_size)
print(" ")

train_generator = gen(root_path+"Train_raw\\",root_path+"Train_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)
valid_generator = gen(root_path+"Valid_raw\\",root_path+"Valid_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)
test_generator = gen(root_path+"Test_raw\\",root_path+"Test_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)

model_checkpoint = ModelCheckpoint('UNet_currentBest.hdf5', monitor='loss',verbose=1, save_best_only=True)

model = UNetModel(image_width, image_height,num_classes=num_classes)
if (num_classes == 1):
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    # model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"], sample_weight_mode="temporal")
else:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    # model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"], sample_weight_mode="temporal")

model.fit_generator(
    generator=train_generator,
    validation_data=valid_generator,
    validation_steps=val_size,
    steps_per_epoch=train_size,
    epochs=epochs,
    callbacks=[model_checkpoint])

model.save_weights(root_path+"UNetW.h5")
# model.save('save_model.pb') # For BioImage.io format

(image,mask,weight) = next(test_generator)

result = model.predict(image)

num_slices = mask.shape[0];
mask = np.reshape(mask,(num_slices,image_width,image_height,num_classes))
result = np.reshape(result,(num_slices,image_width,image_height,num_classes))

fig = plt.figure()
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(mask[0,:,:,1]*255, (image_width,image_height)), cmap="gray")
ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0,:,:,1]*255, (image_width,image_height)), cmap="gray")

plt.show()

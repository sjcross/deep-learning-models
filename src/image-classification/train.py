import argparse

parser = argparse.ArgumentParser()

required = parser.add_argument_group('required arguments')
required.add_argument("-p", "--path", type=str, required=True)
required.add_argument("-iw", "--im_width", type=int, required=True)
required.add_argument("-ih", "--im_height", type=int, required=True)

optional = parser.add_argument_group('optional arguments')
optional.add_argument("-ic", "--im_channels", type=int, required=False, default=1)
optional.add_argument("-nc", "--num_classes", type=int, required=False, default=1)
optional.add_argument("-bs", "--batch_size", type=int, required=False, default=1)
optional.add_argument("-e", "--epochs", type=int, required=False, default=1000)
optional.add_argument("-mp", "--model_path", type=str, required=False, default=None)

args = parser.parse_args()

root_path = args.path
image_width = args.im_width
image_height = args.im_height
image_channels = args.im_channels
num_classes = args.num_classes
batch_size = args.batch_size
epochs = args.epochs
model_path = args.model_path


# The main imports
import math
import os
import random

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import ImageClassificationModel

# Initialising the system
seed = 2023
random.seed = seed
tf.seed = seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

train_count = 0
for path, dirs, files in os.walk(root_path+"Train\\"):
    train_count += len(files)
train_size = math.ceil(train_count/batch_size)

valid_count = 0
for path, dirs, files in os.walk(root_path+"Valid\\"):
    valid_count += len(files)
val_size = math.ceil(valid_count/batch_size)

data_gen_args = dict(rotation_range=180,
                     shear_range=0.2,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     horizontal_flip=True,
                     vertical_flip=True,
                     zoom_range=0.2)
train_datagen = ImageDataGenerator(**data_gen_args)
test_datagen = ImageDataGenerator(**data_gen_args)

train_generator = train_datagen.flow_from_directory(
    root_path+"train\\",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

valid_generator = test_datagen.flow_from_directory(
    root_path+"valid\\",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

model_checkpoint = ModelCheckpoint('Classification_currentBest_E{epoch}_CatAcc{categorical_accuracy:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='categorical_accuracy',verbose=1, save_best_only=True)
tboard = TensorBoard(log_dir="log",histogram_freq=0, write_graph=True, write_images=False)

model = ImageClassificationModel(image_height,image_width,image_channels=image_channels,num_classes=num_classes)
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

if model_path is not None:
    model.load_weights(model_path)

model.fit_generator(
    generator=train_generator,
    validation_data=valid_generator,
    validation_steps=val_size,
    steps_per_epoch=train_size,
    epochs=epochs,
    callbacks=[model_checkpoint])

model.save_weights(root_path+"Classification_final.hdf5")

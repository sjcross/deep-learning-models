import argparse

parser = argparse.ArgumentParser()

required = parser.add_argument_group('required arguments')
required.add_argument("-p", "--path", type=str, required=True)
required.add_argument("-iw", "--im_width", type=int, required=True)
required.add_argument("-ih", "--im_height", type=int, required=True)

optional = parser.add_argument_group('optional arguments')
optional.add_argument("-id", "--im_depth", type=int, required=False, default=1)
optional.add_argument("-ic", "--im_channels", type=int, required=False, default=1)
optional.add_argument("-nc", "--num_classes", type=int, required=False, default=1)
optional.add_argument("-bs", "--batch_size", type=int, required=False, default=1)
optional.add_argument("-e", "--epochs", type=int, required=False, default=1000)
optional.add_argument("-mp", "--model_path", type=str, required=False, default=None)
optional.add_argument("-w", "--weighted", action="store_true")

args = parser.parse_args()

root_path = args.path
image_width = args.im_width
image_height = args.im_height
image_depth = args.im_depth
image_channels = args.im_channels
num_classes = args.num_classes
batch_size = args.batch_size
epochs = args.epochs
model_path = args.model_path
weighted = args.weighted


# The main imports
import math
import os
import random

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from unet import *
from fileloading import gen

# Initialising the system
seed = 2023
random.seed = seed
tf.seed = seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

path, dirs, files = next(os.walk(root_path+"train_raw\\class1"))
train_size = math.ceil(len(files)/batch_size)
path, dirs, files = next(os.walk(root_path+"valid_raw\\class1"))
val_size = math.ceil(len(files)/batch_size)

train_generator = gen(root_path+"train_raw\\",root_path+"train_class\\",image_height=image_height,image_width=image_width,image_depth=image_depth,image_channels=image_channels,batch_size=batch_size,num_classes=num_classes)
valid_generator = gen(root_path+"valid_raw\\",root_path+"valid_class\\",image_height=image_height,image_width=image_width,image_depth=image_depth,image_channels=image_channels,batch_size=batch_size,num_classes=num_classes)

if num_classes == 1:
    model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_Acc{acc:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
else:
    model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_CatAcc{categorical_accuracy:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

tboard = TensorBoard(log_dir="log",histogram_freq=0, write_graph=True, write_images=False)

model = UNetModel(image_height,image_width,image_channels=image_channels,num_classes=num_classes)

if num_classes == 1:
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
else:
    if weighted:
        def dice_coefficient(y_true, y_pred, smooth=1):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def weighted_categorical_crossentropy(weights):
            weights = K.variable(weights)
        
            def loss(y_true, y_pred):
                # Scale predictions so that the class probabilities of each sample sum to 1
                y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            
                # Clip predictions to prevent log(0)
                y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            
                # Calculate the loss
                loss = y_true * K.log(y_pred) * weights
            
                # Return the mean loss over all sampless
                return -K.sum(loss, -1)

            return loss

        class_weights = [0.01]
        for i in range(num_classes-1):
            class_weights.append(1.0)
        
        print(class_weights)

        model.compile(optimizer="adam", loss=weighted_categorical_crossentropy(class_weights), metrics=["categorical_accuracy", dice_coefficient])

    else:
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

if model_path is not None:
    model.load_weights(model_path)

model.fit_generator(
    generator=train_generator,
    validation_data=valid_generator,
    validation_steps=val_size,
    steps_per_epoch=train_size,
    epochs=epochs,
    callbacks=[model_checkpoint])

model.save_weights(root_path+"UNet_final.hdf5")

args = parser.parse_args()

root_path = args.path
image_width = args.im_width
image_height = args.im_height
image_depth = args.im_depth
image_channels = args.im_channels
num_classes = args.num_classes
batch_size = args.batch_size
epochs = args.epochs
model_path = args.model_path
weighted = args.weighted


# The main imports
import math
import os
import random

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from unet import *
from fileloading import gen

# Initialising the system
seed = 2023
random.seed = seed
tf.seed = seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

path, dirs, files = next(os.walk(root_path+"train_raw\\class1"))
train_size = math.ceil(len(files)/batch_size)
path, dirs, files = next(os.walk(root_path+"valid_raw\\class1"))
val_size = math.ceil(len(files)/batch_size)

train_generator = gen(root_path+"train_raw\\",root_path+"train_class\\",image_height=image_height,image_width=image_width,image_depth=image_depth,image_channels=image_channels,batch_size=batch_size,num_classes=num_classes)
valid_generator = gen(root_path+"valid_raw\\",root_path+"valid_class\\",image_height=image_height,image_width=image_width,image_depth=image_depth,image_channels=image_channels,batch_size=batch_size,num_classes=num_classes)

if num_classes == 1:
    model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_Acc{acc:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
else:
    model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_CatAcc{categorical_accuracy:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

tboard = TensorBoard(log_dir="log",histogram_freq=0, write_graph=True, write_images=False)

model = UNetModel(image_height,image_width,image_channels=image_channels,num_classes=num_classes)

if num_classes == 1:
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
else:
    if weighted:
        def dice_coefficient(y_true, y_pred, smooth=1):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            intersection = K.sum(y_true_f * y_pred_f)
            return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def weighted_categorical_crossentropy(weights):
            weights = K.variable(weights)
        
            def loss(y_true, y_pred):
                # Scale predictions so that the class probabilities of each sample sum to 1
                y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            
                # Clip predictions to prevent log(0)
                y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            
                # Calculate the loss
                loss = y_true * K.log(y_pred) * weights
            
                # Return the mean loss over all sampless
                return -K.sum(loss, -1)

            return loss

        class_weights = [0.01]
        for i in range(num_classes-1):
            class_weights.append(1.0)
        
        print("Class weights "+class_weights)

        model.compile(optimizer="adam", loss=weighted_categorical_crossentropy(class_weights), metrics=["validation_loss", dice_coefficient])

    else:
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

if model_path is not None:
    model.load_weights(model_path)

model.fit_generator(
    generator=train_generator,
    validation_data=valid_generator,
    validation_steps=val_size,
    steps_per_epoch=train_size,
    epochs=epochs,
    callbacks=[model_checkpoint])

model.save_weights(root_path+"UNet_final.hdf5")

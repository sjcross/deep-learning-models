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

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from unet import UNetModel
from fileloading import FileLoader

# Initialising the system
seed = 2023
random.seed = seed
tf.seed = seed

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

path, dirs, files = next(os.walk(os.path.join(root_path,"train_raw","class1")))
train_size = math.ceil(len(files)/batch_size)
path, dirs, files = next(os.walk(os.path.join(root_path,"valid_raw","class1")))
val_size = math.ceil(len(files)/batch_size)

print("Loading files into memory")
train_file_loader = FileLoader(os.path.join(root_path,"train_raw"),os.path.join(root_path,"train_class"),image_height=image_height,image_width=image_width,image_depth=image_depth,image_channels=image_channels, num_classes=num_classes, batch_size=batch_size, shuffle=True)
# train_generator = train_file_loader.gen(batch_size=batch_size,num_classes=num_classes)
valid_file_loader = FileLoader(os.path.join(root_path,"valid_raw"),os.path.join(root_path,"valid_class"),image_height=image_height,image_width=image_width,image_depth=image_depth,image_channels=image_channels, num_classes=num_classes, batch_size=batch_size, shuffle=False)
# valid_generator = valid_file_loader.gen(batch_size=batch_size,num_classes=num_classes)

if num_classes == 1:
    model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_Acc{acc:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
else:
    model_checkpoint = ModelCheckpoint('UNet_currentBest_E{epoch}_CatAcc{categorical_accuracy:.3f}_ValLoss{val_loss:.3f}.hdf5', monitor='val_loss',verbose=1, save_best_only=True)

tboard = TensorBoard(log_dir="log",histogram_freq=0, write_graph=True, write_images=False)

model = UNetModel(image_height,image_width,image_channels=image_channels,num_classes=num_classes)

if num_classes == 1:
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
else:
    # if weighted:
    #     def dice_coefficient(y_true, y_pred, smooth=1):
    #         y_true_f = K.flatten(y_true)
    #         y_pred_f = K.flatten(y_pred)
    #         intersection = K.sum(y_true_f * y_pred_f)
    #         return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    #     def weighted_categorical_crossentropy(weights):
    #         weights = K.variable(weights)
        
    #         def loss(y_true, y_pred):
    #             # Scale predictions so that the class probabilities of each sample sum to 1
    #             y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            
    #             # Clip predictions to prevent log(0)
    #             y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            
    #             # Calculate the loss
    #             loss = y_true * K.log(y_pred) * weights
            
    #             # Return the mean loss over all sampless
    #             return -K.sum(loss, -1)

    #         return loss

    #     masks = next(train_generator)[1]
    #     class_weights = np.zeros(masks.shape[3])
    #     for slice in range(masks.shape[0]):
    #         for class_idx in range(masks.shape[3]):
    #             class_weights[class_idx] = class_weights[class_idx] + np.sum(masks[slice,:,:,class_idx])        
        
    #     for class_idx in range(masks.shape[3]):
    #             class_weights[class_idx] = 1/(1+class_weights[class_idx])
                
    #     print(f'Using weights {class_weights}')
        
    #     model.compile(optimizer="adam", loss=weighted_categorical_crossentropy(class_weights), metrics=["categorical_accuracy", dice_coefficient])

    # else:
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

if model_path is not None:
    model.load_weights(model_path)

val_size = 20
train_size = 20
model.fit_generator(
    generator=train_file_loader,
    validation_data=valid_file_loader,
    # validation_steps=val_size,
    # steps_per_epoch=train_size,
    epochs=epochs,
    callbacks=[model_checkpoint],
    use_multiprocessing=False,
    workers=4,
    max_queue_size=64)

model.save_weights(root_path+"UNet_final.hdf5")

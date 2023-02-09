import os

from predictor import Predictor
from unet import *
from fileloading import gen

# Setting parameters
num_classes = 1
image_width = 512
image_height = 512
model_path = "C:\\Users\\steph\\Documents\\People\\Christoph Wuelfing\\2022-11-23 SpinSR cell intensity analysis\\2023-02-09 Training 1\\UNet_currentBest_E25_acc0.986_ValLoss0.045.hdf5"
input_image_path = "C:\\Users\\steph\\Documents\\People\\Christoph Wuelfing\\2022-11-23 SpinSR cell intensity analysis\\2023-02-09 Test images 512\\"
output_image_path = "C:\\Users\\steph\\Documents\\People\\Christoph Wuelfing\\2022-11-23 SpinSR cell intensity analysis\\Test output\\"
# input_image_path = "C:\\Users\\steph\\Desktop\\test\\"
# output_image_path = "C:\\Users\\steph\\Desktop\\testout\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_width, image_height, model_path)
predictor.process_folder(input_image_path, output_image_path)

# model = UNetModel(image_width, image_height,num_classes=num_classes)

# train_generator = gen(root_path+"Train_raw\\",root_path+"Train_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)
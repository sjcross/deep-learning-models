import os

from predictor import Predictor
from unet import *
from fileloading import gen

# Setting parameters
num_classes = 3
image_width = 640
image_height = 640
model_path = "Z:\\Stephen\\People\\V\\Jordan Vautrinot\\2023-04-19 Cell analysis\\Chip analysis\\Models\\2023-07-20 Chip class\\UNet_currentBest_E124_acc0.969_ValLoss0.033.hdf5"
input_image_path = "Z:\\Stephen\\People\\V\\Jordan Vautrinot\\2023-04-19 Cell analysis\\Chip analysis\\Training sets\\2023-07-18 Two class whole\\Test_raw\\"
output_image_path = "Z:\\Stephen\\People\\V\\Jordan Vautrinot\\2023-04-19 Cell analysis\\Chip analysis\\Training sets\\2023-07-18 Two class whole\\Test_output\\"
# input_image_path = "C:\\Users\\steph\\Desktop\\test\\"
# output_image_path = "C:\\Users\\steph\\Desktop\\testout\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_width, image_height, model_path)
predictor.process_folder(input_image_path, output_image_path)

# model = UNetModel(image_width, image_height,num_classes=num_classes)

# train_generator = gen(root_path+"Train_raw\\",root_path+"Train_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)

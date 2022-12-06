import os 

from utils.predictor import Predictor

# Setting parameters
num_classes = 2
image_width = 1280
image_height = 1280
model_path = "C:\\Users\\steph\\Documents\\Programming\\Python Projects\\deep-learning-models\\UNet_currentBest.hdf5"
input_image_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\Test_raw\\"
output_image_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\Test_out\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_width, image_height, model_path)
predictor.process_folder(input_image_path, output_image_path)
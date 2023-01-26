import os

from utils.predictor import Predictor

# Setting parameters
num_classes = 1
image_width = 640
image_height = 640
model_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\2022-12-13_UNet_Scale_640px_16-256.hdf5"
input_image_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\Test_raw\\"
output_image_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\Test_out\\"
# input_image_path = "C:\\Users\\steph\\Desktop\\test\\"
# output_image_path = "C:\\Users\\steph\\Desktop\\testout\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_width, image_height, model_path)
predictor.process_folder(input_image_path, output_image_path)

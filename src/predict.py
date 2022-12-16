import os

from utils.predictor import Predictor

# Setting parameters
num_classes = 1
image_width = 640
image_height = 640
# model_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\2022-12-13_UNet_Scale_640px_16-256.hdf5"
# input_image_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\Test_raw\\"
# output_image_path = "C:\\Users\\steph\\Documents\\People\\Qiao Tong\\2022-10-06 DL scale segmentation\\TIF\\Test_out\\"

model_path = "F:\\Python Projects\\deep-learning-models\\UNet_currentBest_E7_Acc0.962_ValLoss0.056.hdf5"
input_image_path = "C:\\Users\\sc13967\\Desktop\\2022-12-15 Scale training images\\Test_raw\\"
output_image_path = "C:\\Users\\sc13967\\Desktop\\2022-12-15 Scale training images\\Test_out\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_width, image_height, model_path)
predictor.process_folder(input_image_path, output_image_path)

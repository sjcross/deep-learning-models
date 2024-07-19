import os

from predictor import Predictor
from unet import *

# Setting parameters
num_classes = 5
image_width = 1024
image_height = 512
model_path = "Z:\\Stephen\\People\\M\\Jennyfer Mitchell\\Models\\2024-05-03\\Raw\\UNet_currentBest_E26_Valacc0.874_ValLoss0.260.hdf5"
input_image_path = "C:\\Users\\sc13967\\Desktop\\Jennyfer\\2024-05-03 Set 2 (3 classes)\\valid_raw\\"
output_image_path = "C:\\Users\\sc13967\\Desktop\\Jennyfer\\2024-05-03 Set 2 (3 classes)\\test_output\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_height, image_width, model_path)
predictor.process_folder(input_image_path, output_image_path)


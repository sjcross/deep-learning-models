import os

from utils.predictor import Predictor

# Setting parameters
num_classes = 2
image_width = 1280
image_height = 1280
model_path = "C:\\Users\\sc13967\\Desktop\\2022-12-02 Qiao\\UNetW.h5"
input_image_path = "C:\\Users\\sc13967\\Desktop\\2022-12-02 Qiao\\Test_raw\\"
output_image_path = "C:\\Users\\sc13967\\Desktop\\2022-12-02 Qiao\\Test_out\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_width, image_height, model_path)
predictor.process_folder(input_image_path, output_image_path)
import os

from predictor import Predictor
from unet import *
from fileloading import gen

# Setting parameters
num_classes = 5
image_width = 1024
image_height = 512
model_path = "Z:\\Stephen\\People\\M\\Jennyfer Mitchell\\Models\\2024-05-03\\Raw\\UNet_currentBest_E26_Valacc0.874_ValLoss0.260.hdf5"
input_image_path = "C:\\Users\\sc13967\\Desktop\\Jennyfer\\2024-05-03 Set 2 (3 classes)\\valid_raw\\"
output_image_path = "C:\\Users\\sc13967\\Desktop\\Jennyfer\\2024-05-03 Set 2 (3 classes)\\test_output\\"
# input_image_path = "C:\\Users\\steph\\Desktop\\test\\"
# output_image_path = "C:\\Users\\steph\\Desktop\\testout\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

predictor = Predictor(num_classes, image_height, image_width, model_path)
predictor.process_folder(input_image_path, output_image_path)

# model = UNetModel(image_width, image_height,num_classes=num_classes)

# train_generator = gen(root_path+"Train_raw\\",root_path+"Train_class\\",image_size=(image_width,image_height),batch_size=batch_size,num_classes=num_classes)

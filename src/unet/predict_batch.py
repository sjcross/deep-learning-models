import os

from utils.predictor import Predictor

# Setting parameters
num_classes = 1
image_width = 640
image_height = 640
# model_path = "Z:\\Stephen\\People\\T\\Qiao Tong\\2022-10-06 DL scale segmentation\\Models\\2023-01-24_(8-128)"
model_path = "F:\\Python Projects\\deep-learning-models\\"
input_image_path = "C:\\Users\\sc13967\\Desktop\\2023-01-13 No TRAP images\\Test_raw\\"
input_mask_path = "C:\\Users\\sc13967\\Desktop\\2023-01-13 No TRAP images\\Test_class\\"
output_image_path = "C:\\Users\\sc13967\\Desktop\\2023-01-13 No TRAP images\\Test_out\\"
# input_image_path = "C:\\Users\\steph\\Desktop\\test\\"
# output_image_path = "C:\\Users\\steph\\Desktop\\testout\\"

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
for subdir, dir, files in os.walk(model_path):     
    for file in files:        
        (fname,ext) = os.path.splitext(file)     
        if ext == ".hdf5":
            predictor = Predictor(num_classes, image_width, image_height, os.path.join(model_path,subdir,file))            
            predictor.process_folder(input_image_path, os.path.join(output_image_path,os.path.basename(subdir),fname),input_mask_path=input_mask_path)
            dice = predictor.get_dice()
            print("%.4f %s"%(dice,os.path.basename(subdir)+"_"+fname))
import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tifffile import TiffWriter
from scipy.spatial import distance
from PIL import Image

from unet import UNetModel


class Predictor():
    def __init__(self, num_classes, image_height, image_width, model_path):
        self._num_classes = num_classes
        self._image_height = image_height
        self._image_width = image_width
        
        self._model = UNetModel(image_height=image_height, image_width=image_width, num_classes=num_classes)           
        self._model.load_weights(model_path)     

        self._prev_image = np.zeros((1, 1))
        self._prev_filename = ""
        self._dice_sum = 0
        self._dice_n = 0

    def process_folder(self, input_image_path, output_image_path, input_mask_path=None):        
        # Creating output folder and tif writers
        if not os.path.exists(output_image_path):
            os.makedirs(output_image_path)

        root_filename = output_image_path + os.path.splitext(input_image_path.split("\\")[-2])[0]
        
        # Initialising ImageDataGenerator
        image_data_gen_args = dict()
        image_datagen = ImageDataGenerator(**image_data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            input_image_path,
            class_mode=None,
            batch_size=1,
            color_mode='grayscale',
            shuffle=False,
            target_size=(self._image_height, self._image_width))

        if os.path.exists("%s.tif" % root_filename):
            os.remove("%s.tif" % root_filename)

        tif_options = dict(compress=5)
        
        for filename in image_generator.filenames:
            image = image_generator.next()

            result = self._model.predict(image)

            # image = np.reshape(image, (image.shape[0], self._image_width, self._image_height, 1))
            image = (((image - np.min(image))/(np.max(image) - np.min(image)))*255).astype(np.uint8)
            result = np.reshape(result, (image.shape[0], self._image_height, self._image_width, self._num_classes))

            raw_im = plt.imread(input_image_path+filename)
            print(result.shape)
            result = resize(result[0, :, :, 1],(raw_im.shape[0],raw_im.shape[1]))

            out_name = filename.split("\\")[-1]
            out_name = os.path.splitext(out_name)[0]

            if input_mask_path is not None:
                mask_im = plt.imread(input_mask_path+filename)
                result_1d = result.flatten() > 0.5
                mask_im_1d = mask_im.flatten() > 0.5
                dice = distance.dice(mask_im_1d,result_1d)                
                self._dice_n = self._dice_n + 1
                self._dice_sum = self._dice_sum + dice
                
                rgbArray = np.zeros((raw_im.shape[0],raw_im.shape[1],3), 'uint8')
                rgbArray[..., 0] = (result>0.5)*255
                rgbArray[..., 1] = mask_im*255                
                img = Image.fromarray(rgbArray)
                img.save(output_image_path+"\\"+out_name+("_D%.4f"%dice)+".png")
                
            else:
                raw_tif = TiffWriter(output_image_path+out_name+".tif", bigtiff=False, append=False)
                raw_tif.save((result * 255).astype(np.uint8), **tif_options)                            
                raw_tif.close()            
            
    def get_dice(self):
        return self._dice_sum/self._dice_n

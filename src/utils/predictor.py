import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tifffile import TiffWriter

from models.unet import UNetModel


class Predictor():
    def __init__(self, num_classes, image_width, image_height, model_path):
        self._num_classes = num_classes
        self._image_width = image_width
        self._image_height = image_height

        self._model = UNetModel(image_width, image_height, num_classes=num_classes)           
        self._model.load_weights(model_path)     

        self._prev_image = np.zeros((1, 1))
        self._prev_filename = ""

    def process_folder(self, input_image_path, output_image_path):
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
            target_size=(self._image_width, self._image_height))

        if os.path.exists("%s.tif" % root_filename):
            os.remove("%s.tif" % root_filename)

        tif_options = dict(compress=5)
        
        total = len(image_generator.filenames)
        for (count, filename) in enumerate(image_generator.filenames):
            print("    Processing %i/%i (\"%s\")" % (count+1, total, filename), end="\r")
                        
            image = image_generator.next()
            
            result = self._model.predict(image)

            # image = np.reshape(image, (image.shape[0], self._image_width, self._image_height, 1))
            image = (((image - np.min(image))/(np.max(image) - np.min(image)))*255).astype(np.uint8)
            # result = np.reshape(result, (image.shape[0], self._image_width, self._image_height, self._num_classes))

            raw_im = plt.imread(input_image_path+filename)
            result = resize(result[0, :, :, 0],(raw_im.shape[0],raw_im.shape[1]))

            out_name = filename.split("\\")[-1]
            out_name = os.path.splitext(out_name)[0]
            raw_tif = TiffWriter(output_image_path+out_name+".tif", bigtiff=True, append=False)
            raw_tif.save((result * 255).astype(np.uint8), **tif_options)

        # Closing tif writers
        raw_tif.close()

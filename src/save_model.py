from tensorflow import saved_model
from tensorflow.keras.backend import get_session

from models.unet import UNetModel

model_path = "Z:\\Stephen\\People\\T\\Qiao Tong\\2022-10-06 DL scale segmentation\\2023-01-11_UNet_currentBest_E52_Acc0.984_ValLoss0.024.hdf5"
image_width = 640
image_height = 640
num_classes = 1

model = UNetModel(image_width, image_height, num_classes=num_classes)
model.load_weights(model_path)

# The following is from the page https://github.com/deepimagej/deepimagej-plugin/wiki/TensorFlow-models#tensorflow-models-in-deepimagej (accessed 2022-12-13)
builder = saved_model.builder.SavedModelBuilder('saved_model/2023-01-11_unet-scale-640')
signature = saved_model.signature_def_utils.predict_signature_def(
            # dictionary of 'name' and model inputs (it can have more than one)
            inputs={'input': model.input},
            # dictionary of 'name' and model outputs (it can have more than one)
            outputs={'output': model.output})
signature_def_map = {saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
builder.add_meta_graph_and_variables(get_session(), [saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
builder.save()
from models.unet import UNetModel
from tensorflow.keras.backend import get_session
from tensorflow import saved_model

model_path = "C:\\Users\\steph\\Documents\\Programming\\Python Projects\\deep-learning-models\\UNet_currentBest.hdf5"
image_width = 400
image_height = 400
num_classes = 2

model = UNetModel(image_width, image_height, num_classes=num_classes)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"], sample_weight_mode="temporal")
model.load_weights(model_path)

builder = saved_model.builder.SavedModelBuilder('saved_model/my_model')
signature = saved_model.signature_def_utils.predict_signature_def(
            # dictionary of 'name' and model inputs (it can have more than one)
            inputs={'input': model.input},
            # dictionary of 'name' and model outputs (it can have more than one)
            outputs={'output': model.output})
signature_def_map = {saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
builder.add_meta_graph_and_variables(get_session(), [saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
builder.save()
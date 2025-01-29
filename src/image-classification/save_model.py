import argparse

parser = argparse.ArgumentParser()
required = parser.add_argument_group('required arguments')
required.add_argument("-mp", "--model_path", type=str, required=True)
required.add_argument("-iw", "--im_width", type=int, required=True)
required.add_argument("-ih", "--im_height", type=int, required=True)
required.add_argument("-on", "--output_name", type=str, required=True)

optional = parser.add_argument_group('optional arguments')
required.add_argument("-ic", "--im_channels", type=int, required=False, default=1)
optional.add_argument("-nc", "--num_classes", type=int, required=False, default=1)

args = parser.parse_args()

model_path = args.model_path
image_width = args.im_width
image_height = args.im_height
image_channels = args.im_channels
output_name = args.output_name
num_classes = args.num_classes


# The main imports
from tensorflow import saved_model
from tensorflow.keras.backend import get_session

from model import ImageClassificationModel

model = ImageClassificationModel(image_height=image_height,image_width=image_width, image_channels=image_channels, num_classes=num_classes)
model.load_weights(model_path)

# The following is from the page https://github.com/deepimagej/deepimagej-plugin/wiki/TensorFlow-models#tensorflow-models-in-deepimagej (accessed 2022-12-13)
builder = saved_model.builder.SavedModelBuilder('saved_model/'+output_name)
signature = saved_model.signature_def_utils.predict_signature_def(
            # dictionary of 'name' and model inputs (it can have more than one)
            inputs={'input': model.input},
            # dictionary of 'name' and model outputs (it can have more than one)
            outputs={'output': model.output})
signature_def_map = {saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
builder.add_meta_graph_and_variables(get_session(), [saved_model.tag_constants.SERVING], signature_def_map=signature_def_map)
builder.save()
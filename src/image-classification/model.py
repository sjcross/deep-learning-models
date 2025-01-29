from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

def ImageClassificationModel(image_width,image_height,image_channels,num_classes):
	model = Sequential()

	# First Group
	model.add(Conv2D(32, (3, 3), input_shape=(image_width,image_height,image_channels)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Second Group
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Third Group
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Fourth Group
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))

	# Fifth Group
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('sigmoid'))

	return model

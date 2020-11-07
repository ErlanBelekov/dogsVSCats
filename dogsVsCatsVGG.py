import os, shutil

import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
# 
#   SET UP DATA
# 

curr_dir = os.path.dirname(os.path.realpath(__file__))
base_dir = '{}/cats_and_dogs_model_data'.format(curr_dir)
validation_dir = os.path.join(base_dir, 'validation')
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20

conv_base = VGG16(
	weights='imagenet',
	include_top=False,
	input_shape=(150, 150, 3)
)


def extract_features(directory, sample_count):
	features = np.zeros(shape=(sample_count, 4, 4, 512))
	labels = np.zeros(shape=(sample_count))
	generator = datagen.flow_from_directory(
		directory,
		target_size=(150, 150),
		batch_size=batch_size,
		class_mode='binary'
	)

	i = 0
	
	for inputs_batch, labels_batch in  generator:
		features_batch = conv_base.predict(inputs_batch)
		features[i * batch_size : (i + 1) * batch_size] = features_batch
		labels[i * batch_size : (i + 1) * batch_size] = labels_batch
		i += 1
		if i * batch_size >= sample_count:
			break
	
	return features, labels

train_features, train_labels = extract_features(train_dir, 2000)
validation_features, validation_labels = extract_features(validation_dir, 2000)
test_features, test_labels = extract_features(validation_dir, 2000)


train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
test_features = np.reshape(test_features, (2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features, (2000, 4 * 4 * 512))


model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(
	optimizer=optimizers.RMSprop(lr=1e-4),
	loss='binary_crossentropy',
	metrics=['acc']
)

history = model.fit(
	train_features,
	train_labels,
	epochs=30,
	batch_size=20,
	validation_data=(validation_features, validation_labels)
)



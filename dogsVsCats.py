import os, shutil

from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
# 
#   SET UP DATA
# 

curr_dir = os.path.dirname(os.path.realpath(__file__))

# original_dataset_data = '{}/train'.format(curr_dir)

base_dir = '{}/cats_and_dogs_model_data'.format(curr_dir)
# os.makedirs(base_dir)

validation_dir = os.path.join(base_dir, 'validation')

train_dir = os.path.join(base_dir, 'train')
# os.makedirs(train_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.makedirs(test_dir)
# validation_dir = os.path.join(base_dir, 'validation')
# os.makedirs(validation_dir)

# train_cats_dir = os.path.join(train_dir, 'cats')
# os.makedirs(train_cats_dir)

# train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.makedirs(train_dogs_dir)

# validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.makedirs(validation_cats_dir)

# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.makedirs(validation_dogs_dir)

# test_cats_dir = os.path.join(test_dir, 'cats')
# os.makedirs(test_cats_dir)

# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.makedirs(test_dogs_dir)

# # copying first 1000 cat images
# fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#   src = os.path.join(original_dataset_data, fname)
#   dst = os.path.join(train_cats_dir, fname)
#   shutil.copyfile(src, dst)

# fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#   src = os.path.join(original_dataset_data, fname)
#   dst = os.path.join(validation_cats_dir, fname)
#   shutil.copyfile(src, dst)

# fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#   src = os.path.join(original_dataset_data, fname)
#   dst = os.path.join(test_cats_dir, fname)
#   shutil.copyfile(src, dst)

# fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fname in fnames:
#   src = os.path.join(original_dataset_data, fname)
#   dst = os.path.join(train_dogs_dir, fname)
#   shutil.copyfile(src, dst)

# fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fname in fnames:
#   src = os.path.join(original_dataset_data, fname)
#   dst = os.path.join(validation_dogs_dir, fname)
#   shutil.copyfile(src, dst)

# fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fname in fnames:
#   src = os.path.join(original_dataset_data, fname)
#   dst = os.path.join(test_dogs_dir, fname)
#   shutil.copyfile(src, dst)

# print("total training cat images: ", len(os.listdir(train_cats_dir)))

# print("total training dog images: ", len(os.listdir(train_dogs_dir)))

# print("total validation cat images: ", len(os.listdir(validation_cats_dir)))

# print("total validation dog images: ", len(os.listdir(validation_dogs_dir)))

# print("total test cat images: ", len(os.listdir(test_cats_dir)))

# print("total test dog images: ", len(os.listdir(test_dogs_dir)))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150, 150),
	batch_size=20,
	class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(150, 150),
	batch_size=20,
	class_mode='binary'
)


history = model.fit_generator(
	train_generator,
	steps_per_epoch=100,
	epochs=30,
	validation_data=validation_generator,
	validation_steps=50
)

model.save('cats_and_dogs_1.h5')

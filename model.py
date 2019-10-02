from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
import numpy as np
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import os
import random
from shutil import copyfile

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

total_examples = 1120
examples_per_genre = 100
tt_split = 0.4
img_dir = "img_data"
img_height = 1000
img_width = 1000


# def create_train_test_valid():
#     file_names = [f for folder in os.listdir(img_dir) for f in os.listdir(img_dir + '/' + folder)]
#     file_names.sort()
#     test_list = []
#     train_list = []
#     valid_list = []
    
#     for i in range(int(total_examples/examples_per_genre)):
#         test_list.extend(random.sample(file_names[(i*examples_per_genre+1):(i+1)*examples_per_genre], int(examples_per_genre*tt_split)))
#     random.shuffle(test_list)
#     test_list, valid_list = test_list[:len(test_list)//2], test_list[len(test_list)//2:]
#     train_list = [x for x in file_names if x not in test_list and x not in valid_list]
    
#     try:
#         os.stat('train/')
#     except:
#         os.mkdir('train/')
#     try:
#         os.stat('test/')
#     except:
#         os.mkdir('test/')
#     try:
#         os.stat('valid/')
#     except:
#         os.mkdir('valid/')

#     for folder in os.listdir(img_dir):
#         try:
#             os.stat('train/' + folder)
#         except:
#             os.mkdir('train/' + folder)
#         try:
#             os.stat('test/' + folder)
#         except:
#             os.mkdir('test/' + folder)
#         try:
#             os.stat('valid/' + folder)
#         except:
#             os.mkdir('valid/' + folder)
#         for f in os.listdir(img_dir + '/' + folder):
#             if f in train_list:
#                 try:
#                     copyfile(img_dir + '/' + folder + '/' + f, 'train/' + folder + '/' + f)
#                 except IOError as e:
#                     print("Unable to copy file. %s" % e)
#             if f in test_list:
#                 try:
#                     copyfile(img_dir + '/' + folder + '/' + f, 'test/' + folder + '/' + f)
#                 except IOError as e:
#                     print("Unable to copy file. %s" % e)
#             if f in valid_list:
#                 try:
#                     copyfile(img_dir + '/' + folder + '/' + f, 'valid/' + folder + '/' + f)
#                 except IOError as e:
#                     print("Unable to copy file. %s" % e)


train_path = 'train/'
test_path = 'test/'
valid_path = 'valid/'
classes = ['blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(img_height, img_width), classes=classes, batch_size=1)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(img_height, img_width), classes=classes, batch_size=1)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(img_height, img_width), classes=classes, batch_size=1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(1000, 1000, 3), activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
print(model.summary())

model.fit_generator(train_batches, steps_per_epoch=1, epochs=1, validation_data=valid_batches, validation_steps=1)

model.save_weights('50_epochs.h5')

# model2 = Sequential()
# model2.add(Conv2D(32, kernel_size=(3, 3), input_shape=(1000, 1000, 3), activation="relu"))
# # model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Dense(128, activation='relu'))
# model2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model2.add(MaxPooling2D(pool_size=(2, 2)))
# model2.add(Flatten())
# model2.add(Dropout(0.25))
# model2.add(Dense(128, activation='relu'))
# model2.add(Dropout(0.5))
# model2.add(Dense(10, activation='softmax'))

# model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
# print(model2.summary())

# create_train_test_valid()
# W tensorflow/core/framework/allocator.cc:107] Allocation of 8126595072 exceeds 10% of system memory.
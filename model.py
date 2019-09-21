# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# import keras
import os
import random


total_examples = 1120
examples_per_genre = 100
tt_split = 0.2
img_dir = "img_data"


def create_train_test():
    file_names = [f for folder in os.listdir(img_dir) for f in os.listdir(img_dir + '/' + folder)]
    file_names.sort()
    test_list = []
    train_list = []
    
    for i in range(int(total_examples/examples_per_genre)):
        test_list.extend(random.sample(file_names[(i*examples_per_genre+1):(i+1)*examples_per_genre], int(examples_per_genre*tt_split)))

    train_list = [x for x in file_names if x not in test_list]
    print(len(train_list), len(test_list))
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(1000, 1000, 3), activation="relu"))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))


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

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
# print(model.summary())

# model2.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
# print(model2.summary())

create_train_test()
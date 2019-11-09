from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import keras
import numpy as np
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import os
import random
from shutil import copyfile
import contextlib
import wave
from scipy.io import wavfile
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from PIL import Image
import librosa

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

total_examples = 1120
examples_per_genre = 100
tt_split = 0.4
img_dir = "img_data"
img_height = 34
img_width = 50
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(10,10))


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

def findDuration(fname):
    with contextlib.closing(wave.open(fname,'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        sw   = f.getsampwidth()
        chan = f.getnchannels()
        duration = frames / float(rate)
        #print("File:", fname, "--->",frames, rate, sw, chan)
        return duration

def graph_spectrogram(wav_file, nfft=512, noverlap=511):
    findDuration(wav_file)
    rate, data = wavfile.read(wav_file)
    #print("")
    fig,ax = plt.subplots(1)
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    # pxx, freqs, bins, im = ax.specgram(x=data, Fs=rate, noverlap=noverlap, NFFT=nfft)
    ax.axis('off')
    plt.rcParams['figure.figsize'] = [0.75,0.5]
    #fig.savefig('sp_xyz.png', dpi=300, frameon='false')
    fig.canvas.draw()
    size_inches  = fig.get_size_inches()
    dpi          = fig.get_dpi()
    width, height = fig.get_size_inches() * fig.get_dpi()

    #print(size_inches, dpi, width, height)
    mplimage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    #print("MPLImage Shape: ", np.shape(mplimage))
    imarray = np.reshape(mplimage, (int(height), int(width), 3))
    plt.close(fig)
    return imarray

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#Normalize Gray colored image
def normalize_gray(array):
    return (array - array.min())/(array.max() - array.min())

def create_train_test(audio_dir):
    file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]
    file_names.sort()
    test_list = []
    train_list = []
    
    # for i in range(int(total_examples/examples_per_speaker)):
    #     test_list.extend(random.sample(file_names[(i*examples_per_speaker+1):(i+1)*examples_per_speaker], int(examples_per_speaker*tt_split)))

    for i in range(int(total_examples/examples_per_genre)):
        test_list.extend(random.sample(file_names[(i*examples_per_genre+1):(i+1)*examples_per_genre], int(examples_per_genre*tt_split)))
    random.shuffle(test_list)

    train_list = [x for x in file_names if x not in test_list]

    y_test = np.empty((len(test_list)), dtype=object)
    y_train = np.empty((len(train_list)), dtype=object)
    x_train = np.zeros((len(train_list), img_height, img_width))
    x_test = np.zeros((len(test_list), img_height, img_width))

    tuni1   = np.zeros(len(test_list))
    tuni2   = np.zeros(len(test_list))

    for i, f in enumerate(test_list):
        y_test[i]     = f
        y, sr = librosa.load(audio_dir + f, mono=True, duration=30)
        spectrogram = plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        # spectrogram   = graph_spectrogram( audio_dir + f )
        # img = Image.fromarray(spectrogram, 'RGB')
        plt.imshow(spectrogram)
        plt.show()
        graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(graygram)
        norm_shape    = normgram.shape
        if(norm_shape[0]>150):
            continue
        redgram       = block_reduce(normgram, block_size = (3,3), func = np.mean)
        
        x_test[i,:,:] = redgram
        print("Progress Test Data: {:2.1%}".format(float(i) / len(test_list)), end="\r")

    for i, f in enumerate(train_list):
        y_train[i] = f
        spectrogram   = graph_spectrogram( audio_dir + f )
        graygram      = rgb2gray(spectrogram)
        normgram      = normalize_gray(graygram)
        norm_shape    = normgram.shape
        if(norm_shape[0]>150):
            continue
        redgram       = block_reduce(normgram, block_size = (3,3), func = np.mean)
        x_train[i,:,:] = redgram
        print("Progress Training Data: {:2.1%}".format(float(i) / len(train_list)), end="\r")
        
    return x_train, y_train, x_test, y_test


path = 'data/'
train_path = 'train/'
test_path = 'test/'
valid_path = 'valid/'
classes = ['blues', 'classical', 'country', 'disco', 'hip-hop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(img_height, img_width), classes=classes, batch_size=1)
# test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(img_height, img_width), classes=classes, batch_size=1)
# valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(img_height, img_width), classes=classes, batch_size=1)

x_train, y_train, x_test, y_test = create_train_test(path)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.adam(), metrics=['accuracy'])
# print(model.summary())

# model.fit_generator(train_batches, steps_per_epoch=5, epochs=30, validation_data=valid_batches, validation_steps=5)
# model.fit(x_train, y_train, batch_size=4, epochs=10, verbose=1, validation_data=(x_test, y_test))

# model.save_weights('50_epochs.h5')

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
 
import pydub
import os
import librosa
import matplotlib.pyplot as plt

from fastai import *
from fastai.vision import *
import pathlib

p = pathlib.PureWindowsPath('models/')
p = p.as_posix()


cmap = plt.get_cmap('inferno')
plt.figure(figsize=(10,10))


def predictor(song):
    img = song_to_img(song)
    model = load_learner(p)
    return model.predict(img)


def song_to_img(song):
    if song.endswith('.mp3'):
        sound = pydub.AudioSegment.from_mp3(f'songs/{song}')
        song = song[-4] + '.wav'
        sound.export(f'songs/{song}', format="wav")
        y, sr = librosa.load(song, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB')
        plt.axis('off')
        plt.savefig(f'songs/{song[:-4]}.png', transparent=True)
        plt.clf()
    
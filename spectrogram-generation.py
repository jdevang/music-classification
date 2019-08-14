import librosa
import matplotlib.pyplot as plt
import os

cmap = plt.get_cmap('inferno')
plt.figure(figsize=(10,10))


# list of genres
genres = {
    "blues", 
    "classical",
    "country",
    "disco",
    "hip-hop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
}

try:
    os.stat('img_data/')
except:
    os.mkdir('img_data/')


for genre in genres:
    genre_path = 'data/' + genre
    genre_img_path = 'img_data/' + genre
    try:
        os.stat(genre_img_path)
    except:
        os.mkdir(genre_img_path)
    for filename in os.listdir(genre_path):
        songname = f'{genre_path}/{filename}'
        print(songname)
        y, sr = librosa.load(songname, mono=True, duration=5)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'{genre_img_path}/{filename[:-4]}.png', transparent=True)
        plt.clf()
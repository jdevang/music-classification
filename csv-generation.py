import csv
import os
import librosa
import numpy as np

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += ' mfcc{i}'
    header += ' label'
header = header.split()

file = open('features.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

# list of genres
genres = {
    "blues", 
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
}

for genre in genres:
    for filename in os.listdir(f'data/{genre}'):
        songname = f'data/{genre}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)
        
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        print(chroma_stft)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for item in mfcc:
            to_append += f' {np.mean(item)}'
        to_append += f' {genre}'
        print(filename)
        file = open('features.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
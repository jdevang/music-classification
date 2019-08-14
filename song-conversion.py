import pydub
import os

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

for genre in genres:
    genre_path = 'data/' + genre
    count = 0
    for filename in os.listdir(genre_path):
        count += 1
        print("converting to wav")
        print(f'MP3 File Name: {filename}')
        print(f'WAV File Name: {genre}{count}.wav')
        sound = pydub.AudioSegment.from_mp3(f'{genre_path}/{filename}')
        sound.export(f'{genre_path}/{genre}{count}.wav', format="wav")
        print("deleting mp3")
        os.remove(f'{genre_path}/{filename}')
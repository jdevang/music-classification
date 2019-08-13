from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import requests
import os
import pydub
# Create your own config.py containing client ID and secret
from config import CLIENT_ID, CLIENT_SECRET

# set up with client id and secret 
client_credentials_manager = SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET)

# Make spotipy object
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

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

dataset_size = 3


# get 100 recommendation song in every genre
# Note: won't give 100 recommendations since some songs don't have preview urls. It is advised to run this a few times so that new suggestions get added
# Note 2: There are many genres in the list by spotify, it would be a better idea to choose a few genres which are popular and varying and work on those rather than risk underfitting the data

try:
    os.stat('data')
except:
    os.mkdir('data') 

for genre in genres:
    print(genre)
    genre_path = f'data/{genre}'
    try:
        os.stat(genre_path)
    except:
        os.mkdir(genre_path)
    while len(os.listdir(genre_path)) < dataset_size:
        rec = sp.recommendations(seed_genres=[genre], limit = dataset_size)
        for track in rec['tracks']:
            track_name = track['name']
            track_url = track['preview_url']
            if track_url  and f'{track_name}.wav' not in os.listdir(genre_path):
                print('\t', track_name)
                r = requests.get(track_url)
                with open(f'{genre_path}/{track_name}.mp3','wb') as f:
                    f.write(r.content)
                sound = pydub.AudioSegment.from_mp3(f'{genre_path}/{track_name}.mp3')
                sound.export(f'{genre_path}/{track_name}.wav', format="wav")
                os.remove(f'{genre_path}/{track_name}.mp3')

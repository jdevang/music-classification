from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import requests
import os
from config import CLIENT_ID, CLIENT_SECRET

# set up with client id and secret 
client_credentials_manager = SpotifyClientCredentials(CLIENT_ID, CLIENT_SECRET)

# Make spotipy object
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# get list of genres
genres = sp.recommendation_genre_seeds()['genres']

# get 100 recommendation song in every genre
# Note: won't give 100 recommendations since some songs don't have preview urls. It is advised to run this a few times so that new suggestions get added
# Note 2: There are many genres in the list by spotify, it would be a better idea to choose a few genres which are popular and varying and work on those rather than risk underfitting the data
os.mkdir('data')
for genre in genres:
    print(genre)
    os.mkdir('data/' + genre)
    rec = sp.recommendations(seed_genres=[genre], limit = 100)
    count = 0
    for track in rec['tracks']:
        if track['preview_url']:
            print('\t', track['name'], ' ', track['preview_url'])
            r = requests.get(track['preview_url'])
            with open('data/' + genre + '/' + track['name'] + ".wav",'wb') as f:
                f.write(r.content)

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="3b25e7d7163c4fb48dadd9756a11f774", client_secret="1aed31509ce54261af24a8c4bcd89506"))

playlist_id = '6UeSakyzhiEt4NB3UAd6NQ' #top 100 today


playlist_id = input("enter playlist id: \n")

results = sp.playlist(playlist_id, fields='name, tracks.items(track(name,id,popularity))', market='US', additional_types=('track',))
print(results['name'])

items = results['tracks']['items']
ids = []
for i in items:
  ids.append(i['track']['id'])

features = sp.audio_features(ids)

featurenames = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
csv = "danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, popularity" + "\n"
for i in range(len(features)):
  for f in featurenames:
    csv += str(features[i][f]) + ","
  csv += str(items[i]['track']['popularity'])
  csv += "\n"

f = open(results['name'] + ".csv", "w")
f.write(csv)
f.close()

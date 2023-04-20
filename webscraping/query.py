import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="3b25e7d7163c4fb48dadd9756a11f774", client_secret="1aed31509ce54261af24a8c4bcd89506"))

playlist_id = '6UeSakyzhiEt4NB3UAd6NQ'

playlist_id = input("enter playlist id: \n")

results = sp.playlist(playlist_id, fields='tracks.items(track(name,id,popularity))', market='US', additional_types=('track',))

items = results['tracks']['items']
ids = []
for i in items:
  ids.append(i['track']['id'])

#features = [{'danceability': 0.629, 'energy': 0.733, 'key': 7, 'loudness': -5.445, 'mode': 1, 'speechiness': 0.0419, 'acousticness': 0.0025, 'instrumentalness': 0, 'liveness': 0.357, 'valence': 0.362, 'tempo': 120.001, 'type': 'audio_features', 'id': '3Ua0m0YmEjrMi9XErKcNiR', 'uri': 'spotify:track:3Ua0m0YmEjrMi9XErKcNiR', 'track_href': 'https://api.spotify.com/v1/tracks/3Ua0m0YmEjrMi9XErKcNiR', 'analysis_url': 'https://api.spotify.com/v1/audio-analysis/3Ua0m0YmEjrMi9XErKcNiR', 'duration_ms': 212241, 'time_signature': 4}, {'danceability': 0.707, 'energy': 0.681, 'key': 0, 'loudness': -4.325, 'mode': 1, 'speechiness': 0.0668, 'acousticness': 0.0632, 'instrumentalness': 5.15e-06, 'liveness': 0.0322, 'valence': 0.646, 'tempo': 117.999, 'type': 'audio_features', 'id': '0yLdNVWF3Srea0uzk55zFn', 'uri': 'spotify:track:0yLdNVWF3Srea0uzk55zFn', 'track_href': 'https://api.spotify.com/v1/tracks/0yLdNVWF3Srea0uzk55zFn', 'analysis_url': 'https://api.spotify.com/v1/audio-analysis/0yLdNVWF3Srea0uzk55zFn', 'duration_ms': 200455, 'time_signature': 4}, {'danceability': 0.492, 'energy': 0.675, 'key': 6, 'loudness': -5.456, 'mode': 1, 'speechiness': 0.0389, 'acousticness': 0.467, 'instrumentalness': 0, 'liveness': 0.142, 'valence': 0.478, 'tempo': 203.759, 'type': 'audio_features', 'id': '7K3BhSpAxZBznislvUMVtn', 'uri': 'spotify:track:7K3BhSpAxZBznislvUMVtn', 'track_href': 'https://api.spotify.com/v1/tracks/7K3BhSpAxZBznislvUMVtn', 'analysis_url': 'https://api.spotify.com/v1/audio-analysis/7K3BhSpAxZBznislvUMVtn', 'duration_ms': 163855, 'time_signature': 4}]
features = sp.audio_features(ids)

featurenames = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
csv = "danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo" + "\n"
for i in range(len(features)):
  for f in featurenames:
    csv += str(features[i][f]) + ","
  csv = csv[:-2]
  csv += "\n"

f = open("playlist.csv", "w")
f.write(csv)
f.close()

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="3b25e7d7163c4fb48dadd9756a11f774", client_secret="1aed31509ce54261af24a8c4bcd89506"))

playlist_id = '6UeSakyzhiEt4NB3UAd6NQ'
results = sp.playlist(playlist_id, fields='tracks.items(track(name,id,popularity))', market='US', additional_types=('track',))
#print(results)

items = results['tracks']['items']
ids = []
for i in items:
  ids.append(i['track']['id'])
print(ids)

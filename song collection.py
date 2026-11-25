import requests
from pathlib import Path

import json

with open("random50.json", encoding="utf-16") as f:
    songs = json.load(f)

print(len(songs))  # 50
import pandas as pd

df = pd.DataFrame(songs)[["animeENName","animeJPName", "songName", "songArtist", "songType", "songDifficulty", "audio"]]
for i in range(len(df)):
    invalid_chars = ["<", ">", ":", "\"", "/", "\\", "|", "?", "*"]
    url = f"https://naedist.animemusicquiz.com/{df["audio"][i]}"
    if not any(c in invalid_chars for c in df["animeJPName"][i]): #if the JP Name doesnt have an invalid char use it
        filename = f"C:/Users/karee/PycharmProjects/AMQFly/training clips/{df["animeJPName"][i]}/{df["audio"][i]}"
        Path(f"C:/Users/karee/PycharmProjects/AMQFly/training clips/{df["animeJPName"][i]}").mkdir(parents=True,
                                                                                                   exist_ok=True)
    elif not any(c in invalid_chars for c in df["animeENName"][i]): #elif the EN Name doesnt have an invalid char use it
        filename = f"C:/Users/karee/PycharmProjects/AMQFly/training clips/{df["animeENName"][i]}/{df["audio"][i]}"
        Path(f"C:/Users/karee/PycharmProjects/AMQFly/training clips/{df["animeENName"][i]}").mkdir(parents=True,
                                                                                                   exist_ok=True)
    else: #if either name not valid then dont use for now
        continue

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Saved {filename}")
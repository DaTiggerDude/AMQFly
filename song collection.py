import json

with open("random50.json", encoding="utf-8") as f:
    songs = json.load(f)

print(len(songs))  # 50
import pandas as pd

df = pd.DataFrame(songs)[["animeENName","animeJPName", "songName", "songArtist", "songType", "songDifficulty", "audio"]]

import json
import os
import re
import requests

BASE_URL = "https://naedist.animemusicquiz.com/"
OUT_DIR = "C:/Users/karee/PycharmProjects/AMQFly/training clips"
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

def safe_name(name: str) -> str:
    name = re.sub(r'[<>:"/\\|?*]', "_", name)   # illegal on Windows
    return name.strip().rstrip(".")              # no trailing dot/space

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_manifest(manifest: dict) -> None:
    tmp = MANIFEST_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, MANIFEST_PATH)               # atomic swap

os.makedirs(OUT_DIR, exist_ok=True)
manifest = load_manifest()
session = requests.Session()

for _, row in df.iterrows():
    audio = row["audio"]
    folder_name = safe_name(row["animeENName"])
    folder = os.path.join(OUT_DIR, folder_name)
    os.makedirs(folder, exist_ok=True)

    filename = os.path.join(folder, audio)
    tmp_file = filename + ".part"

    # Already downloaded: make sure it's in the manifest, then skip
    if os.path.exists(filename):
        if audio not in manifest:
            manifest[audio] = {
                "folder": folder_name,
                "animeENName": row["animeENName"],
                "animeJPName": row["animeJPName"],
                "songName": row["songName"],
                "songArtist": row["songArtist"],
            }
            save_manifest(manifest)
        continue

    try:
        response = session.get(BASE_URL + audio, stream=True, timeout=30)
        response.raise_for_status()
        with open(tmp_file, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        os.replace(tmp_file, filename)           # only appears once complete
    except requests.RequestException as e:
        print(f"Failed {audio}: {e}")
        if os.path.exists(tmp_file):
            os.remove(tmp_file)
        continue

    # Download succeeded: record it
    manifest[audio] = {
        "folder": folder_name,
        "animeENName": row["animeENName"],
        "animeJPName": row["animeJPName"],
        "songName": row["songName"],
        "songArtist": row["songArtist"],
    }
    save_manifest(manifest)
    print(f"Saved {filename}")
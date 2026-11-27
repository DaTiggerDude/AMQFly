import json
import requests

URL = "https://anisongdb.com/api/get_n_random_songs"
OUTPUT_FILE = "random50.json"


def main():
    try:
        response = requests.post(
            URL,
            json={"n": 100},
            headers={"accept": "*/*"},
            timeout=15,
        )
        response.raise_for_status()

        songs = response.json()

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(songs, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved {len(songs)} songs to {OUTPUT_FILE}")

    except requests.exceptions.RequestException as e:
        print(f"Network, API, or JSON error: {e}")


if __name__ == "__main__":
    main()
import json
import requests

URL = "https://anisongdb.com/api/get_50_random_songs"
OUTPUT_FILE = "random50.json"


def main():
    try:
        response = requests.post(URL, timeout=15)
        response.raise_for_status()  # Raises an error for 4xx/5xx status codes

        songs = response.json()

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(songs, f, indent=2, ensure_ascii=False)

        print(f"Successfully saved {len(songs)} songs to {OUTPUT_FILE}")

    except requests.exceptions.RequestException as e:
        print(f"Network or API error: {e}")
    except json.JSONDecodeError:
        print("Error: Response was not valid JSON.")


if __name__ == "__main__":
    main()
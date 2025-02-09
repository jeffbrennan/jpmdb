import time
from pathlib import Path

import polars as pl
import requests
from bs4 import BeautifulSoup


def dump_imdb_photo(photo_url: str, tconst: str) -> None:
    path = (
        Path(__file__).parents[1]
        / "data"
        / "bronze"
        / "imdb"
        / "photos"
        / f"{tconst}.jpg"
    )

    path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(photo_url, headers={"User-Agent": "Mozilla/5.0"})
    with open(path, "wb") as f:
        f.write(response.content)


def get_imdb_photo_url(tconst: str, photo_id: str) -> str | None:
    source_url = f"https://www.imdb.com/title/{tconst}/mediaviewer/{photo_id}/"
    response = requests.get(source_url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.find_all("img")
    urls = []
    for result in results:
        if "data-image-id" in result.attrs and result["data-image-id"].endswith("curr"):
            urls.append(result["src"])
            break

    if len(urls) == 0:
        return None

    return urls[0]


def get_imdb_photo_id(tconst: str) -> str | None:
    url = f"https://www.imdb.com/title/{tconst}/"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    results = soup.find_all("a", class_="ipc-lockup-overlay ipc-focusable")
    photo_ids = []
    for result in results:
        try:
            photo_title = result["aria-label"]
            if "poster" not in photo_title.lower():
                continue

            photo_id = result["href"].split("/mediaviewer/")[-1].split("/")[0]
            photo_ids.append(photo_id)
            break

        except Exception as e:
            print(e)
            continue

    if len(photo_ids) != 1:
        return None

    return photo_ids[0]


def scrape_imdb_photo(
    i: int, n_records: int, record: dict, existing_records: set[str]
) -> None:
    print(f"[{i:04d}/{n_records:04d}] {record['primaryTitle']}")
    if record["tconst"] is None:
        return

    if record["tconst"] in existing_records:
        print("photo already exists. skipping")
        print("-" * 40)
        return
    print("scraping photo for", record["primaryTitle"])
    time.sleep(1)
    photo_id = get_imdb_photo_id(record["tconst"])
    if photo_id is None:
        return

    print("found photo id", photo_id)
    photo_url = get_imdb_photo_url(record["tconst"], photo_id)
    if photo_url is None:
        return

    print("found photo url", photo_url)
    dump_imdb_photo(photo_url, record["tconst"])
    print("-" * 40)


def main():
    base_df_path = Path(__file__).parents[1] / "data" / "gold" / "jpmdb"
    base_df = pl.read_delta(str(base_df_path))
    records = base_df.select("tconst", "primaryTitle").to_dicts()
    existing_records = set(
        [
            path.stem
            for path in Path(__file__).parents[1].glob("data/bronze/imdb/photos/*.jpg")
        ]
    )

    for i, record in enumerate(records, 1):
        try:
            scrape_imdb_photo(i, len(records), record, existing_records)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    main()

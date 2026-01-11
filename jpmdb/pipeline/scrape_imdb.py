import datetime
import random
import time

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel


class IMDBScrapingResult(BaseModel):
    title: str
    url: str
    unique_ids: list[str]
    retrieved_at: datetime.datetime


def get_imdb_unique_ids_from_title(title: str, exact: bool) -> IMDBScrapingResult:
    url = f"https://www.imdb.com/find/?q={title.replace(' ', '%20')}"
    if exact:
        url += "&exact=true"

    print(url)
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")
    results = soup.find_all("a", class_="ipc-title-link-wrapper")
    unique_ids = []
    for result in results:
        try:
            tconst = result["href"].split("/title/")[-1].split("/?")[0]
        except Exception as e:
            print(e)
            continue

        unique_ids.append(tconst)

    return IMDBScrapingResult(
        title=title,
        url=url,
        unique_ids=unique_ids,
        retrieved_at=datetime.datetime.now(),
    )


def main():
    titles_to_scrape = ["sponge bob squarepants sponge out of water"]
    all_results = []
    for title in titles_to_scrape:
        time.sleep(random.randint(300, 1000) / 1000)
        result = get_imdb_unique_ids_from_title(title, exact=False)

        print(result)
        all_results.append(result)

    print(all_results)


if __name__ == "__main__":
    main()

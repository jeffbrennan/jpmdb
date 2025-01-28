import datetime
import re
from enum import StrEnum, auto
from pathlib import Path
from typing import Optional

import polars as pl
from pydantic import BaseModel, Field


class JPMDBCategory(StrEnum):
    tv = auto()
    short = auto()
    miniseries = auto()
    movie = auto()


class JPMDBRecord(BaseModel):
    watched_year: int = Field(le=2025, ge=2015)
    watched_order: int = Field(ge=1)
    title_original: str
    title: str
    title_year: Optional[int] = Field(default=None, le=2025, ge=1900)
    category: JPMDBCategory
    season: Optional[int] = Field(default=None, ge=1, le=100)
    sequel: Optional[int] = Field(default=None, ge=1, le=100)
    rating: float = Field(le=10, ge=0)


class ParsedLine(BaseModel):
    title_original: str
    title: str
    title_year: Optional[int] = Field(default=None, le=2025, ge=1900)
    category: JPMDBCategory
    season: Optional[int] = Field(default=None, ge=1, le=100)
    sequel: Optional[int] = Field(default=None, ge=1, le=100)
    rating: float = Field(le=10, ge=0)


def parse_sequel(line: str) -> Optional[int]:
    non_numeric_mapping = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5,
        "vi": 6,
        "vii": 7,
        "viii": 8,
        "ix": 9,
    }
    sequel = None
    sequel_identifiers = ["part", "volume", "series", "chapter"]
    if any(identifier in line for identifier in sequel_identifiers):
        sequel_match = re.search(r"part \d+|volume \d+|series \d+|chapter \d+", line)
        if sequel_match:
            sequel = int(sequel_match.group(0).split(" ")[-1])

    if sequel is None:
        remaining_sequel_match = re.search(r"\d+$", line)
        if remaining_sequel_match:
            sequel = int(remaining_sequel_match.group(0))

    if sequel is None:
        title_lower = line.lower()
        for key, value in non_numeric_mapping.items():
            if re.search(rf"\b{key}\b", title_lower):
                sequel = value
                break

    if sequel is not None and sequel > 100:
        sequel = None

    return sequel


def parse_line(line: str) -> ParsedLine:
    line_clean = line.strip()
    line_lower = line_clean.lower()

    season = None
    if "short" in line_lower:
        category = JPMDBCategory.short
    elif "season" in line_lower or "episode" in line_lower:
        category = JPMDBCategory.tv
        season_match = re.search(r"season \d+", line_lower)
        if season_match:
            season = int(season_match.group(0).split(" ")[-1])
    elif "miniseries" in line_lower or "series" in line_lower:
        category = JPMDBCategory.miniseries
    else:
        category = JPMDBCategory.movie

    rating = float(line_clean.split("/10")[0].split(" ")[-1].removeprefix("(").strip())

    title_year_check = re.search(r"\d{4}]|: \d{4}", line_clean)
    if title_year_check:
        title_year = int(
            title_year_check.group(0)
            .removeprefix("[")
            .removesuffix("]")
            .removeprefix(":")
            .strip()
        )
        if title_year > datetime.datetime.now().year:
            title_year = None
    else:
        title_year = None

    sequel = parse_sequel(line_lower)
    title = (
        line_clean.split(" (")[0]
        .replace("Mini Series", "Miniseries")
        .replace("Complete Miniseries", "")
        .replace("Miniseries", "")
        .split("Season")[0]
        .split("season")[0]
        .split("Series")[0]
        .split("series")[0]
        .split("[")[0]
        .strip()
        .removesuffix(":")
        .removesuffix(".")
    )
    if "Series:" in line:
        title = title + " Series"

    remaining_series_identifiers = ["episode", "Episode"]
    for identifier in remaining_series_identifiers:
        identifier_found = re.search(rf"{identifier} \d+|: \d+ {identifier}", title)
        if identifier_found:
            title = title.split(identifier_found.group(0))[0].strip()

    if sequel is None:
        sequel = parse_sequel(title)

    if sequel is not None and season is not None and "part" not in line_lower:
        sequel = None

    return ParsedLine(
        title=title,
        title_original=line_clean,
        title_year=title_year,
        category=category,
        season=season,
        sequel=sequel,
        rating=rating,
    )


def parse_jpmdb():
    in_file_dir = Path(__file__).parents[1] / "data" / "bronze" / "jp_movies"
    latest_file = max(in_file_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime)

    watched_year = 0
    prev_watched_year = 0
    section_start_index = 2
    all_records: list[JPMDBRecord] = []
    with latest_file.open("r") as f:
        raw_text = f.readlines()

    for i, line in enumerate(raw_text):
        print(i)
        if line.strip() == "":
            continue
        section_contains_year = re.search(r"\d{4}", line)
        prev_line_blank = i > 0 and raw_text[i - 1].strip() == ""
        next_line_blank = i < len(raw_text) - 1 and raw_text[i + 1].strip() == ""

        section_start = (
            section_contains_year and i == 0 or (prev_line_blank and next_line_blank)
        )

        if section_start and section_contains_year is not None:
            watched_year = int(section_contains_year.group(0))
            if watched_year > prev_watched_year:
                prev_watched_year = watched_year
                section_start_index = i + 2
            continue

        watched_order = i - section_start_index + 1
        parsed_line = parse_line(line)
        all_records.append(
            JPMDBRecord(
                watched_order=watched_order,
                watched_year=watched_year,
                title_original=parsed_line.title_original,
                title=parsed_line.title,
                title_year=parsed_line.title_year,
                category=parsed_line.category,
                season=parsed_line.season,
                sequel=parsed_line.sequel,
                rating=parsed_line.rating,
            )
        )

    return all_records


def validate_jpmdb(records: list[JPMDBRecord]) -> pl.DataFrame:
    df = (
        pl.DataFrame(records)
        .with_columns(
            pl.first()
            .cum_count()
            .alias("watched_order_calc")
            .over(partition_by="watched_year", order_by="watched_order")
        )
        .drop("watched_order")
        .rename({"watched_order_calc": "watched_order"})
        .select(
            "watched_year",
            "watched_order",
            "title_original",
            "title",
            "title_year",
            "category",
            "season",
            "sequel",
            "rating",
        )
    )

    check_min_watch_order = (
        df.group_by("watched_year")
        .agg(pl.col("watched_order").min().alias("min_order"))
        .filter(pl.col("min_order") != 1)
    )
    if check_min_watch_order.height > 0:
        raise ValueError("Minimum watched order is not 1 for some years")

    check_rating = df.filter((pl.col("rating") < 0) | (pl.col("rating") > 10))
    if check_rating.height > 0:
        raise ValueError("Rating is not between 0 and 10 for some records")

    check_no_year_gaps = (
        df.select(pl.col("watched_year").unique())
        .sort("watched_year")
        .with_columns(pl.col("watched_year").diff().alias("year_diff"))
        .filter(pl.col("year_diff") != 1)
    )
    if check_no_year_gaps.height > 0:
        print(check_no_year_gaps.head())
        raise ValueError("There are gaps in the watched years")

    check_no_titles_end_with_special_char = df.filter(
        pl.col("title").str.ends_with(":")
    )
    if check_no_titles_end_with_special_char.height > 0:
        print(check_no_titles_end_with_special_char.head())
        raise ValueError("There are titles that end with a special character")

    check_no_dupes = (
        df.group_by("watched_year", "title_year", "title", "season", "sequel")
        .agg(pl.col("rating").count().alias("count"))
        .filter(pl.col("count") > 1)
    )
    if check_no_dupes.height > 0:
        print(check_no_dupes.head())
        raise ValueError("There are duplicate titles")

    check_no_series_identifiers_in_title = df.filter(
        pl.col("title").str.contains(" (?i)season ")
        | pl.col("title").str.contains(" (?i)episode ")
        | pl.col("title").str.contains(" (?i)series ")
    )

    if check_no_series_identifiers_in_title.height > 0:
        print(check_no_series_identifiers_in_title.head())
        raise ValueError("There are series identifiers in the title")

    return df


def write_jpmdb(df: pl.DataFrame) -> None:
    out_file_dir = Path(__file__).parents[1] / "data" / "silver"
    out_file_dir.mkdir(parents=True, exist_ok=True)
    df.write_delta(
        out_file_dir / "jpmdb",
        mode="overwrite",
        delta_write_options={"schema_mode": "overwrite"},
    )


if __name__ == "__main__":
    records = parse_jpmdb()
    df = validate_jpmdb(records)
    write_jpmdb(df)

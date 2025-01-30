from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import typer
from pydantic import BaseModel

from jpmdb.scrape_imdb import get_imdb_unique_ids_from_title

app = typer.Typer()


class MatchConfig(BaseModel):
    record: StagingRecord
    imdb_data: pl.DataFrame
    scrape: bool
    prompt_tconst: bool


class StagingRecord(BaseModel):
    watched_year: int
    watched_order: int
    title_original: str
    title: str
    title_year: int | None
    category: str
    season: int | None
    sequel: int | None
    rating: float
    title_normalized: str
    watched_id: str
    tconst: str | None
    last_updated: datetime.datetime
    manually_reviewed_at: datetime.datetime | None
    manually_approved: bool | None


class RecordToReview(BaseModel):
    title_original: str
    title: str
    title_year: int | None
    category: str
    season: int | None
    tconst: str | None
    primaryTitle: str | None
    originalTitle: str | None
    numVotes: int | None

class IMDBRecordData(BaseModel):
    tconst: str
    primaryTitle: str
    originalTitle: str
    startYear: int
    endYear: int | None
    runtimeMinutes: int | None
    genres: str
    averageRating: float | None
    numVotes: int | None


def print_matched_record(record: RecordToReview) -> None:
    if record.title_year is None:
        title_suffix = ""
    else:
        title_suffix = f" ({record.title_year})"

    print("=" * 40)
    print(f"Title: {record.title} {title_suffix}")
    print(f"Original title: {record.title_original}")
    print(f"Category: {record.category}")

    if record.season is not None:
        print(f"Season: {record.season}")

    print("-" * 40)
    print(f"tconst: {record.tconst}")
    print(f"Primary title: {record.primaryTitle}")
    print(f"Original title: {record.originalTitle}")
    print(f"Number of votes: {record.numVotes}")
    print("=" * 40)


def review_matched_record(record: RecordToReview) -> bool:
    print_matched_record(record)
    return typer.confirm("Do you approve this record?")


def scrape_record(config: MatchConfig, unique_id: str | None) -> None:
    record = config.record

    if unique_id is None:
        search_query = record.title
        if record.title_year is not None:
            search_query += f" {record.title_year}"
        try:
            unique_ids = get_imdb_unique_ids_from_title(record.title).unique_ids
        except Exception as e:
            print(f"error scraping record: {record.title}")
            print(e)
            return

    else:
        unique_ids = [unique_id]

    imdb_data_filtered = config.imdb_data.filter(
        pl.col("tconst").is_in(unique_ids)
    ).to_dicts()

    imdb_data_filtered = [IMDBRecordData(**record) for record in imdb_data_filtered]

    for imdb_record in imdb_data_filtered:
        model = RecordToReview(
            title_original=record.title_original,
            title=record.title,
            title_year=record.title_year,
            category=record.category,
            season=record.season,
            tconst=imdb_record.tconst,
            primaryTitle=imdb_record.primaryTitle,
            originalTitle=imdb_record.originalTitle,
            numVotes=imdb_record.numVotes,
        )
        approved = review_matched_record(model)
        if approved:
            log_reviewed_record(record, True)
            return

    log_reviewed_record(record, False)


def log_reviewed_record(record: StagingRecord, approved: bool) -> None:
    out_dir = Path(__file__).parents[1] / "data" / "silver" / "jpmdb"
    out_path = out_dir / "stg_jpmdb_combined"
    record.manually_approved = approved
    record.manually_reviewed_at = datetime.datetime.now(datetime.timezone.utc)

    if not approved:
        record.tconst = None

    df = pl.DataFrame(record.model_dump())
    df.write_delta(
        out_path,
        mode="merge",
        delta_merge_options={
            "predicate": "s.watched_id = t.watched_id",
            "source_alias": "s",
            "target_alias": "t",
        },
    ).when_matched_update_all().execute()


def handle_unmatched_record(config: MatchConfig) -> None:
    if not config.scrape and not config.prompt_tconst:
        print("skipping record without tconst")
        return

    tconst = None
    if config.prompt_tconst:
        tconst = typer.prompt("enter tconst:")

    if config.scrape or config.prompt_tconst:
        scrape_record(config, tconst)


def handle_matched_record(config: MatchConfig) -> None:
    approved = review_matched_record(RecordToReview(**config.record.model_dump()))
    if approved:
        log_reviewed_record(config.record, True)
        return

    handle_unmatched_record(config)


def main(scrape: bool = True, prompt_tconst: bool = False) -> None:
    base_dir = Path(__file__).parents[1] / "data" / "silver"
    base_path = base_dir / "jpmdb" / "stg_jpmdb_combined"
    staging_df = pl.read_delta(str(base_path))

    imdb_data = pl.read_delta(str(base_dir / "imdb" / "title_basics")).join(
        pl.read_delta(str(base_dir / "imdb" / "title_ratings")), on="tconst"
    )

    records_to_review = (
        staging_df.filter(
            (pl.col("manually_approved").is_null()) | (~pl.col("manually_approved"))
        )
        .join(imdb_data, on="tconst", how="left")
        .to_dicts()
    )
    records_to_review = [StagingRecord(**record) for record in records_to_review]
    print("Number of records to review:", len(records_to_review))

    for record in records_to_review:
        config = MatchConfig(
            record=record,
            imdb_data=imdb_data,
            scrape=scrape,
            prompt_tconst=prompt_tconst,
        )

        if record.tconst is None:
            handle_unmatched_record(config)
        else:
            handle_matched_record(config)


if __name__ == "__main__":
    typer.run(main)

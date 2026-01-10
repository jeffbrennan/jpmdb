from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Annotated

import polars as pl
import typer
from pydantic import BaseModel

from jpmdb.common import timeit
from jpmdb.scrape_imdb import get_imdb_unique_ids_from_title

app = typer.Typer(pretty_exceptions_enable=False)


@dataclass
class MatchConfig:
    review_record: RecordToReview
    record: StagingRecord
    imdb_data: pl.DataFrame
    scrape: bool
    prompt_tconst: bool
    auto_approve: bool


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
    primaryTitle: str | None
    originalTitle: str | None
    titleType: str | None
    startYear: int | None
    numVotes: int | None


class IMDBRecordData(BaseModel):
    tconst: str
    primaryTitle: str
    originalTitle: str
    titleType: str | None
    startYear: int | None
    numVotes: int | None


def print_jpmdb_record(
    watched_id: str,
    title: str,
    title_year: int | None,
    category: str,
    season: int | None,
    **kwargs,
) -> None:
    if title_year is None:
        title_suffix = ""
    else:
        title_suffix = f"({title_year})"

    header_len = 40
    header_side = int(((header_len - 6) - len(watched_id)) / 2)

    print("=" * header_side, {watched_id}, "=" * header_side)
    print(f"Title: {title} {title_suffix}")
    print(f"Category: {category}")

    if season is not None:
        print(f"Season: {season}")

    print("=" * 40)


def print_matched_record(record: RecordToReview) -> None:
    print_jpmdb_record(**record.model_dump())

    print(f"tconst: {record.tconst}")
    print(f"Primary title: {record.primaryTitle}")
    print(f"Original title: {record.originalTitle}")
    print(f"Title type: {record.titleType}")
    if record.startYear is not None:
        print(f"Start year: {record.startYear}")
    print(f"Number of votes: {record.numVotes:,}")
    print("=" * 40)
    print("")


def clean_string(s: str) -> str:
    s = s.lower().replace(" ", "").replace("&", "and").replace("the", "")
    s = re.sub('[^0-9a-zA-Z]+', '', s)
    return s


def review_matched_record(record: RecordToReview, auto_approve: bool) -> bool:
    if auto_approve:
        incoming_title = clean_string(record.title)

        imdb_title_primary = (record.primaryTitle or record.originalTitle)
        if imdb_title_primary is not None:
            imdb_title_primary = clean_string(imdb_title_primary)

        imdb_title_original = (record.originalTitle or record.primaryTitle)
        if imdb_title_original is not None:
            imdb_title_original = clean_string(imdb_title_original)

        title_pass =  incoming_title == imdb_title_primary or incoming_title == imdb_title_original
        year_pass =  record.startYear is not None and record.watched_year >= record.startYear

        print('-' * 40)
        print(f"auto-matching {record.title}")
        if (title_pass and year_pass):
            print(f"matched to {record.tconst}")
            print('-' * 40)
            return True
        else:
            print(f"{title_pass=} {year_pass=}")

    print_matched_record(record)
    return typer.confirm("Do you approve this record?")


def scrape_record(config: MatchConfig, unique_id: str | None, exact: bool) -> bool:
    record = config.record
    approved = False
    if unique_id is None:
        search_query = record.title
        if record.title_year is not None:
            search_query += f" {record.title_year}"
        try:
            print("scraping record:", record.title)
            unique_ids = get_imdb_unique_ids_from_title(record.title, exact=exact).unique_ids
        except Exception as e:
            print(f"error scraping record: {record.title}")
            print(e)
            return approved

    else:
        unique_ids = [unique_id]

    if len(unique_ids) == 0:
        print('no matches found!')
        return approved

    imdb_data_filtered = (
        config.imdb_data.filter(pl.col("tconst").is_in(unique_ids))
        .with_columns(
            pl.when(pl.col("numVotes").is_null())
            .then(0)
            .otherwise(pl.col("numVotes"))
            .alias("numVotes")
        )
        .to_dicts()
    )

    imdb_data_filtered = [IMDBRecordData(**record) for record in imdb_data_filtered]

    imdb_data_filtered = sorted(  # pyright: ignore [reportCallIssue]
        imdb_data_filtered,
        key=lambda x: x.numVotes,  # pyright: ignore [reportArgumentType]
        reverse=True,
    )

    for imdb_record in imdb_data_filtered:
        review_record = RecordToReview(
            **config.record.model_dump(),
            primaryTitle=imdb_record.primaryTitle,
            originalTitle=imdb_record.originalTitle,
            numVotes=imdb_record.numVotes,
            titleType=imdb_record.titleType,
            startYear=imdb_record.startYear,
        )

        approved = review_matched_record(review_record, config.auto_approve)
        if approved:
            record.tconst = imdb_record.tconst
            log_reviewed_record(record, True)
            return approved

    log_reviewed_record(record, False)
    return approved


def log_reviewed_record(record: StagingRecord, approved: bool) -> None:
    out_dir = Path(__file__).parents[1] / "data" / "silver" / "jpmdb"
    out_path = out_dir / "stg_jpmdb_combined"
    record.manually_approved = approved
    record.manually_reviewed_at = datetime.datetime.now(datetime.timezone.utc)

    if not approved:
        record.tconst = None

    df = pl.DataFrame(record.model_dump())
    df_schema = pl.read_delta(str(out_path)).schema
    df = df.cast(df_schema)  # pyright: ignore [reportArgumentType]

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
        print_jpmdb_record(**config.record.model_dump())
        tconst = typer.prompt("enter tconst")

    if config.scrape or config.prompt_tconst:
        matched = scrape_record(config, tconst, exact=True)
        if not matched:
            scrape_record(config, tconst, exact=False)


def handle_matched_record(config: MatchConfig) -> None:
    approved = review_matched_record(config.review_record, config.auto_approve)
    if approved:
        log_reviewed_record(config.record, True)
        return

    handle_unmatched_record(config)


@app.command()
def delete(unique_id: Annotated[str, typer.Option(prompt=True)]) -> None:
    base_dir = Path(__file__).parents[1] / "data" / "silver"
    base_path = base_dir / "jpmdb" / "stg_jpmdb_combined"
    staging_df = pl.read_delta(str(base_path))

    unique_ids = unique_id.split(",")

    records_to_delete = staging_df.filter(
        pl.col("watched_id").is_in(unique_ids)
    ).to_dicts()

    if len(records_to_delete) == 0:
        print("No records found to delete")
        return

    for record in records_to_delete:
        record = StagingRecord(**record)
        print(record)
        confirm_deletion = typer.confirm("Remove approval?")
        if not confirm_deletion:
            continue

        record.tconst = None
        log_reviewed_record(record, False)


@app.command()
def review(
    scrape: bool = True,
    prompt_tconst: bool = False,
    review_again: bool = False,
    watched_id: str | None = None,
    auto_approve: bool = False,
) -> None:
    base_dir = Path(__file__).parents[1] / "data" / "silver"
    base_path = base_dir / "jpmdb" / "stg_jpmdb_combined"
    staging_df = pl.read_delta(str(base_path))

    imdb_data = (
        pl.read_delta(str(base_dir / "imdb" / "title_basics"))
        .join(pl.read_delta(str(base_dir / "imdb" / "title_ratings")), on="tconst")
        .select(
            "tconst",
            "primaryTitle",
            "originalTitle",
            "numVotes",
            "titleType",
            "startYear",
        )
    )
    if watched_id is None:
        staging_df = staging_df.filter(
            (pl.col("manually_approved").is_null())
            | (~pl.col("manually_approved"))
            | (pl.col("tconst").is_null())
        )
    else:
        staging_df = staging_df.filter(pl.col("watched_id") == watched_id)

    if not review_again:
        staging_df = staging_df.filter(pl.col("manually_reviewed_at").is_null())

    records_to_review = staging_df.join(imdb_data, on="tconst", how="left").to_dicts()

    records_to_review = [RecordToReview(**record) for record in records_to_review]
    print("Number of records to review:", len(records_to_review))

    for record in records_to_review:
        config = MatchConfig(
            record=StagingRecord(**record.model_dump()),
            imdb_data=imdb_data,
            scrape=scrape,
            prompt_tconst=prompt_tconst,
            review_record=record,
            auto_approve=auto_approve,
        )

        if record.tconst is None:
            handle_unmatched_record(config)
        else:
            handle_matched_record(config)


if __name__ == "__main__":
    app()

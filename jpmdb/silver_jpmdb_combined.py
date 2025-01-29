from concurrent.futures.process import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

import jellyfish
import polars as pl
import unidecode
from rapidfuzz import fuzz

from jpmdb.common import timeit


def add_phonetic_similarity(
    df: pl.DataFrame,
    similarity_func: type(jellyfish.metaphone)  # pyright: ignore [reportInvalidTypeForm]
    | type(jellyfish.nysiis),  # pyright: ignore [reportInvalidTypeForm]
    similarity_name: str,
    col1: str,
    col2: str,
) -> pl.DataFrame:
    df_clean = df.with_columns(
        pl.col(col1)
        .map_elements(
            lambda t: similarity_func(t).replace(" ", ""),
            return_dtype=pl.String,
            returns_scalar=True,
        )
        .alias(f"{col1}_{similarity_name}")
    )

    df_clean = df_clean.with_columns(
        pl.col(col2)
        .map_elements(
            lambda t: similarity_func(t),
            return_dtype=pl.String,
            returns_scalar=True,
        )
        .alias(f"{col2}_{similarity_name}")
    )

    df_clean = df_clean.with_columns(
        pl.struct(
            f"{col1}_{similarity_name}",
            f"{col2}_{similarity_name}",
        ).alias(f"{similarity_name}_struct"),
    )

    df_clean = df_clean.with_columns(
        pl.col(f"{similarity_name}_struct")
        .map_elements(
            lambda t: fuzz.ratio(
                t[f"{col1}_{similarity_name}"],
                t[f"{col2}_{similarity_name}"],
            ),
            return_dtype=pl.Float32,
            returns_scalar=True,
        )
        .alias(f"{similarity_name}_similarity")
    )

    return df_clean


@timeit
def fuzzy_match_remaining(record: dict, imdb_filtered: pl.DataFrame) -> list[dict]:
    fuzzy_matches_base = (
        imdb_filtered.with_columns(
            pl.lit(record["title_normalized"]).alias("jpmdb_title_normalized")
        )
        .with_columns(
            pl.struct("title_normalized", "jpmdb_title_normalized").alias(
                "title_struct"
            ),
        )
        .with_columns(
            pl.col("title_struct")
            .map_elements(
                lambda t: fuzz.ratio(
                    t["title_normalized"],
                    t["jpmdb_title_normalized"],
                ),
                return_dtype=pl.Float32,
                returns_scalar=True,
            )
            .alias("similarity")
        )
    )

    fuzzy_matches_with_metaphone = add_phonetic_similarity(
        fuzzy_matches_base,
        jellyfish.metaphone,
        "metaphone",
        "title_normalized",
        "jpmdb_title_normalized",
    )

    fuzzy_matches_with_nysiis = add_phonetic_similarity(
        fuzzy_matches_with_metaphone,
        jellyfish.nysiis,
        "nysiis",
        "title_normalized",
        "jpmdb_title_normalized",
    )

    weights = {
        "similarity": {"metaphone": 0.25, "nysiis": 0.25, "raw": 0.5},
        "overall": {"vote": 0.7, "similarity": 0.3},
    }

    max_matches = 10
    similarity_score_threshold = 85
    fuzzy_matches = (
        fuzzy_matches_with_nysiis.with_columns(
            (
                pl.col("metaphone_similarity").mul(weights["similarity"]["metaphone"])
                + pl.col("nysiis_similarity").mul(weights["similarity"]["nysiis"])
                + pl.col("similarity").mul(weights["similarity"]["raw"])
            ).alias("overall_similarity_score")
        )
        .filter(pl.col("overall_similarity_score").ge(similarity_score_threshold))
        .with_columns(
            (
                100
                * (pl.col("numVotes") - pl.col("numVotes").min())
                / (pl.col("numVotes").max() - pl.col("numVotes").min())
            ).alias("vote_score")
        )
        .with_columns(
            pl.when(pl.col("vote_score").is_nan())
            .then(100)
            .otherwise(pl.col("vote_score"))
            .alias("vote_score")
        )
        .with_columns(
            (
                pl.col("vote_score").mul(weights["overall"]["vote"])
                + pl.col("overall_similarity_score").mul(
                    weights["overall"]["similarity"]
                )
            ).alias("weighted_score")
        )
        .sort(pl.col("weighted_score"), descending=True)
        .limit(max_matches)
    )

    return fuzzy_matches.to_dicts()


@timeit
def handle_fuzzy_matching(
    unmatched_titles: pl.DataFrame,
    imdb_filtered: pl.DataFrame,
    jpmdb_silver: pl.DataFrame,
    sequential: bool = False,
) -> pl.DataFrame:
    unmatched_titles_records = unmatched_titles.to_dicts()

    if sequential:
        fuzzy_match_results_raw = [
            fuzzy_match_remaining(record, imdb_filtered)
            for record in unmatched_titles_records
        ]
    else:
        with ProcessPoolExecutor(max_workers=8) as executor:
            fuzzy_match_results_raw = list(
                executor.map(
                    fuzzy_match_remaining,
                    unmatched_titles_records,
                    repeat(imdb_filtered),
                )
            )

    fuzzy_match_results = [
        item for sublist in fuzzy_match_results_raw for item in sublist
    ]
    fuzzy_match_df = (
        jpmdb_silver.join(
            pl.DataFrame(fuzzy_match_results).rename(
                {
                    "title_normalized": "imdb_title_normalized",
                    "jpmdb_title_normalized": "title_normalized",
                }
            ),
            on="title_normalized",
            how="inner",
        )
        .filter(
            pl.col("title_year").is_null()
            | (pl.col("title_year") == pl.col("startYear"))
        )
        .with_columns(pl.col("weighted_score").mul(-1).alias("weighted_score_negative"))
        .with_columns(
            pl.arange(0, pl.len())
            .over(
                partition_by="watched_id",
                order_by=["weighted_score_negative"],
            )
            .alias("row_number")
        )
        .filter(pl.col("row_number") <= 5)
    )

    return fuzzy_match_df


@timeit
def prep_base_dataframes():
    base_path = Path(__file__).parents[1] / "data" / "silver"
    jpmdb_silver = (
        pl.read_delta(f"{base_path}/jpmdb")
        .with_columns(
            pl.col("title")
            .map_elements(
                unidecode.unidecode, return_dtype=pl.String, returns_scalar=True
            )
            .alias("title_normalized")
        )
        .with_columns(
            pl.col("title_normalized")
            .str.to_lowercase()
            .str.replace("\\: chapter ", " ")
            .str.replace_all("\\.|\\,|\\'|\\:|\\!|\\?|’|\\-| ", "")
            .str.replace_all(" and ", " & ")
            .str.replace(r"\bthe ", "")
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("watched_year"),
                    pl.lit("_"),
                    pl.col("watched_order").cast(pl.String).str.zfill(3),
                ]
            ).alias("watched_id")
        )
    )

    imdb_ratings = pl.read_delta(f"{base_path}/imdb/title_ratings")
    imdb_title_silver = (
        pl.read_delta(f"{base_path}/imdb/title_basics")
        .filter(
            pl.col("titleType").str.contains_any(
                ["movie", "tvSeries", "tvMovie", "tvMiniSeries", "tvSpecial"]
            )
        )
        .join(imdb_ratings, on="tconst", how="left")
        .filter(pl.col("numVotes") > 500)
        .with_columns(
            pl.col("primaryTitle")
            .map_elements(
                unidecode.unidecode, return_dtype=pl.String, returns_scalar=True
            )
            .alias("title_normalized")
        )
        .with_columns(
            pl.col("title_normalized")
            .str.to_lowercase()
            .str.replace("\\: chapter ", " ")
            .str.replace_all("\\.|\\,|\\'|\\:|\\!|\\?|’|\\-| ", "")
            .str.replace(r"\bthe ", "")
            .str.replace_all(" and ", " & ")
        )
    )

    return jpmdb_silver, imdb_title_silver


def get_exact_matches(jpmdb_silver, imdb_filtered) -> pl.DataFrame:
    exact_matches_raw = jpmdb_silver.join(
        imdb_filtered, on="title_normalized", how="inner"
    )

    exact_matches = (
        exact_matches_raw.with_columns(
            pl.col("numVotes").mul(-1).alias("numVotesNegative")
        )
        .with_columns(
            pl.col("watched_id").count().over("watched_id").alias("n_matches")
        )
        .with_columns(pl.col("n_matches").gt(1).alias("is_duplicate"))
        .with_columns(
            pl.arange(0, pl.len())
            .over(
                partition_by="watched_id",
                order_by=["numVotesNegative"],
            )
            .alias("match_order")
        )
        .filter(pl.col("match_order") == 0)
        .sort("watched_id")
    )
    return exact_matches


def main():
    jpmdb_silver, imdb_filtered = prep_base_dataframes()
    exact_matches = get_exact_matches(jpmdb_silver, imdb_filtered)

    unmatched_titles = jpmdb_silver.join(
        exact_matches.select("title_normalized").unique(),
        on="title_normalized",
        how="anti",
    )

    fuzzy_match_df = handle_fuzzy_matching(
        unmatched_titles,
        imdb_filtered,
        jpmdb_silver,
        sequential=False,
    )

    # TODO: ensure tv matched to tv, movie to movie
    # TODO: add imdb scraping for remaining unmatched titles
    # TODO: normalize roman numerals, numbers
    combined_df = pl.concat([fuzzy_match_df, exact_matches], how="diagonal_relaxed")

    unmatched_titles2 = (
        combined_df.select("title_normalized")
        .unique()
        .join(unmatched_titles, on="title_normalized", how="anti")
    )

    # remaining_round2


if __name__ == "__main__":
    main()

from jpmdb.parse_jpmdb import JPMDBCategory, ParsedLine, parse_line


def test_sequel_parsing():
    line = "Danger 5: Season 1 (8/10)"
    result = parse_line(line)

    expected = ParsedLine(
        title_original="Danger 5: Season 1 (8/10)",
        title="Danger 5",
        title_year=None,
        category=JPMDBCategory.tv,
        season=1,
        sequel=None,
        rating=8,
    )

    assert result == expected


def test_movie_with_sequel():
    result = parse_line("How To Train Your Dragon 2 (8.6/10)")

    expected = ParsedLine(
        title_original="How To Train Your Dragon 2 (8.6/10)",
        title="How To Train Your Dragon 2",
        title_year=None,
        category=JPMDBCategory.movie,
        season=None,
        sequel=2,
        rating=8.6,
    )

    assert result == expected


def test_tv_with_season_and_part():
    result = parse_line("The Way of the Househusband: Season 1 - Part 2 (8/10)")

    expected = ParsedLine(
        title_original="The Way of the Househusband: Season 1 - Part 2 (8/10)",
        title="The Way of the Househusband",
        title_year=None,
        category=JPMDBCategory.tv,
        season=1,
        sequel=2,
        rating=8,
    )

    assert result == expected


def test_tv_with_series_in_name():
    result = parse_line("X-Men - The Animated Series: season 1 (6.1/10)")

    expected = ParsedLine(
        title_original="X-Men - The Animated Series: season 1 (6.1/10)",
        title="X-Men - The Animated Series",
        title_year=None,
        category=JPMDBCategory.tv,
        season=1,
        sequel=None,
        rating=6.1,
    )

    assert result == expected


def test_miniseries():
    result = parse_line("Show Me A Hero: Complete Mini Series (9.7/10)")

    expected = ParsedLine(
        title_original="Show Me A Hero: Complete Mini Series (9.7/10)",
        title="Show Me A Hero",
        title_year=None,
        category=JPMDBCategory.miniseries,
        season=None,
        sequel=None,
        rating=9.7,
    )

    assert result == expected

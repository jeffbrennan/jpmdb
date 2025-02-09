from enum import StrEnum, auto


class SitePalette(StrEnum):
    PAGE_BACKGROUND_COLOR_LIGHT = "rgb(242, 240, 227)"
    BRAND_TEXT_COLOR_LIGHT = "rgb(33, 33, 33)"
    PAGE_BACKGROUND_COLOR_DARK = "rgb(33, 33, 33)"
    BRAND_TEXT_COLOR_DARK = "rgb(242, 240, 227)"


class ScreenWidth(StrEnum):
    xs = auto()
    sm = auto()
    md = auto()
    lg = auto()
    xl = auto()


def get_site_colors(dark_mode: bool, contrast: bool) -> tuple[str, str]:
    colors = {
        "dark": {
            "background": SitePalette.PAGE_BACKGROUND_COLOR_DARK,
            "text": SitePalette.BRAND_TEXT_COLOR_DARK,
        },
        "light": {
            "background": SitePalette.PAGE_BACKGROUND_COLOR_LIGHT,
            "text": SitePalette.BRAND_TEXT_COLOR_LIGHT,
        },
    }

    # invert the selector if contrast is True
    selector = not dark_mode if contrast else dark_mode
    selector_str = "dark" if selector else "light"

    return colors[selector_str]["background"], colors[selector_str]["text"]

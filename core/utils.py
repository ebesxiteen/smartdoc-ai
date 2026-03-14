def clean_spaces(text: str) -> str:
    """
    Normalizes whitespace: removes leading/trailing spaces and
    collapses multiple internal spaces into a single one.
    """
    if not text:
        return ""
    # .split() without arguments splits by any whitespace (space, \n, \t)
    # ' '.join(...) puts exactly one space between each word
    return " ".join(text.split())
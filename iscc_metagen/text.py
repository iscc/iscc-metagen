"""Functions for plain-text manipulation"""


def text_extract_parts(text, num_chars=1000):
    # type: (str, int) -> tuple[str, str, str]
    """Dynamically extract start, middle, and end chunks from text based on available context size"""
    middle = len(text) // 2
    s, m, e = text[:num_chars], text[middle : middle + num_chars], text[-num_chars:]
    return s, m, e

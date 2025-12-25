import re

_url = re.compile(r"https?://\S+|www\.\S+")
_non_word = re.compile(r"[^a-z0-9\s]")

def clean_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = _url.sub(" ", text)
    text = _non_word.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

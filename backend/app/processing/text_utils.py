import re
import spacy


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load(
                "en_core_web_md",
                disable=["tagger", "ner", "lemmatizer", "attribute_ruler"],
            )
        except Exception as exc:
            raise RuntimeError("Failed to load spaCy model 'en_core_web_md'") from exc
    return _NLP


def _pre_clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = re.sub(r"http\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"www\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^> *", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__|~~)(.*?)\1", r"\2", text)
    text = re.sub(r"(\*|_|~)(.*?)(\1|$)", r"\2", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r" +", " ", text)
    return text.strip()


_MARKDOWN_LINK_RE = re.compile(r"\[.+?\]\(https?://")


def _is_noisy_sentence(s: str) -> bool:
    """Return True if the sentence is noise that should be discarded."""
    words = s.split()
    if len(words) < 5:
        return True
    if len(s) > 600:
        return True
    digit_count = sum(1 for c in s if c.isdigit())
    if digit_count / len(s) > 0.20:
        return True
    if _MARKDOWN_LINK_RE.search(s):
        return True
    return False


def split_into_sentences(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    clean_text = _pre_clean_text(text)
    if not clean_text:
        return []

    nlp = _get_nlp()
    doc = nlp(clean_text)

    clean_sentences = []
    for sent in doc.sents:
        s_clean = sent.text.strip()
        if len(s_clean) < 15:
            continue
        if _is_noisy_sentence(s_clean):
            continue
        clean_sentences.append(s_clean)
    return clean_sentences


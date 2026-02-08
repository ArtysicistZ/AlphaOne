import re
import spacy


_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_md")
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
        clean_sentences.append(s_clean)
    return clean_sentences


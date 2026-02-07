import re
import spacy

# 1. Load the 'md' model (as before)
print("Loading spaCy model (en_core_web_md)...")
try:
    NLP = spacy.load("en_core_web_md")
except OSError:
    print("\n[Error] Medium spaCy model 'en_core_web_md' not found.")
    print("Please run: python -m spacy download en_core_web_md\n")
    exit()
print("spaCy model loaded successfully.")


def _pre_clean_text(text: str) -> str:
    """
    Aggressively removes junk, but *preserves* sentence boundaries
    (like . and \n) for spaCy to use.
    """
    if not isinstance(text, str):
        return ""
        
    # Remove URLs
    text = re.sub(r'http\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'www\S+', '', text, flags=re.IGNORECASE)
    
    # Remove Reddit markdown for quotes, but keep the text
    # Changes "> Hello" to "Hello"
    text = re.sub(r'^> *', '', text, flags=re.MULTILINE)
    
    # Remove markdown for bold/italics/strikethrough
    # Changes "**hello**" to "hello"
    text = re.sub(r'(\*\*|__|~~)(.*?)\1', r'\2', text) # Paired markdown
    text = re.sub(r'(\*|_|~)(.*?)(\1|$)', r'\2', text) # Single markdown

    # Remove inline code blocks
    text = re.sub(r'`[^`]+`', '', text)
    
    # Replace tabs with a space
    text = re.sub(r'\t+', ' ', text)
    
    # Squeeze multiple spaces into one (but *don't* touch newlines)
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    A smart sentence splitter that uses the spaCy NLP library
    on pre-cleaned text.
    """
    if not isinstance(text, str):
        return []
        
    # 1. Aggressively clean the text first
    clean_text = _pre_clean_text(text)
    
    if not clean_text:
        return []

    # 2. Process the *clean* text with spaCy.
    # spaCy will now correctly use the preserved . and \n to find boundaries.
    doc = NLP(clean_text)
    
    # 3. Iterate and filter
    clean_sentences = []
    for sent in doc.sents:
        # 'sent' is a spaCy 'Span' object. We get its text.
        s_clean = sent.text.strip()
        
        # Filter out junk sentences
        if len(s_clean) < 15: # Filter out short junk (like "1.57")
            continue
            
        clean_sentences.append(s_clean)
            
    return clean_sentences
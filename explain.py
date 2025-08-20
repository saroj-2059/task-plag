# explain.py
from typing import List, Tuple
from collections import Counter
from features import normalize_text, tokenize, ngram_shingles

def shared_ngrams(a: str, b: str, n: int = 3) -> List[Tuple[str,...]]:
    """Find shared n-grams between two texts."""
    ta, tb = tokenize(normalize_text(a)), tokenize(normalize_text(b))
    A, B = ngram_shingles(ta, n), ngram_shingles(tb, n)
    setA, setB = Counter(A), Counter(B)
    return [sh for sh in setA if sh in setB]

def highlight_with_ngrams(text: str, ngrams: List[Tuple[str,...]]) -> str:
    """Highlight shared n-grams in HTML using <mark>."""
    toks = tokenize(normalize_text(text))
    indices = set()
    for sh in ngrams:
        for i in range(len(toks) - len(sh) + 1):
            if tuple(toks[i:i+len(sh)]) == sh:
                indices.update(range(i, i+len(sh)))
    words = text.split()
    out = []
    wi = 0
    for w in words:
        if wi in indices:
            out.append(f"<mark>{w}</mark>")
        else:
            out.append(w)
        wi += 1
    return " ".join(out)

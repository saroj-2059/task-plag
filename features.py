# features.py
import re
import string
import ast
from typing import List, Tuple

# =====================
# Preprocessing & Tokenization
# =====================

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

def normalize_text(s: str) -> str:
    """Lowercase, remove punctuation, collapse spaces."""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    """Split into tokens using regex."""
    return _WORD_RE.findall(s)

def maybe_python_code(s: str) -> bool:
    """Heuristic: detect if paragraph might be Python code."""
    return ("def " in s) or ("class " in s) or ("import " in s)

def ast_tokens_from_python(s: str) -> List[str]:
    """Convert Python code into a sequence of AST node type names."""
    try:
        tree = ast.parse(s)
    except Exception:
        return []
    toks = []
    for node in ast.walk(tree):
        toks.append(type(node).__name__)
    return toks

def paragraphs(raw: str) -> List[str]:
    """Split raw text into paragraphs on blank lines."""
    parts = re.split(r"\n\s*\n", raw.strip())
    return [p.strip() for p in parts if p.strip()]

def ngram_shingles(tokens: List[str], n: int = 3) -> List[Tuple[str,...]]:
    """Generate n-gram shingles from a token list."""
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# =====================
# TF-IDF & Similarity
# =====================
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf(paragraph_texts: List[str]):
    """
    Build a TF-IDF vectorizer and fit it on the provided texts.
    Returns: (vectorizer, tfidf_matrix)
    """
    vec = TfidfVectorizer(ngram_range=(1, 3), min_df=1, norm="l2")
    X = vec.fit_transform(paragraph_texts)
    return vec, X

def tfidf_for(vec, texts: List[str]):
    """Transform new texts into TF-IDF vectors."""
    return vec.transform(texts)

def cosine_sim_matrix(A, B):
    """Compute cosine similarity between two TF-IDF matrices."""
    return cosine_similarity(A, B)

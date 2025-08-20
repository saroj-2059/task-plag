import os
import glob
import re
from html import escape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Optional: for PDF and DOCX
try:
    import docx
except ImportError:
    docx = None
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

lemmatizer = WordNetLemmatizer()

def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx" and docx:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".pdf" and fitz:
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    else:
        raise ValueError(f"Unsupported file type or missing library: {path}")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def similarity_score(text1, text2, ngram_range=(1,3)):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range)
    tfidf = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return score

def highlight_text(query, reference):
    query_words = set(query.lower().split())
    reference_words = reference.split()
    highlighted = []
    for word in reference_words:
        if word.lower() in query_words:
            highlighted.append(f"<mark>{escape(word)}</mark>")
        else:
            highlighted.append(escape(word))
    return " ".join(highlighted)

def detect_plagiarism(submissions, references, topk=5, min_score=0.0, ngram_range=(1,3)):
    results = {}
    for sub_name, sub_text in submissions.items():
        pre_sub = preprocess(sub_text)
        scores = []
        for ref_name, ref_text in references.items():
            pre_ref = preprocess(ref_text)
            score = similarity_score(pre_sub, pre_ref, ngram_range)
            if score >= min_score:
                scores.append((ref_name, score, sub_text, ref_text))
        scores.sort(key=lambda x: x[1], reverse=True)
        results[sub_name] = scores[:topk]
    return results

def detect_collusion(submissions, min_score=0.0, ngram_range=(1,3)):
    collusion = []
    sub_items = list(submissions.items())
    for i in range(len(sub_items)):
        for j in range(i+1, len(sub_items)):
            sub1_name, sub1_text = sub_items[i]
            sub2_name, sub2_text = sub_items[j]
            pre1 = preprocess(sub1_text)
            pre2 = preprocess(sub2_text)
            score = similarity_score(pre1, pre2, ngram_range)
            if score >= min_score:
                collusion.append((sub1_name, sub2_name, score))
    return collusion

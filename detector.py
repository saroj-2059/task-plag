import os
import ast
from docx import Document
import fitz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from html import escape

def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    text = ""
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    elif ext == ".docx":
        doc = Document(path)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif ext == ".pdf":
        pdf = fitz.open(path)
        text = "\n".join([page.get_text() for page in pdf])
    elif ext == ".py":
        text = extract_code_ast(path)
    return text

def extract_code_ast(path):
    with open(path, "r", encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except:
        return source
    tokens = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            tokens.append(f"func:{node.name}")
        elif isinstance(node, ast.ClassDef):
            tokens.append(f"class:{node.name}")
        elif isinstance(node, ast.Name):
            tokens.append(f"name:{node.id}")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                tokens.append(f"call:{node.func.id}")
    return " ".join(tokens)

def similarity_score(text1, text2, ngram_size=3):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, ngram_size))
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def minhash_signature(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in set(text.lower().split()):
        m.update(word.encode('utf8'))
    return m

def build_lsh_index(references, threshold=0.3, num_perm=128):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    signatures = {}
    for ref_name, ref_text in references.items():
        sig = minhash_signature(ref_text, num_perm)
        lsh.insert(ref_name, sig)
        signatures[ref_name] = sig
    return lsh, signatures

def lsh_similarity(sub_text, lsh, ref_signatures, num_perm=128):
    sig = minhash_signature(sub_text, num_perm)
    matches = lsh.query(sig)
    results = []
    for ref_name in matches:
        score = sig.jaccard(ref_signatures[ref_name])
        results.append((ref_name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def highlight_snippet(query, reference, snippet_len=50):
    query_words = query.lower().split()
    ref_words = reference.split()
    highlighted = []
    for word in ref_words:
        if word.lower() in query_words:
            highlighted.append(f"<mark>{escape(word)}</mark>")
        else:
            highlighted.append(escape(word))
    return " ".join(highlighted[:snippet_len])

def detect_plagiarism(submissions, references, topk=5, min_score=0.0, ngram_size=3, use_lsh=False):
    results = {}
    if use_lsh:
        lsh, ref_signatures = build_lsh_index(references, threshold=min_score)
    for sub_name, sub_text in submissions.items():
        scores = []
        if use_lsh:
            lsh_matches = lsh_similarity(sub_text, lsh, ref_signatures)
            for ref_name, score in lsh_matches:
                if score >= min_score:
                    snippet = highlight_snippet(sub_text, references[ref_name])
                    scores.append((ref_name, score, snippet))
        else:
            for ref_name, ref_text in references.items():
                score = similarity_score(sub_text, ref_text, ngram_size)
                if score >= min_score:
                    snippet = highlight_snippet(sub_text, ref_text)
                    scores.append((ref_name, score, snippet))
        scores.sort(key=lambda x: x[1], reverse=True)
        results[sub_name] = scores[:topk]
    return results

def detect_collusion(submissions, min_score=0.0, ngram_size=3):
    collusion = []
    sub_items = list(submissions.items())
    for i in range(len(sub_items)):
        for j in range(i+1, len(sub_items)):
            sub1_name, sub1_text = sub_items[i]
            sub2_name, sub2_text = sub_items[j]
            score = similarity_score(sub1_text, sub2_text, ngram_size)
            if score >= min_score:
                collusion.append((sub1_name, sub2_name, score))
    return collusion

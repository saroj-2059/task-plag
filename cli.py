import os
import glob
import argparse
import fitz
import docx2txt
import webbrowser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from html import escape
import spacy
import ast

nlp = spacy.load("en_core_web_sm")

def read_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".pdf":
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    elif ext == ".docx":
        return docx2txt.process(path)
    else:
        return ""

def preprocess_text(text):
    doc = nlp(text)
    cleaned = " ".join(token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space)
    return cleaned

def extract_python_code(text):
    try:
        tree = ast.parse(text)
        return ast.unparse(tree)
    except Exception:
        return text

def similarity_score(text1, text2, ngram_range=(1,5)):
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=ngram_range)
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def highlight_text(query, reference):
    query_words = set(query.lower().split())
    reference_words = reference.split()
    highlighted = [f"<mark>{escape(word)}</mark>" if word.lower() in query_words else escape(word)
                   for word in reference_words]
    return " ".join(highlighted)

def get_minhash(text, num_perm=128, ngram_size=5):
    shingles = [text[i:i+ngram_size] for i in range(len(text)-ngram_size+1)]
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(sh.encode("utf-8"))
    return m

def build_lsh_index(references, threshold=0.3, num_perm=128, ngram_size=5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    ref_minhashes = {}
    for ref_name, ref_text in references.items():
        m = get_minhash(ref_text, num_perm=num_perm, ngram_size=ngram_size)
        lsh.insert(ref_name, m)
        ref_minhashes[ref_name] = m
    return lsh, ref_minhashes

def detect_plagiarism(submissions, references, topk=5, min_score=0.0, ngram_size=5):
    refs_processed = {k: preprocess_text(v) for k,v in references.items()}
    subs_processed = {k: preprocess_text(v) for k,v in submissions.items()}

    lsh, ref_minhashes = build_lsh_index(refs_processed, threshold=min_score, ngram_size=ngram_size)

    results = {}
    for sub_name, sub_text in subs_processed.items():
        sub_minhash = get_minhash(sub_text, ngram_size=ngram_size)
        candidate_refs = lsh.query(sub_minhash)
        scores = []
        for ref_name in candidate_refs:
            score = similarity_score(sub_text, refs_processed[ref_name], ngram_range=(1, ngram_size))
            scores.append((ref_name, score, highlight_text(sub_text, refs_processed[ref_name])))
        scores.sort(key=lambda x: x[1], reverse=True)
        results[sub_name] = scores[:topk]
    return results

def detect_collusion(submissions, min_score=0.0, ngram_size=5):
    collusion = []
    sub_items = list(submissions.items())
    subs_processed = {k: preprocess_text(v) for k,v in submissions.items()}
    for i in range(len(sub_items)):
        for j in range(i+1, len(sub_items)):
            s1_name, s2_name = sub_items[i][0], sub_items[j][0]
            s1_text, s2_text = subs_processed[s1_name], subs_processed[s2_name]
            score = similarity_score(s1_text, s2_text, ngram_range=(1, ngram_size))
            if score >= min_score:
                collusion.append((s1_name, s2_name, score))
    return collusion

def generate_report(plag_results, collusion_results, out_file):
    html = ["<html><head><title>Plagiarism Report</title>"
            "<style>body{font-family:Arial;} table{border-collapse:collapse;} td,th{padding:5px;border:1px solid #ddd;} mark{background-color:yellow;}</style>"
            "</head><body>"]
    html.append("<h2>Plagiarism Report</h2>")
    html.append("<h3>Submission Matches Against References</h3>")
    for sub_name, matches in plag_results.items():
        html.append(f"<h4>{sub_name}</h4>")
        if matches:
            html.append("<table><tr><th>Reference File</th><th>Score</th><th>Plagiarized Text</th></tr>")
            for ref_name, score, snippet in matches:
                html.append(f"<tr><td>{ref_name}</td><td>{score:.2f}</td><td>{snippet}</td></tr>")
            html.append("</table>")
        else:
            html.append("<p>No matches found</p>")
    html.append("<h3>Submission Collusion Detection</h3>")
    if collusion_results:
        html.append("<table><tr><th>Submission 1</th><th>Submission 2</th><th>Score</th></tr>")
        for s1, s2, score in collusion_results:
            html.append(f"<tr><td>{s1}</td><td>{s2}</td><td>{score:.2f}</td></tr>")
        html.append("</table>")
    else:
        html.append("<p>No collusion detected</p>")
    html.append("</body></html>")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"Report saved to {out_file}")
    webbrowser.open(f"file://{os.path.abspath(out_file)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submissions", required=True, help="Folder with submission files")
    parser.add_argument("--refs", required=True, help="Reference files (supports glob)")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--ngram-size", type=int, default=5)
    parser.add_argument("--out", default="report.html")
    args = parser.parse_args()

    submissions = {}
    for path in glob.glob(os.path.join(args.submissions, "*")):
        text = read_file(path)
        if text.strip():
            submissions[os.path.basename(path)] = text

    references = {}
    for path in glob.glob(args.refs):
        text = read_file(path)
        if text.strip():
            references[os.path.basename(path)] = text

    plag_results = detect_plagiarism(submissions, references, topk=args.topk,
                                     min_score=args.min_score, ngram_size=args.ngram_size)
    collusion_results = detect_collusion(submissions, min_score=args.min_score, ngram_size=args.ngram_size)
    generate_report(plag_results, collusion_results, args.out)

if __name__ == "__main__":
    main()

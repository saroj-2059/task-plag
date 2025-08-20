from typing import List, Tuple
from features import paragraphs, normalize_text, build_tfidf, cosine_sim_matrix

def collusion_pairs(submissions: List[str], min_score: float = 0.3) -> List[Tuple[int,int,float]]:
    """
    Detect collusion between submissions at a paragraph level.
    Returns list of tuples: (submission_idx1, submission_idx2, similarity_score)
    """
    all_paragraphs = []
    submission_para_indices = [] 
    for idx, text in enumerate(submissions):
        paras = paragraphs(text)
        norm_paras = [normalize_text(p) for p in paras if p.strip()]
        all_paragraphs.extend(norm_paras)
        submission_para_indices.extend([idx]*len(norm_paras))

    if not all_paragraphs:
        return []

    vec, X = build_tfidf(all_paragraphs)
    S = cosine_sim_matrix(X, X)

    n = len(submissions)
    scores = [[0.0]*n for _ in range(n)]
    counts = [[0]*n for _ in range(n)]

    m = len(all_paragraphs)
    for i in range(m):
        for j in range(i+1, m):
            sub_i = submission_para_indices[i]
            sub_j = submission_para_indices[j]
            if sub_i != sub_j:
                scores[sub_i][sub_j] += S[i,j]
                counts[sub_i][sub_j] += 1

    out = []
    for i in range(n):
        for j in range(i+1, n):
            if counts[i][j] > 0:
                avg_score = scores[i][j] / counts[i][j]
                if avg_score >= min_score:
                    out.append((i,j,avg_score))

    return out

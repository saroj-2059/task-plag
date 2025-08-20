# tests/test_detector.py
from detector import detect_plagiarism

def test_basic_match():
    refs = ["this is an original paragraph about cats and dogs",
            "unrelated content entirely"]
    query = "A paragraph about cats and dogs appears here."
    res = detect_plagiarism(query, refs, topk=3, min_score=0.2)
    assert any(m.reference_doc_id == 0 for m in res.paragraph_matches)
    assert max(res.doc_scores.values()) > 0.2

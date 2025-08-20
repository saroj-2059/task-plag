# tests/test_explain.py
from explain import shared_ngrams

def test_shared():
    a = "machine learning is fun"
    b = "deep learning is fun sometimes"
    common = shared_ngrams(a, b, n=2)
    assert len(common) > 0

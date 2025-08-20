# Text Plagiarism Detector
## Overview

The Text Plagiarism Detector is a Python-based tool that detects potential plagiarism in text and code documents. It computes similarity scores between submissions and a reference corpus, cross-compares submissions for collusion, and generates a clean HTML report highlighting plagiarized content.

## Features

Supports .txt, .docx, and .pdf files.

Configurable n-gram size for detecting fine-grained plagiarism.

TF-IDF similarity computation with optional MinHash/LSH for large datasets.

Stemming/lemmatization for normalized matching.

AST/code analysis for detecting code similarity beyond text.

Cross-submission collusion detection.

Generates a highlighted HTML report showing plagiarized text, similarity scores, and references.

## Installation

Requires Python 3.9+ and the following libraries:

pip install scikit-learn python-docx PyMuPDF spacy
python -m spacy download en_core_web_sm

Usage
python cli.py --submissions <path_to_submissions_folder> \
              --refs "<path_to_reference_files/*>" \
              --topk 5 \
              --min-score 0.3 \
              --ngram-size 5 \
              --out report.html

## Parameters

--submissions: Folder containing student submission files.

--refs: Reference files (supports wildcards).

--topk: Number of top matching reference documents to display per submission.

--min-score: Minimum similarity score to report a match (set 0 to see all).

--ngram-size: N-gram size for text comparison.

--out: Path for saving the HTML report.

## Output

HTML report with:

Submission file name

Reference file name

Similarity score

Highlighted plagiarized text

Collusion detection results for submissions exceeding threshold.

## Limitations

- Cannot fully detect deep paraphrasing or translations.

- PDF extraction may fail for scanned/image-only PDFs.

- MinHash/LSH may miss very low similarity cases.

- Advanced code obfuscation may not be detected by AST analysis.

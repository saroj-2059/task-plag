Text Plagiarism Detector – Design Note

Methods

Preprocessing

Normalizes text by lowercasing, removing extra spaces, and optional punctuation removal.

Supports .txt, .docx, and .pdf files.

Optional stemming and lemmatization for improved token matching.

For code submissions, AST parsing extracts logical structure to detect semantic similarity.

Feature Extraction

N-gram shingles: Converts text into sequences of N consecutive words (configurable ngram_size) to capture contextual similarity.

TF-IDF Vectorization: Converts n-grams into numeric vectors weighted by term frequency and inverse document frequency.

MinHash / LSH (optional): For large datasets, approximate nearest neighbors can speed up similarity computation with probabilistic guarantees.

Similarity Detection

Cosine similarity between TF-IDF vectors is used to measure text similarity.

MinHash/LSH can be used for scalable fuzzy matching in larger corpora.

Configurable min_score threshold to report matches.

Plagiarism Reporting

Generates a clean HTML report with:

Submission file name

Reference file name

Similarity score

Highlighted plagiarized text in a single column for readability

Detects submission collusion by cross-comparing all submissions.

Trade-offs

Accuracy vs Speed:

Using full TF-IDF with n-grams is accurate but slower for large corpora.

MinHash/LSH trades exact similarity for faster approximate results.

Text vs Code:

Text similarity works well for essays and documents.

Code similarity detection via AST improves semantic detection but increases preprocessing complexity.

File Type Support:

Supports .txt, .docx, .pdf.

Requires external libraries (python-docx, PyMuPDF) which may limit portability without installation.

Limitations

Cannot detect plagiarism if content is completely paraphrased or translated into another language.

PDF extraction might occasionally fail for scanned PDFs or images.

MinHash/LSH approximation may occasionally miss low-level similarities.

Current AST/code analysis is basic; advanced obfuscation in code may not be detected.
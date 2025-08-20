from jinja2 import Template
from detector import DocResult

_HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>Plagiarism Report</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
        h1, h2, h3 { color: #333; }
        mark { background: #fff59d; }
        .score { font-weight: bold; color: #00796B; }
        .card { border: 1px solid #ddd; border-radius: 8px; padding: 12px; margin-bottom: 15px; background: #fafafa; }
        .para { white-space: pre-wrap; background: #fff; border-radius: 6px; padding: 8px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ccc; padding: 6px; text-align: left; }
    </style>
</head>
<body>
    <h1>Plagiarism Report</h1>
    <p><strong>Parameters:</strong> TopK={{ res.params.topk }}, Min Score={{ res.params.min_score }}</p>

    <h2>Document Scores</h2>
    <ul>
    {% for doc_id, score in res.doc_scores.items() %}
        <li>Reference {{ doc_id }} — <span class="score">{{ "%.3f"|format(score) }}</span></li>
    {% endfor %}
    </ul>

    <h2>Top Matches</h2>
    {% for m in res.paragraph_matches %}
        <div class="card">
            <div><strong>Reference Doc:</strong> {{ m.reference_doc_id }} | 
                 <strong>Paragraph:</strong> {{ m.reference_para_id }} | 
                 <strong>Score:</strong> <span class="score">{{ "%.3f"|format(m.score) }}</span></div>
            <h3>Query Paragraph</h3>
            <div class="para">{{ m.query_highlight_html | safe }}</div>
            <h3>Reference Paragraph</h3>
            <div class="para">{{ m.ref_highlight_html | safe }}</div>
        </div>
    {% endfor %}

    <h2>Collusion Detection</h2>
    {% if res.collusion_pairs %}
    <table>
        <tr><th>Reference Doc A</th><th>Reference Doc B</th><th>Similarity Score</th></tr>
        {% for c in res.collusion_pairs %}
        <tr>
            <td>{{ c.doc_a_id }}</td>
            <td>{{ c.doc_b_id }}</td>
            <td class="score">{{ "%.3f"|format(c.score) }}</td>
        </tr>
        {% endfor %}
    </table>
    {% else %}
    <p>No collusion detected above threshold.</p>
    {% endif %}
</body>
</html>
"""

def render_html(res: DocResult) -> str:
    template = Template(_HTML_TEMPLATE)
    return template.render(res=res)

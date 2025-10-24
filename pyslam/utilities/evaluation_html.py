import pandas as pd
from pathlib import Path
from jinja2 import Template
import webbrowser
import html
from datetime import datetime
import getpass


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background: #f9f9f9;
            padding: 2rem;
            color: #333;
        }
        h1 {
            text-align: center;
            margin-bottom: 2rem;
        }
        .table-wrapper {
            margin-bottom: 2.5rem;
        }
        .caption {
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: #222;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            border-radius: 6px;
            overflow: hidden;
            font-size: 0.8rem;  /* Smaller font for table content */
        }
        th, td {
            border: 1px solid #ddd;
            padding: 0.5rem 0.6rem;
            text-align: left;
        }
        thead th {
            background: #007acc;
            color: white;
            font-weight: 600;
        }
        tbody tr:nth-child(even) {
            background: #f5faff;
        }
    </style>
</head>
<body>

<h1>{{ title }}</h1>
<div style="text-align: center; font-size: 0.85rem;  margin-bottom: 2rem;">
    {{ user }} &nbsp;|&nbsp; {{ date }} &nbsp;|&nbsp; Commit: <code>{{ git_hash }}</code>
</div>

{% for table in tables %}
<div class="table-wrapper">
    <div class="caption">{{ table.caption }}</div>
    <table>
        <thead>
            <tr>
                {% for col in table.columns %}
                <th>{{ col }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in table.data %}
            <tr>
                {% for cell in row %}
                <td>{{ cell }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
</div>
{% endfor %}

</body>
</html>
"""


def escape_html_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.applymap(lambda x: html.escape(str(x)) if pd.notna(x) else "")


def csv_list_to_html(
    csv_paths, output_html_path, title="Evaluation Report", git_commit_hash=None, open_browser=True
):
    tables = []
    author_name = getpass.getuser().capitalize()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
        if df.empty:
            continue
        escaped_df = escape_html_df(df)
        table_data = {
            "caption": Path(csv_path).stem.replace("_", " ").title(),
            "columns": escaped_df.columns.tolist(),
            "data": escaped_df.values.tolist(),
        }
        tables.append(table_data)

    template = Template(html_template)
    rendered_html = template.render(
        tables=tables,
        title=title,
        user=author_name,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        git_hash=git_commit_hash,
    )

    Path(output_html_path).write_text(rendered_html, encoding="utf-8")
    print(f"✅ HTML report saved to: {output_html_path}")

    if open_browser:
        webbrowser.open(f"file://{Path(output_html_path).resolve()}")

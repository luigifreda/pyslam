import pandas as pd
from pathlib import Path
from jinja2 import Template, Environment
import subprocess
import tempfile
import getpass
import re
import os
import platform


def open_pdf(filepath):
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["open", filepath])
        elif system == "Windows":
            os.startfile(filepath)
        elif system == "Linux":
            subprocess.run(["xdg-open", filepath])
    except Exception as e:
        print(f"❌ Failed to open PDF: {e}")


def escape_latex(text):
    if not isinstance(text, str):
        return text
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


latex_template = r"""
\documentclass{article}
\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{ltablex}
\usepackage{caption}
\usepackage{capt-of}
\usepackage{textcomp}
\usepackage{array}
\usepackage{ragged2e}
\usepackage{adjustbox}
\keepXColumns
\renewcommand{\arraystretch}{1.2}

\title{ {{ title }} }
\author{ {{ user }} \\ Commit: {{ git_hash }} }
\begin{document}

\maketitle

{% for table in tables %}
\begin{minipage}{\textwidth}
\noindent
\captionsetup{type=table}
\captionof{table}{ {{ table.caption }} }\label{tab:{{ table.label }}}

{% set col_count = table.columns | length %}
{% if col_count > 12 %}
\fontsize{5pt}{6pt}\selectfont
{% elif col_count > 9 %}
\fontsize{7pt}{8pt}\selectfont
{% elif col_count > 6 %}
\fontsize{8pt}{9pt}\selectfont
{% else %}
\fontsize{9pt}{10pt}\selectfont
{% endif %}

\begin{tabularx}{\linewidth}{ {% for w in table.widths %}>{\RaggedRight\arraybackslash}p{ {{w }} } {% endfor %} }\toprule
{{ table.columns | map('escape') | join(' & ') }} \\
\midrule
{% for row in table.data -%}
{{ row | map('escape') | join(' & ') }} \\
{% endfor %}
\bottomrule
\end{tabularx}
\end{minipage}

\bigskip
{% endfor %}

\end{document}
"""


def clean_non_ascii(df):
    return df.map(lambda x: re.sub(r"[^\x20-\x7E]", "", str(x)))


def compute_column_widths(
    df, max_total_width_cm=15.5, max_chars_per_col=50, first_col_scale=0.9, col_scale=0.6
):
    header_lens = [len(str(col)) for col in df.columns]
    content_lens = [min(max(df[col].astype(str).map(len)), max_chars_per_col) for col in df.columns]

    col_lengths = [max(h, c) for h, c in zip(header_lens, content_lens)]

    # Scale down the columns
    for i, col_length in enumerate(col_lengths):
        if i == 0:
            col_lengths[i] = int(col_length * first_col_scale)
        else:
            col_lengths[i] = int(col_length * col_scale)

    total_chars = sum(col_lengths)

    # Scale so that total width only matches actual content, or max limit
    base_cm = min(max_total_width_cm / total_chars, 0.25) if total_chars > 0 else 0.2

    # Compute width per column
    min_width_cm = 0.8
    widths = [f"{max(round(cl * base_cm, 2), min_width_cm)}cm" for cl in col_lengths]

    return widths


def csv_list_to_pdf(
    csv_paths, output_pdf_path, title="CSV Tables", git_commit_hash=None, open_pdf_output=True
):
    latex_tables = []

    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path, keep_default_na=False, na_values=[])
        if df.empty or df.shape[1] == 0:
            print(f"⚠️ Skipping empty or malformed CSV: {csv_path}")
            continue

        df = clean_non_ascii(df)

        print(f"stem: {Path(csv_path).stem}")

        table_data = {
            "caption": Path(csv_path).stem.replace("_", " ").title(),
            "label": Path(csv_path).stem,
            "columns": df.columns.tolist(),
            "data": df.values.tolist(),
            "widths": compute_column_widths(df),
        }
        print(f"🧪 Table {table_data['label']} columns: {table_data['columns']}")
        print(f"📐 Widths: {table_data['widths']}")
        print(f"📊 Caption: {table_data['caption']}")
        if git_commit_hash:
            print(f"🔑 Git commit hash: {git_commit_hash}")
        latex_tables.append(table_data)

    author_name = getpass.getuser().capitalize()

    env = Environment()
    env.filters["escape"] = escape_latex
    template = env.from_string(latex_template)
    rendered_latex = template.render(
        tables=latex_tables, title=title, user=author_name, git_hash=git_commit_hash
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tex_path = Path(tmpdir) / "report.tex"
        with open(tex_path, "w") as f:
            f.write(rendered_latex)

        try:
            output_path = os.path.dirname(output_pdf_path)
            output_latex_file_path = os.path.join(output_path, f"report.tex")
            Path(output_latex_file_path).write_bytes(tex_path.read_bytes())
            print(f"✅ Successfully saved tex: {output_latex_file_path}")
        except Exception as e:
            print(f"❌ Error saving tex: {e}")

        try:
            subprocess.run(
                [
                    "pdflatex",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    tmpdir,
                    tex_path.name,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ LaTeX Compilation Error:\n{e}")
            print(tex_path.read_text())
            return

        pdf_path = Path(tmpdir) / "report.pdf"
        Path(output_pdf_path).write_bytes(pdf_path.read_bytes())
        print(f"✅ Successfully created PDF: {output_pdf_path}")
        if open_pdf_output:
            open_pdf(str(output_pdf_path))

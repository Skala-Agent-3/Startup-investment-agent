import markdown
import pdfkit
import os

MARKDOWN_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "toc"]


def markdown_to_html(markdown_text):
    """🎉 Render markdown with table support and embed lightweight styling."""
    html_content = markdown.markdown(markdown_text, extensions=MARKDOWN_EXTENSIONS)

    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8" />
        <style>
            body {{
                font-family: 'Malgun Gothic', sans-serif;
                color: #333;
                line-height: 1.5;
            }}
            h1, h2, h3 {{
                font-family: 'Malgun Gothic', sans-serif;
                margin-top: 1.6em;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1.2em 0;
                font-size: 0.95em;
            }}
            th, td {{
                border: 1px solid #999;
                padding: 8px 10px;
                vertical-align: top;
            }}
            th {{
                background-color: #f0f3f8;
                font-weight: 600;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    return html_content


def save_report_to_pdf(report_generation, output_path):
    """🎉 Convert markdown output into a PDF using wkhtmltopdf."""
    html_content = markdown_to_html(report_generation)
    html_content = html_content.replace('src="assets/', 'src="assets/')

    path_to_wkhtmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
    config = pdfkit.configuration(wkhtmltopdf=path_to_wkhtmltopdf)

    options = {
        'no-outline': None,
        'encoding': 'UTF-8',
        'disable-javascript': '',
        'zoom': '1.2',
        'enable-local-file-access': None,
    }

    pdfkit.from_string(html_content, output_path, configuration=config, options=options)

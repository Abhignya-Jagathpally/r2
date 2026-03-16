"""
HTML report generation for MM risk-signature pipeline.
Integrates figures, tables, metrics, and MLflow tracking.
"""

import os
import base64
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt


class ReportGenerator:
    """Generate comprehensive HTML reports for pipeline analysis."""

    def __init__(self, output_dir='reports', title='MM Risk-Signature Analysis'):
        """
        Initialize report generator.

        Parameters
        ----------
        output_dir : str
            Output directory for reports
        title : str
            Report title
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.title = title
        self.figures = {}
        self.tables = {}
        self.metrics = {}
        self.sections = []

    def add_figure(self, name, fig_or_path, section='Results'):
        """
        Add figure to report.

        Parameters
        ----------
        name : str
            Figure name/caption
        fig_or_path : matplotlib.figure.Figure or str
            Figure object or path to image file
        section : str
            Section name
        """
        if section not in self.figures:
            self.figures[section] = []

        if isinstance(fig_or_path, str):
            # Load from path
            with open(fig_or_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()
            ext = Path(fig_or_path).suffix
        else:
            # Convert matplotlib figure to PNG
            buf = BytesIO()
            fig_or_path.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode()
            ext = '.png'

        self.figures[section].append({
            'name': name,
            'data': img_data,
            'ext': ext,
        })

    def add_table(self, name, df, section='Results', index=True):
        """
        Add table to report.

        Parameters
        ----------
        name : str
            Table name/caption
        df : pd.DataFrame
            Table data
        section : str
        index : bool
        """
        if section not in self.tables:
            self.tables[section] = []

        self.tables[section].append({
            'name': name,
            'data': df,
            'index': index,
        })

    def add_metric(self, name, value, section='Metrics'):
        """
        Add scalar metric to report.

        Parameters
        ----------
        name : str
        value : float or str
        section : str
        """
        if section not in self.metrics:
            self.metrics[section] = {}

        self.metrics[section][name] = value

    def _html_header(self):
        """Generate HTML header."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        nav {{
            background-color: #333;
            padding: 10px 20px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}

        nav ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}

        nav a {{
            color: white;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }}

        nav a:hover {{
            color: #667eea;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }}

        section {{
            background: white;
            margin-bottom: 40px;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        section h2 {{
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
            font-size: 1.8em;
            color: #333;
        }}

        section h3 {{
            margin-top: 30px;
            margin-bottom: 15px;
            color: #555;
            font-size: 1.3em;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }}

        .metric-card .value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}

        .metric-card .label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}

        .figure-container {{
            margin: 30px 0;
            text-align: center;
        }}

        .figure-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin: 15px 0;
        }}

        .figure-caption {{
            font-style: italic;
            color: #666;
            margin-top: 10px;
            font-size: 0.95em;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.95em;
        }}

        table thead {{
            background-color: #667eea;
            color: white;
        }}

        table th, table td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        table tbody tr:hover {{
            background-color: #f9f9f9;
        }}

        table tbody tr:nth-child(even) {{
            background-color: #f5f5f5;
        }}

        .table-container {{
            overflow-x: auto;
            margin: 20px 0;
        }}

        footer {{
            background-color: #333;
            color: white;
            text-align: center;
            padding: 20px;
            margin-top: 40px;
            font-size: 0.9em;
        }}

        .toc {{
            background-color: #f9f9f9;
            border-left: 4px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 4px;
        }}

        .toc ul {{
            list-style: none;
            padding-left: 0;
        }}

        .toc li {{
            padding: 5px 0;
        }}

        .toc a {{
            color: #667eea;
            text-decoration: none;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        @media (max-width: 768px) {{
            header h1 {{
                font-size: 1.8em;
            }}

            nav ul {{
                flex-direction: column;
                gap: 10px;
            }}

            .metrics-grid {{
                grid-template-columns: 1fr;
            }}

            section {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>{self.title}</h1>
        <div class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </header>
"""
        return html

    def _html_nav(self):
        """Generate navigation menu."""
        all_sections = set(list(self.metrics.keys()) +
                          list(self.tables.keys()) +
                          list(self.figures.keys()))

        nav_items = ''.join([
            f'<li><a href="#{sec.replace(" ", "-")}">{sec}</a></li>'
            for sec in sorted(all_sections)
        ])

        return f"""
    <nav>
        <ul>
            {nav_items}
        </ul>
    </nav>
"""

    def _html_toc(self):
        """Generate table of contents."""
        all_sections = set(list(self.metrics.keys()) +
                          list(self.tables.keys()) +
                          list(self.figures.keys()))

        toc_items = ''.join([
            f'<li><a href="#{sec.replace(" ", "-")}">{sec}</a></li>'
            for sec in sorted(all_sections)
        ])

        return f"""
    <section class="toc">
        <h2>Table of Contents</h2>
        <ul>
            {toc_items}
        </ul>
    </section>
"""

    def _html_metrics_section(self, section_name):
        """Generate metrics section HTML."""
        if section_name not in self.metrics:
            return ''

        metrics = self.metrics[section_name]

        cards = ''.join([
            f"""
            <div class="metric-card">
                <div class="label">{name}</div>
                <div class="value">{value:.4f if isinstance(value, float) else value}</div>
            </div>
            """
            for name, value in metrics.items()
        ])

        return f"""
    <section id="{section_name.replace(" ", "-")}">
        <h2>{section_name}</h2>
        <div class="metrics-grid">
            {cards}
        </div>
    </section>
"""

    def _html_tables_section(self, section_name):
        """Generate tables section HTML."""
        if section_name not in self.tables:
            return ''

        tables = self.tables[section_name]

        tables_html = ''.join([
            f"""
            <h3>{table['name']}</h3>
            <div class="table-container">
                {table['data'].to_html(classes='data-table', index=table['index'])}
            </div>
            """
            for table in tables
        ])

        return f"""
    <section id="{section_name.replace(" ", "-")}">
        <h2>{section_name}</h2>
        {tables_html}
    </section>
"""

    def _html_figures_section(self, section_name):
        """Generate figures section HTML."""
        if section_name not in self.figures:
            return ''

        figures = self.figures[section_name]

        figures_html = ''.join([
            f"""
            <div class="figure-container">
                <img src="data:image/{fig['ext'].strip('.')};base64,{fig['data']}"
                     alt="{fig['name']}" style="max-width: 100%; height: auto;">
                <div class="figure-caption">{fig['name']}</div>
            </div>
            """
            for fig in figures
        ])

        return f"""
    <section id="{section_name.replace(" ", "-")}">
        <h2>{section_name}</h2>
        {figures_html}
    </section>
"""

    def _html_footer(self):
        """Generate HTML footer."""
        return """
    <footer>
        <p>This report was automatically generated by the MM Risk-Signature Pipeline.</p>
        <p>&copy; 2026 Analysis Report</p>
    </footer>
</body>
</html>
"""

    def generate(self, filename='report.html'):
        """
        Generate complete HTML report.

        Parameters
        ----------
        filename : str
            Output filename
        """
        html = self._html_header()
        html += self._html_nav()

        # Add content in order
        html += '<div class="container">'
        html += self._html_toc()

        all_sections = set(list(self.metrics.keys()) +
                          list(self.tables.keys()) +
                          list(self.figures.keys()))

        for section in sorted(all_sections):
            html += self._html_metrics_section(section)
            html += self._html_tables_section(section)
            html += self._html_figures_section(section)

        html += '</div>'
        html += self._html_footer()

        # Write file
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            f.write(html)

        print(f"Report generated: {output_path}")
        return output_path


class MLflowReporter:
    """MLflow integration for pipeline tracking."""

    @staticmethod
    def log_figure(fig, name, artifact_path='figures'):
        """
        Log matplotlib figure to MLflow.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
        name : str
            Figure name
        artifact_path : str
        """
        import mlflow

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)

        with open(f'/tmp/{name}.png', 'wb') as f:
            f.write(buf.getvalue())

        mlflow.log_artifact(f'/tmp/{name}.png', artifact_path=artifact_path)

    @staticmethod
    def log_table(df, name, artifact_path='tables'):
        """
        Log DataFrame to MLflow.

        Parameters
        ----------
        df : pd.DataFrame
        name : str
        artifact_path : str
        """
        import mlflow

        csv_path = f'/tmp/{name}.csv'
        df.to_csv(csv_path)
        mlflow.log_artifact(csv_path, artifact_path=artifact_path)

    @staticmethod
    def log_metrics(metrics_dict, prefix=''):
        """
        Log scalar metrics to MLflow.

        Parameters
        ----------
        metrics_dict : dict
        prefix : str
        """
        import mlflow

        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f'{prefix}{key}', value)

    @staticmethod
    def log_params(params_dict, prefix=''):
        """
        Log parameters to MLflow.

        Parameters
        ----------
        params_dict : dict
        prefix : str
        """
        import mlflow

        for key, value in params_dict.items():
            mlflow.log_param(f'{prefix}{key}', str(value))


def create_summary_table(metrics_dict):
    """
    Create summary table from metrics dictionary.

    Parameters
    ----------
    metrics_dict : dict
        {model_name: {metric: value}}

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(metrics_dict).T
    df = df.round(4)
    return df


def format_confidence_interval(estimate, ci_lower, ci_upper):
    """
    Format confidence interval string.

    Parameters
    ----------
    estimate : float
    ci_lower, ci_upper : float

    Returns
    -------
    str
        "0.72 (0.68-0.76)"
    """
    return f"{estimate:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"

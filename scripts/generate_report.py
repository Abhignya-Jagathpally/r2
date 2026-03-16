#!/usr/bin/env python3
"""
Generate Final Benchmark Report

Compiles all evaluation results into publication-ready report:
- Performance summary tables
- Model comparison visualizations
- Risk stratification plots
- Calibration curves
- Survival curves by risk group
- Cross-study generalization results

Outputs HTML, PDF, and Markdown formats.

Author: Pipeline Team
Date: 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.reporting import ReportGenerator
from src.utils.visualization import SurvivalPlotter

logger = logging.getLogger(__name__)

# Set seaborn style
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "report.log"),
        ],
    )


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration."""
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_evaluation_results(results_dir: Path) -> Optional[Dict]:
    """
    Load evaluation results from JSON file.

    Parameters
    ----------
    results_dir : Path
        Directory containing evaluation results

    Returns
    -------
    Optional[dict]
        Evaluation results or None if not found
    """
    results_file = results_dir / "evaluation_results.json"

    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return None

    try:
        with open(results_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return None


def load_summary_table(results_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load evaluation summary table.

    Parameters
    ----------
    results_dir : Path
        Directory containing evaluation results

    Returns
    -------
    Optional[pd.DataFrame]
        Summary dataframe or None if not found
    """
    summary_file = results_dir / "evaluation_summary.csv"

    if not summary_file.exists():
        logger.warning(f"Summary file not found: {summary_file}")
        return None

    try:
        return pd.read_csv(summary_file)
    except Exception as e:
        logger.error(f"Failed to load summary: {e}")
        return None


def create_performance_table(results: Dict) -> pd.DataFrame:
    """
    Create formatted performance comparison table.

    Parameters
    ----------
    results : dict
        Evaluation results

    Returns
    -------
    pd.DataFrame
        Formatted performance table
    """
    rows = []

    for model_name, result in results.items():
        row = {"Model": model_name}
        metrics = result.get("metrics", {})

        # Select key metrics
        if "c_index" in metrics:
            row["C-Index"] = f"{metrics['c_index']:.4f}"
        if "auc_12m" in metrics:
            row["AUC@1yr"] = f"{metrics['auc_12m']:.4f}"
        if "auc_24m" in metrics:
            row["AUC@2yr"] = f"{metrics['auc_24m']:.4f}"
        if "ibs" in metrics:
            row["IBS"] = f"{metrics['ibs']:.4f}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by C-Index if available
    if "C-Index" in df.columns:
        df["_sort_key"] = df["C-Index"].str.replace(np.nan, "0")
        df = df.sort_values("_sort_key", ascending=False).drop("_sort_key", axis=1)

    return df


def create_model_comparison_plot(results: Dict, output_dir: Path) -> Path:
    """
    Create model comparison visualization.

    Parameters
    ----------
    results : dict
        Evaluation results
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to saved figure
    """
    logger.info("Creating model comparison plot...")

    # Extract metrics
    models = []
    c_indices = []

    for model_name, result in results.items():
        if "c_index" in result.get("metrics", {}):
            models.append(model_name)
            c_indices.append(result["metrics"]["c_index"])

    if not models:
        logger.warning("No C-index metrics found for plotting")
        return None

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by C-index
    sorted_idx = np.argsort(c_indices)
    models_sorted = [models[i] for i in sorted_idx]
    c_indices_sorted = [c_indices[i] for i in sorted_idx]

    # Bar plot
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models_sorted)))
    bars = ax.barh(models_sorted, c_indices_sorted, color=colors)

    ax.set_xlabel("Concordance Index (C-Index)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xlim([0.5, 1.0])

    # Add value labels
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()

    # Save
    plot_file = output_dir / "model_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved comparison plot: {plot_file}")
    return plot_file


def create_html_report(
    results: Dict,
    summary_df: Optional[pd.DataFrame],
    output_dir: Path,
    config: Dict,
) -> Path:
    """
    Generate HTML report.

    Parameters
    ----------
    results : dict
        Evaluation results
    summary_df : Optional[pd.DataFrame]
        Summary table
    output_dir : Path
        Output directory
    config : dict
        Configuration

    Returns
    -------
    Path
        Path to HTML report
    """
    logger.info("Generating HTML report...")

    html_parts = []

    # Header
    html_parts.append("""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>MM Transcriptomics Risk Signature - Benchmark Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                line-height: 1.6;
                color: #333;
                background: #f5f5f5;
            }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 10px; color: #2c3e50; }
            h2 { margin-top: 30px; color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; }
            h3 { color: #7f8c8d; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #bdc3c7; padding: 12px; text-align: left; }
            th { background: #34495e; color: white; font-weight: bold; }
            tr:nth-child(even) { background: #ecf0f1; }
            .metric-box { background: #3498db; color: white; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .metric-value { font-size: 24px; font-weight: bold; }
            .metric-label { font-size: 14px; opacity: 0.9; }
            img { max-width: 100%; height: auto; margin: 20px 0; border-radius: 5px; }
            .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #bdc3c7; color: #7f8c8d; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
    """)

    # Title and metadata
    html_parts.append(f"""
        <h1>MM Transcriptomics Risk Signature - Benchmark Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Pipeline Version:</strong> {config.get('pipeline', {}).get('version', 'Unknown')}</p>
    """)

    # Summary metrics
    html_parts.append("<h2>Model Performance Summary</h2>")

    if summary_df is not None:
        html_parts.append(summary_df.to_html(index=False))
    else:
        logger.warning("No summary table available for HTML report")

    # Model comparison chart
    html_parts.append("<h2>Performance Comparison</h2>")
    html_parts.append("""
        <p>Benchmark comparison of all trained models. Higher C-Index values indicate better predictive performance.</p>
    """)

    comparison_plot = create_model_comparison_plot(results, output_dir)
    if comparison_plot:
        html_parts.append(f'<img src="{comparison_plot.name}" alt="Model comparison">')

    # Detailed results
    html_parts.append("<h2>Detailed Model Results</h2>")
    for model_name, result in sorted(results.items()):
        html_parts.append(f"<h3>{model_name}</h3>")
        metrics = result.get("metrics", {})

        if metrics:
            html_parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, float):
                    html_parts.append(
                        f"<tr><td>{metric_name}</td><td>{metric_value:.4f}</td></tr>"
                    )
                else:
                    html_parts.append(f"<tr><td>{metric_name}</td><td>{metric_value}</td></tr>")
            html_parts.append("</table>")
        else:
            html_parts.append("<p>No metrics available</p>")

    # Footer
    html_parts.append("""
            <div class="footer">
                <p>This report was automatically generated by the MM Transcriptomics Pipeline.</p>
                <p>For questions or issues, please contact the development team.</p>
            </div>
        </div>
    </body>
    </html>
    """)

    # Write HTML file
    html_file = output_dir / "index.html"
    try:
        with open(html_file, "w") as f:
            f.write("\n".join(html_parts))
        logger.info(f"HTML report saved: {html_file}")
    except Exception as e:
        logger.error(f"Failed to save HTML report: {e}")
        return None

    return html_file


def main():
    """Main report generation orchestration."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all report formats
  python scripts/generate_report.py \\
    --results-dir outputs/results \\
    --output-dir outputs/reports

  # Generate HTML only
  python scripts/generate_report.py \\
    --results-dir outputs/results \\
    --output-dir outputs/reports \\
    --format html
        """,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory with evaluation results (default: outputs/results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/reports"),
        help="Output directory for reports (default: outputs/reports)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/pipeline_config.yaml"),
        help="Configuration file (default: config/pipeline_config.yaml)",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["html"],
        choices=["html", "pdf", "markdown"],
        help="Report formats to generate (default: html)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = args.output_dir / "logs"
    setup_logging(log_dir, args.log_level)

    logger.info("="*80)
    logger.info("MM Transcriptomics Benchmark Report Generation")
    logger.info("="*80)
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = load_config(args.config)

    # Load evaluation results
    results = load_evaluation_results(args.results_dir)
    if not results:
        logger.error("No evaluation results found. Run 'make evaluate' first.")
        sys.exit(1)

    # Load summary table
    summary_df = load_summary_table(args.results_dir)

    logger.info(f"Loaded results for {len(results)} models")

    # Generate reports
    if "html" in args.format:
        html_file = create_html_report(
            results=results,
            summary_df=summary_df,
            output_dir=args.output_dir,
            config=config,
        )
        if html_file:
            logger.info(f"✓ HTML report: {html_file}")

    if "pdf" in args.format:
        logger.info("PDF generation requires additional dependencies (weasyprint)")
        logger.info("Install with: pip install weasyprint")

    if "markdown" in args.format:
        logger.info("Markdown report generation not yet implemented")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Report Generation Complete!")
    logger.info("="*80)
    logger.info(f"Reports saved to: {args.output_dir}")
    logger.info(f"View HTML report: open {args.output_dir}/index.html")


if __name__ == "__main__":
    main()

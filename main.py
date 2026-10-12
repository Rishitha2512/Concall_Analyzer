from pathlib import Path
import argparse
import sys

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import OUTPUT_FOLDER, MAX_QUARTERS

import pdf_extractor
import nlp_analyser
import html_reporter
import csv_reporter
print("MAIN.PY LOADED")
console = Console()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Concall Analyser"
    )

    parser.add_argument(
    "--company",
    help="Company name, NSE symbol or BSE code"
    )

    

    parser.add_argument(
        "--manual",
        nargs="*",
        help="Manual PDF files"
    )

    parser.add_argument(
        "--output",
        nargs="*",
        default=["html", "csv", "json", "xlsx"],
        choices=["html", "csv", "json", "xlsx"],
        help="Output formats"
    )

    parser.add_argument(
        "--verbose",
        action="store_true"
    )

    return parser


def load_transcripts(args):

    console.print(f"[cyan]Loading company: {args.company}[/cyan]")

    transcripts = pdf_extractor.load_company_pdfs(args.company)

    return transcripts

def save_reports(
    analysis_result,
    company_name,
    output_formats
):
    output_dir = Path(OUTPUT_FOLDER)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_name = (
        company_name
        .replace(" ", "_")
        .replace("/", "_")
    )

    # ----------------------------------
    # HTML
    # ----------------------------------

    if "html" in output_formats:

        html_file = (
            output_dir /
            f"{safe_name}_report.html"
        )

        html_reporter.generate_html(
            analysis_result,
            company_name,
            html_file
        )

        console.print(
            f"[green]HTML:[/green] {html_file}"
        )

    # ----------------------------------
    # CSV
    # ----------------------------------

    if "csv" in output_formats:

        csv_file = (
            output_dir /
            f"{safe_name}_report.csv"
        )

        csv_reporter.generate_csv_report(
            analysis_result["quarterly_analyses"],
            company_name,
            csv_file
        )

        console.print(
            f"[green]CSV:[/green] {csv_file}"
        )

    # ----------------------------------
    # JSON
    # ----------------------------------

    if "json" in output_formats:

        json_file = (
            output_dir /
            f"{safe_name}_report.json"
        )

        csv_reporter.generate_json_report(
            analysis_result["quarterly_analyses"],
            company_name,
            json_file
        )

        console.print(
            f"[green]JSON:[/green] {json_file}"
        )

    # ----------------------------------
    # EXCEL
    # ----------------------------------

    if "xlsx" in output_formats:

        excel_file = (
            output_dir /
            f"{safe_name}_report.xlsx"
        )

        csv_reporter.generate_excel_report(
            analysis_result["quarterly_analyses"],
            company_name,
            excel_file
        )

        console.print(
            f"[green]EXCEL:[/green] {excel_file}"
        )

def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.company:

        company_name = input(
            "Enter Company Name: "
        ).strip()

        if not company_name:
            console.print(
                "[red]Company name cannot be empty.[/red]"
            )
            sys.exit(1)

        args.company = company_name.upper()

    else:
        args.company = args.company.strip()

    try:
        if args.company.upper() == "ALL":
            target_companies = pdf_extractor.list_companies()
            console.print(f"[bold cyan]Processing all {len(target_companies)} companies across all quarters...[/bold cyan]")
        else:
            target_companies = [args.company]

        for comp in target_companies:
            console.print(f"\n[cyan]Loading company: {comp}[/cyan]")
            transcripts = pdf_extractor.load_company_pdfs(comp)

            if not transcripts:
                console.print(f"[yellow]No transcripts available for {comp}.[/yellow]")
                continue

            quarter_labels = [t["quarter"] for t in transcripts]
            console.print(
                f"[green]Loaded {len(transcripts)} quarter(s) for {comp}: {', '.join(quarter_labels)}[/green]"
            )

            console.print(f"[cyan]Running intelligence pipeline across {len(transcripts)} quarter(s)...[/cyan]")
            analysis_result = nlp_analyser.run_full_analysis(
                transcripts=transcripts,
                company_name=comp,
                verbose=args.verbose,
            )

            save_reports(
                analysis_result,
                comp,
                args.output,
            )

            console.print(f"[bold green]Analysis Complete for {comp}[/bold green]")

        console.print("\n[bold green]Pipeline execution completed successfully for all quarters.[/bold green]")

    except Exception as e:
        console.print(
            f"[bold red]Error:[/bold red] {e}"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
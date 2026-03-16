process generate_report {
    tag "report"
    publishDir "${params.outdir}/reports", mode: 'copy'

    input:
    path results

    output:
    path "pipeline_report.html"

    script:
    """
    python -c "
html = '<html><body><h1>MM Pipeline Report</h1><p>See outputs/ for detailed results.</p></body></html>'
open('pipeline_report.html', 'w').write(html)
"
    """
}

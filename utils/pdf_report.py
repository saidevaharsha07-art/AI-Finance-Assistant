"""PDF report generation for Finance Assistant V2."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO


def build_pdf(inputs: dict, result: dict) -> BytesIO:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("AI-Powered Finance Assistant Report", styles["Title"]),
        Paragraph(datetime.now().strftime("Generated on %d %B %Y, %I:%M %p"), styles["Normal"]),
        Spacer(1, 14),
    ]

    summary = [
        ["Health Score", f"{result['health_score']}/100"],
        ["Health Category", result["health_category"]],
        ["Spender Class", result["spender_class"]],
        ["Overspending", "Yes" if result["overspending"] else "No"],
        ["Estimated Savings", f"Rs. {result['actual_savings']:.0f}"],
        ["6-Month Forecast", f"Rs. {result['forecast']['month_6']:.0f}"],
        ["12-Month Forecast", f"Rs. {result['forecast']['month_12']:.0f}"],
    ]
    story.append(Paragraph("Prediction Summary", styles["Heading2"]))
    story.append(_table(summary))
    story.append(Spacer(1, 12))

    input_rows = [[key.replace("_", " ").title(), str(value)] for key, value in inputs.items()]
    story.append(Paragraph("Input Summary", styles["Heading2"]))
    story.append(_table(input_rows))
    story.append(Spacer(1, 12))

    budget_rows = [
        [name.replace("_", " ").title(), f"Rs. {amount:.0f}", f"{result['recommended_percentages'][name]}%"]
        for name, amount in result["recommended_amounts"].items()
    ]
    story.append(Paragraph("Recommended Monthly Budget", styles["Heading2"]))
    story.append(_table([["Category", "Amount", "Percent"]] + budget_rows, header=True))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Personalized Advice", styles["Heading2"]))
    for item in result["advice"]:
        story.append(Paragraph(f"- {item}", styles["Normal"]))

    story.append(Spacer(1, 12))
    story.append(Paragraph("Suggested Investment Plan", styles["Heading2"]))
    investment_rows = [[item["name"], f"{item['percentage']}%", f"Rs. {item['amount']:.0f}"] for item in result["investment_plan"]]
    story.append(_table([["Category", "Share", "Amount"]] + investment_rows, header=True))

    doc.build(story)
    buffer.seek(0)
    return buffer


def _table(rows: list[list[str]], header: bool = False):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(rows, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dce3ec")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf1fb") if header else colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold" if header else "Helvetica"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    table.setStyle(TableStyle(style))
    return table

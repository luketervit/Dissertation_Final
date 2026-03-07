from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit

OUT = "/Users/luketervit/Desktop/dissertation_final/startup_pricing_sheet_2026-03-04.pdf"

PAGE_W, PAGE_H = letter
MARGIN = 0.55 * inch
WIDTH = PAGE_W - 2 * MARGIN

TITLE_BG = colors.HexColor("#102A43")
ACCENT = colors.HexColor("#F0B429")
SECTION_BG = colors.HexColor("#F7FAFC")
TEXT = colors.HexColor("#1F2933")
SUBTLE = colors.HexColor("#52606D")


def draw_wrapped(c, text, x, y, font="Helvetica", size=9.8, color=TEXT, max_width=WIDTH, leading=12):
    c.setFont(font, size)
    c.setFillColor(color)
    lines = simpleSplit(text, font, size, max_width)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def section_header(c, y, title):
    h = 16
    c.setFillColor(SECTION_BG)
    c.roundRect(MARGIN, y - h + 3, WIDTH, h, 4, stroke=0, fill=1)
    c.setFillColor(ACCENT)
    c.rect(MARGIN, y - h + 3, 6, h, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#243B53"))
    c.setFont("Helvetica-Bold", 10.8)
    c.drawString(MARGIN + 12, y - 8, title)
    return y - h - 4


def bullet(c, y, text, indent=0):
    x = MARGIN + 12 + indent
    c.setFillColor(TEXT)
    c.setFont("Helvetica", 9.8)
    c.drawString(x, y, u"\u2022")
    return draw_wrapped(c, text, x + 10, y, size=9.8, max_width=WIDTH - (x - MARGIN) - 12, leading=11.5)


def main():
    c = canvas.Canvas(OUT, pagesize=letter)

    # Explicit white page background for consistent rendering.
    c.setFillColor(colors.white)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)

    # Header block
    c.setFillColor(TITLE_BG)
    c.rect(0, PAGE_H - 1.5 * inch, PAGE_W, 1.5 * inch, stroke=0, fill=1)

    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(MARGIN, PAGE_H - 0.78 * inch, "Political Messaging Forecasting")
    c.setFont("Helvetica", 11)
    c.drawString(MARGIN, PAGE_H - 1.08 * inch, "Pricing Sheet | March 4, 2026 | v1 (Pilot to Validated)")

    # Accent line
    c.setFillColor(ACCENT)
    c.rect(MARGIN, PAGE_H - 1.23 * inch, 3.3 * inch, 0.045 * inch, stroke=0, fill=1)

    y = PAGE_H - 1.75 * inch

    y = section_header(c, y, "Who This Is For")
    y = draw_wrapped(
        c,
        "Campaign teams and political advertisers that want pre-post message risk forecasting before publishing: backlash risk, sentiment mix, aggression/offensive likelihood, and engagement direction.",
        MARGIN + 12,
        y,
        size=9.8,
        max_width=WIDTH - 20,
    ) - 6

    y = section_header(c, y, "1) Design Partner Pricing (Pre-Validation)")
    c.setFont("Helvetica-Bold", 10.2)
    c.setFillColor(colors.HexColor("#243B53"))
    c.drawString(MARGIN + 12, y, "Pilot Lite")
    y -= 12
    y = bullet(c, y, "$2,000/month + $1,000 setup | 6-8 weeks")
    y = bullet(c, y, "1 account, limited simulation volume, baseline scorecards")
    y -= 4

    c.setFont("Helvetica-Bold", 10.2)
    c.setFillColor(colors.HexColor("#243B53"))
    c.drawString(MARGIN + 12, y, "Pilot Pro")
    y -= 12
    y = bullet(c, y, "$5,000-$8,000/month + $2,000 setup | 6-10 weeks")
    y = bullet(c, y, "1-2 accounts, higher simulation volume, weekly review, priority support")
    y -= 4

    c.setFont("Helvetica-Bold", 10.2)
    c.setFillColor(colors.HexColor("#243B53"))
    c.drawString(MARGIN + 12, y, "Design Partner Terms")
    y -= 12
    y = bullet(c, y, "Discount tied to case-study/reference rights")
    y = bullet(c, y, "Predictions logged before posting (pre-registered)")
    y = bullet(c, y, "Post-hoc scorecard at pilot close; monthly upfront billing")
    y -= 2

    y = section_header(c, y, "2) Validated Pricing (State Politics)")
    y = bullet(c, y, "State House / Small District: $1,500-$3,500 per month (or $9,000-$20,000 per cycle)")
    y = bullet(c, y, "State Senate / Competitive District: $3,500-$8,000 per month (or $20,000-$50,000 per cycle)")
    y = bullet(c, y, "Statewide / Caucus / Party Org: $12,000-$30,000 per month (enterprise support/SLA)")
    y -= 2

    y = section_header(c, y, "3) Pricing Logic")
    y = bullet(c, y, "Self-serve target: 1-3% of campaign budget")
    y = bullet(c, y, "Managed/embedded target: 3-8% of campaign budget")
    y = bullet(c, y, "No free pilots: paid pilots improve signal quality and commitment")
    y -= 2

    y = section_header(c, y, "4) Commercial Guardrails")
    y = bullet(c, y, "Scope controls by account count, simulation volume, and support tier")
    y = bullet(c, y, "Usage-based overage add-ons above plan limits")
    y = bullet(c, y, "Renewal tied to measured decision-lift vs baseline workflow")
    y -= 2

    y = section_header(c, y, "5) Validation Gate Before Full Rollout")
    y = bullet(c, y, "Multi-account forward validation")
    y = bullet(c, y, "Multi-seed robustness reporting (mean/std)")
    y = bullet(c, y, "Prospective decision-lift evidence in live pilots")

    # Footer
    c.setFillColor(SUBTLE)
    c.setFont("Helvetica-Oblique", 8.6)
    c.drawString(MARGIN, 0.42 * inch, "Package: Pilot SOW + MSA + data-use addendum + scorecard template")

    c.save()


if __name__ == "__main__":
    main()

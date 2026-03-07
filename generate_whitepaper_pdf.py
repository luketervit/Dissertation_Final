#!/usr/bin/env python3
"""Generate a one-page technical whitepaper PDF."""

from fpdf import FPDF


class WhitepaperPDF(FPDF):
    def __init__(self) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=False)

    def section(self, title: str) -> None:
        self.set_font("Helvetica", "B", 9.8)
        self.set_text_color(31, 60, 90)
        self.set_x(self.l_margin)
        self.multi_cell(0, 4.6, title)
        self.set_draw_color(31, 60, 90)
        self.set_line_width(0.2)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(1.2)

    def body(self, text: str) -> None:
        self.set_font("Helvetica", "", 8.45)
        self.set_text_color(25, 25, 25)
        self.set_x(self.l_margin)
        self.multi_cell(0, 3.95, text)

    def bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 8.45)
        self.set_text_color(25, 25, 25)
        self.set_x(self.l_margin)
        self.multi_cell(0, 3.9, f"- {text}")


def draw_results_table(pdf: WhitepaperPDF) -> None:
    rows = [
        ("1800961685949198714", "97.93"),
        ("1800996791078236518", "94.83"),
        ("1801248635654390240", "94.58"),
        ("1801290362700394984", "90.57"),
        ("1801402597707767909", "94.79"),
        ("Mean ± SD", "94.54 ± 2.34"),
    ]

    table_x = pdf.l_margin + 4
    col1 = 78
    col2 = 34
    row_h = 3.85
    y0 = pdf.get_y()

    pdf.set_font("Helvetica", "B", 8.2)
    pdf.set_fill_color(238, 242, 247)
    pdf.rect(table_x, y0, col1, row_h + 0.1, "F")
    pdf.rect(table_x + col1, y0, col2, row_h + 0.1, "F")
    pdf.set_xy(table_x, y0)
    pdf.cell(col1, row_h, "Holdout Root Tweet ID", border=1, align="L")
    pdf.cell(col2, row_h, "Accuracy (%)", border=1, align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 8.2)
    for i, (tweet_id, acc) in enumerate(rows):
        pdf.set_x(table_x)
        if i == len(rows) - 1:
            pdf.set_font("Helvetica", "B", 8.2)
        pdf.cell(col1, row_h, tweet_id, border=1, align="L")
        pdf.cell(col2, row_h, acc, border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        if i == len(rows) - 1:
            pdf.set_font("Helvetica", "", 8.2)


def main() -> None:
    pdf = WhitepaperPDF()
    pdf.set_margins(14, 10, 14)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 14.5)
    pdf.set_text_color(31, 60, 90)
    pdf.cell(
        0,
        6.0,
        "Root-Tweet Forward Prediction of Political Reply Dynamics",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.set_font("Helvetica", "", 9.0)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 4.2, "Technical Whitepaper", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8.0)
    pdf.set_text_color(95, 95, 95)
    pdf.cell(
        0,
        3.8,
        "Luke Tervit | University of Edinburgh | March 2026",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(1.1)

    pdf.section("Objective")
    pdf.body(
        "This project tests whether downstream political discourse can be predicted from the root "
        "tweet alone. The target is distribution-level forecasting across political alignment, "
        "emotion, sentiment, and aggression, rather than exact text reconstruction."
    )

    pdf.section("System Summary")
    pdf.bullet(
        "Agent personas are built from historical Twitter data using five classifiers: political "
        "leaning, emotion, sentiment, hate, and offensive language."
    )
    pdf.bullet(
        "Each holdout simulation starts with only the root tweet in metadata (0-shot mode; no "
        "target-thread replies used as generation input)."
    )
    pdf.bullet(
        "A Mesa ABM runs 10 synchronous rounds. Agent reply likelihood is behaviorally weighted by aggression."
    )
    pdf.bullet(
        "Validation reclassifies real and simulated tweets with the same five-model stack and scores "
        "JSD-based similarity plus aggression-mean alignment."
    )

    pdf.section("Forward Holdout")
    pdf.body("Target account: Sen. Marsha Blackburn (user_id 278145569).")
    pdf.body(
        "Protocol: 8 chronological root tweets found; first 3 used for responder-profile construction; "
        "last 5 held out for strict forward prediction."
    )
    pdf.body(
        "Per-run prediction input: root tweet text only "
        "(source=forward_holdout_zero_shot, use_few_shot=false); 200 synthetic agents sampled from "
        "124 matched historical responders."
    )
    draw_results_table(pdf)
    pdf.ln(0.7)
    pdf.body(
        "Component means across the 5 predicted tweets: political similarity 96.45%, emotion similarity "
        "91.78%, sentiment similarity 97.39%, aggression similarity 90.17%."
    )

    pdf.section("Interpretation and Limits")
    pdf.body(
        "Using only the root tweet and history-derived responder personas, the system predicted the next "
        "five held-out conversations with 94.54% mean weighted accuracy. This complements broader "
        "project evidence of 92.3% mean accuracy across 100 held-out threads."
    )
    pdf.bullet("The 94.54% score reflects distributional fidelity, not exact tweet-by-tweet text matching.")
    pdf.bullet("Reply-volume calibration was not directly optimized by this metric and should be a next target.")
    pdf.bullet("External validity remains bounded to USC X 24 political discourse and tested model settings.")

    pdf.ln(0.4)
    pdf.set_font("Helvetica", "B", 8.4)
    pdf.set_text_color(20, 20, 20)
    pdf.multi_cell(
        0,
        3.95,
        "Bottom line: historical tweets from a specific political account can be transformed into a "
        "predictive behavioral population that anticipates likely reaction patterns to that account's "
        "next root tweets before they are observed.",
    )

    pdf.output("whitepaper.pdf")
    print("Generated whitepaper.pdf")


if __name__ == "__main__":
    main()

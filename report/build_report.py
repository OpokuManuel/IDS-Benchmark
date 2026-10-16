"""Build the 2-page PDF write-up from results/. Usage: python report/build_report.py --author "Your Name" """
import argparse, sys
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

#File paths
ROOT = Path(__file__).resolve().parents[1]; RES = ROOT / "results"; FIG = RES / "figures"
ap = argparse.ArgumentParser(); ap.add_argument("--author", default="Opoku Emmanuel Atta"); args = ap.parse_args()

#Loading CSV
S = pd.read_csv(RES / "summary_by_protocol.csv").set_index(["protocol", "model"])

#Styled Text Objects used
B = ParagraphStyle("b", fontName="Times-Roman", fontSize=9.8, leading=12.2, alignment=4, spaceAfter=3)
H = ParagraphStyle("h", fontName="Times-Bold", fontSize=10.8, leading=13, spaceBefore=6, spaceAfter=2)
T = ParagraphStyle("t", fontName="Times-Bold", fontSize=14, leading=16.5, alignment=1, spaceAfter=1)
A = ParagraphStyle("a", fontName="Times-Italic", fontSize=8.8, leading=10.5, alignment=1, spaceAfter=4)
C = ParagraphStyle("c", fontName="Times-Italic", fontSize=8.3, leading=10, alignment=4, spaceAfter=4)
P = lambda t, s=B: Paragraph(t, s)

def row(proto, label):
    r = []
    for m in ["logreg", "random_forest", "grad_boosting"]:
        x = S.loc[(proto, m)]; r.append(f"{x.macro_f1:.2f} / {x.fpr_at_tpr95:.2f}")
    return [label] + r

#Table Summary
tab = Table([["Protocol", "Logistic reg.", "Random forest", "Grad. boosting"],
             row("P0_official", "P0 official split (raw)"), row("P1_random_clean", "P1 random, deduplicated"),
             row("P2_grouped", "P2 grouped (leakage-safe)")], colWidths=[52*mm, 31*mm, 31*mm, 31*mm])
tab.setStyle(TableStyle([("FONT", (0, 0), (-1, 0), "Times-Bold", 9), ("FONT", (0, 1), (-1, -1), "Times-Roman", 9),
    ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black), ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.black),
    ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black), ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e8f1f4")),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))

#Image Helper
def img(name, w):
    from PIL import Image as PI
    iw, ih = PI.open(FIG / name).size; return Image(str(FIG / name), width=w, height=w * ih / iw)

#Report Document
doc = SimpleDocTemplate(str(ROOT / "report" / "writeup.pdf"), pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                        topMargin=13*mm, bottomMargin=12*mm, title="Leakage-safe ML intrusion detection benchmark", author=args.author)
g = lambda p, m, k: S.loc[(p, m)][k]
st = [P("How much do standard evaluations overstate ML intrusion detection? A leakage-aware benchmark on UNSW-NB15", T),
      P(f"{args.author} &middot; code, data checksums and all results: see the accompanying repository", A),
      P("<b>Abstract.</b> Published intrusion-detection results are often measured on splits that leak near-identical flows between training and test data. "
        "I audit the official UNSW-NB15 partitions, clean them, and compare three baselines (logistic regression, random forest, gradient boosting) under three protocols: "
        "the official split, a random split of the cleaned data, and a split that keeps clusters of near-duplicate flows on one side. "
        f"Moving from the random to the grouped protocol lowers the best model's macro-F1 from {g('P1_random_clean','grad_boosting','macro_f1'):.2f} to {g('P2_grouped','grad_boosting','macro_f1'):.2f} "
        f"and raises its false-positive rate at 95% detection from {g('P1_random_clean','grad_boosting','fpr_at_tpr95'):.2f} to {g('P2_grouped','grad_boosting','fpr_at_tpr95'):.2f}. "
        "Many remaining errors come from contradictory labels in the data rather than from the models.", B),
      P("1. Data and problems found", H),
      P("I used the official training (175,341 flows) and testing (82,332) CSVs, pinned by SHA-256 (obtained from GitHub mirrors, since UNSW's server was unreachable; the training file was byte-identical across two mirrors). "
        "Three problems matter. <b>Duplicates:</b> 42% of training and 34% of test rows are exact feature duplicates. <b>Leakage:</b> 8,541 test rows (10.4%) have an exact twin in the training set and 20.4% a near-twin "
        "(nearest-neighbour distance &lt; 0.1 in standardised log-feature space). <b>Contradictory labels:</b> 414 feature vectors carry both benign and malicious labels (dropped) and 2,031 more carry several attack categories "
        "(kept once, majority category); 77% of Analysis, 85% of Backdoor and 77% of DoS rows are in such vectors. Pooling and cleaning gives 153,270 flows. "
        "The class balance changes sharply (Generic: 58,871 raw rows, 7,284 clean), which is why raw-split scores are not comparable with cleaned-data scores.", B),
      P("2. Method", H),
      P("<b>Protocols.</b> P0: the official split as published. P1: stratified random 5-fold on the cleaned pool. P2: stratified <i>group</i> 5-fold, where flows sharing protocol, service, state, both TTLs and a coarse log-scale packet/byte profile form one group "
        "(no labels used). The official files have no timestamps or IP addresses, so a true time- or session-based split is impossible; P2 is a near-duplicate proxy and a deliberately harsh test. "
        "It reduces near-twin leakage from 15.7% (P1) to 0.7% with no shared groups (verified by unit tests). "
        "<b>Models and metrics.</b> Hyper-parameters were chosen on an inner validation split of one training fold, never on test data. I report per-class precision/recall, macro-F1, binary AUC, and the false-positive rate (FPR) at 95% and 99% detection; "
        "the 95% threshold is also chosen on validation data only and then measured on test.", B),
      KeepTogether([img("fig2_leakage_audit.png", 118*mm), P("Figure 1. Share of test flows with a near-twin in training (left) and median nearest-neighbour distance (right). Grouping removes almost all near-duplicate leakage.", C)]),
      P("3. Results", H),
      tab,
      P("Table 1. Macro-F1 / FPR at 95% detection (pooled out-of-fold predictions, seed 42). Fold-to-fold macro-F1 sd under P2 is about 0.025.", C),
      img("fig1_protocol_comparison.png", 172*mm),
      P("Figure 2. The leakage-safe protocol (P2) is lowest for every model. The official split (P0) is <i>not</i> simply inflated relative to P1: it also carries its own distribution shift. Model ranking is stable across protocols.", C),
      P(f"With the transferred threshold, gradient boosting under P2 detects {g('P2_grouped','grad_boosting','val_thr_tpr')*100:.1f}% of attacks at a {g('P2_grouped','grad_boosting','val_thr_fpr')*100:.1f}% false-positive rate, too noisy to deploy alone. "
        "Per class (gradient boosting, P2), Normal (recall 0.90) and Exploits (0.91) are strong, while Generic (0.07), Reconnaissance (0.15) and DoS (0.25) collapse, although they score 0.97, 0.81 and 0.08 under P0. "
        "Stricter grouping lowers scores monotonically (gradient boosting macro-F1 0.46, 0.43, 0.40 at 4, 2 and 1 size-bins per octave). "
        "Removing TTL features from the random forest raises FPR at 95% detection from 0.22 to 0.37; removing TCP sequence numbers or protocol/service/state changes it little.", B),
      P("4. Error analysis", H),
      P("81% of Generic is predicted as Fuzzers and 64% of DoS as Exploits; the latter matches the label-conflict finding, so part of this error is a data ceiling for flow-level features. "
        "At 95% detection nearly all missed attacks are Fuzzers (15.8% missed; every other class under 3%). False alarms concentrate on benign flows with no identified service (74% of all false alarms; "
        "37% of benign HTTP flows are flagged versus 0.2% of DNS). A spurious correlation is visible directly: 93% of UDP/DNS flows in the official training file are attacks, and logistic regression, "
        "when a 10,255-flow benign DNS cluster was held out, labelled 10,250 of those flows Exploits. This is the spurious-correlation pitfall described by Arp et al. (2022).", B),
      KeepTogether([Table([[img("fig4_confusion_P2_best.png", 80*mm), img("fig6_sensitivity.png", 92*mm)]], colWidths=[84*mm, 94*mm]),
        P("Figure 3. Left: gradient boosting confusion matrix under P2 (row-normalised %). Right: scores fall as near-duplicate grouping gets stricter (sensitivity analysis).", C)]),
      P("5. Limitations", H),
      P("No true temporal or session split was possible with these files; grouping is a heuristic proxy. All P2 numbers use one fold-assignment seed (fold-level spread is reported instead; AUC ranges 0.59-0.96 for logistic regression). "
        "Majority-category resolution of conflicting vectors shrinks rare classes. Baselines are deliberately simple and lightly tuned. "
        "Next step: repeat with CIC-IDS2017 and a genuine day/time split.", B),
      P("<b>References.</b> Moustafa &amp; Slay (2015), UNSW-NB15, MilCIS. Arp et al. (2022), Dos and Don'ts of ML in Computer Security, USENIX Security. "
        "Engelen et al. (2021), Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study, IEEE S&amp;P Workshops. Zoghi &amp; Serpen (2021), arXiv:2101.05067.", C)]
doc.build(st)
from pypdf import PdfReader; n = len(PdfReader(str(ROOT / "report" / "writeup.pdf")).pages); print("pages:", n)

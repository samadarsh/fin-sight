#!/usr/bin/env python3
"""Build eval/questions.jsonl from the two annual reports it is written against.

    python eval/build_questions.py data/documents eval/questions.jsonl

Each question has a regex (`where`) for the passage that answers it. Its expected pages are every
page whose text matches, found here rather than typed by hand. Every key fact must appear on at
least one of those pages, or the build fails. Page text comes from PyMuPDF, exactly as ingestion
reads it, so page numbers match what FinSight stores.
"""

# ruff: noqa: E501  (question text and regexes read best on one line)
import json
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

DOCS = Path(sys.argv[1])
TCS_FILE, IOC_FILE = "annual-report-2025-2026.pdf", "SingleAnnualReport202425.pdf"


def _read(name):
    with fitz.open(DOCS / name) as doc:
        return {i + 1: " ".join(page.get_text("text").split()) for i, page in enumerate(doc)}


PAGES = {TCS_FILE: _read(TCS_FILE), IOC_FILE: _read(IOC_FILE)}


def norm(t):  # same normalisation as src/finsight/evaluation.py
    t = t.lower().replace("₹", "")
    t = re.sub(r"(?<=\d),(?=\d)", "", t)
    return re.sub(r"\s+", " ", t)


def _digits(t):  # 8,45,513 (Indian) and 845,513 (international) both become 845513
    return re.sub(r"(?<=\d),(?=\d)", "", t)


def pages_for(file, where):
    # Same for the pattern, except commas inside regex quantifiers like {0,30}.
    where = re.sub(r"(?<=\d),(?=\d)(?![^{]*\})", "", where)
    return [n for n, t in PAGES[file].items() if re.search(where, _digits(t))]


TCS, IOC = "TCS", "IOC"
Q = [
    # ---- TCS, Integrated Annual Report 2025-26 ----
    (
        "tcs-revenue",
        TCS,
        "What was TCS's consolidated revenue from operations in FY 2026?",
        r"2,67,021|267,021",
        ["267021"],
    ),
    (
        "tcs-revenue-growth",
        TCS,
        "By what percentage did TCS's revenue from operations grow in FY 2026?",
        r"higher by 4\.6%|growth of 4\.6%|Revenue Growth 4\.6%|267,021 100\.0 4\.6",
        ["4.6%"],
    ),
    (
        "tcs-operating-margin",
        TCS,
        "What was TCS's operating margin in FY 2026?",
        r"[Oo]perating [Mm]argin (of |Highest in last 4 years )?25(\.0)?%|operating margin of 25\.0%|EBIT margins reached 25\.0%",
        [["25.0%", "25%"]],
    ),
    (
        "tcs-margin-drivers",
        TCS,
        "What helped TCS expand its operating margin in FY 2026, and what offset it?",
        r"rebalancing of pyramid",
        [["pyramid", "business mix", "productivity", "realisation", "currency"]],
    ),
    ("tcs-net-income", TCS, "What was TCS's net income in FY 2026?", r"52,820", ["52820"]),
    ("tcs-eps", TCS, "What was TCS's earnings per share in FY 2026?", r"145\.99", ["145.99"]),
    (
        "tcs-dividend",
        TCS,
        "What was TCS's total dividend per share for FY 2026?",
        r"total dividend is .{0,3}110|for a total of .{0,3}110 per share",
        ["110"],
    ),
    (
        "tcs-tcv",
        TCS,
        "What total contract value (TCV) did TCS report for FY 2026?",
        r"Total Contract Value( \(TCV\))? \$40\.7 billion|total contract value exceeded US\$ 40 bn|order book of US\$ 40\.7 billion",
        [["40.7", "40 bn", "40 billion"]],
    ),
    (
        "tcs-ai-revenue",
        TCS,
        "What is TCS's annualised AI revenue?",
        r"Annualised AI Revenue[^.]{0,60}\$2\.3 billion|2\.3 billion annualised AI revenue|annualised revenues of US\$ 2\.3 billion in AI",
        ["2.3"],
    ),
    (
        "tcs-attrition",
        TCS,
        "What was TCS's voluntary attrition rate in IT services for FY 2026?",
        r"[Aa]ttrition[^.]{0,30}13\.7%",
        ["13.7%"],
    ),
    (
        "tcs-workforce",
        TCS,
        "Roughly how many employees does TCS have?",
        r"workforce of over 580,000|Total employees \(D \+ E\) 6,?17,?437",
        [["580000", "5.8 lakh", "580k", "617437", "6.17 lakh"]],
    ),
    (
        "tcs-countries",
        TCS,
        "Across how many countries is TCS's workforce spread?",
        r"580,000 consultants spread across 56 countries",
        ["56"],
    ),
    (
        "tcs-ceo",
        TCS,
        "Who is the Chief Executive Officer and Managing Director of TCS?",
        r"Krithivasan Chief Executive Officer",
        ["Krithivasan"],
    ),
    (
        "tcs-chairman",
        TCS,
        "Who is the Chairman of TCS?",
        r"Chandrasekaran Chairman",
        ["Chandrasekaran"],
    ),
    (
        "tcs-auditor",
        TCS,
        "Which firm are TCS's statutory auditors?",
        r"B S R & Co\. LLP, the Statutory auditors",
        [["B S R", "BSR"]],
    ),
    (
        "tcs-patents",
        TCS,
        "How many patents has TCS filed and been granted cumulatively as of March 31, 2026?",
        r"9,596",
        ["9596", "5500"],
    ),
    (
        "tcs-csr",
        TCS,
        "How much did TCS spend on CSR globally in FY 2026?",
        r"Global CSR Spend .{0,3}1,153",
        ["1153"],
    ),
    (
        "tcs-cyber-risk",
        TCS,
        "What cyber security risks does TCS highlight, and how does it mitigate them?",
        r"Cyber Attacks \(R & O\)",
        [["ISO 27001", "security operations", "AI/ML"]],
    ),
    (
        "tcs-americas-revenue",
        TCS,
        "How much revenue did TCS earn from the Americas in FY 2026?",
        r"Americas 134,998",
        ["134998"],
    ),
    (
        "tcs-india-revenue",
        TCS,
        "How did TCS's revenue from India change in FY 2026 compared with FY 2025?",
        r"India 15,775 22,060",
        ["15775", "22060"],
    ),
    (
        "tcs-roe",
        TCS,
        "What was TCS's return on equity in FY 2026?",
        r"Return on Equity 51\.4%|FY 2026 51\.4% Best in class RoE",
        ["51.4%"],
    ),
    (
        "tcs-renewable",
        TCS,
        "What share of TCS's total energy consumption came from renewable energy?",
        r"Renewable energy as % of total energy consumed 79%",
        ["79%"],
    ),
    (
        "tcs-ai-skills",
        TCS,
        "How many TCS employees have advanced AI skills?",
        r"270,000 employees now have advanced AI skills|270K Employees with higher order skills|(more than|Over) 270,000 (associates|Higher Order)",
        [["270000", "270k"]],
    ),
    # ---- IndianOil, Integrated Annual Report 2024-25 ----
    (
        "ioc-revenue",
        IOC,
        "What was IndianOil's standalone revenue from operations in 2024-25?",
        r"845,?513",
        ["845513"],
    ),
    (
        "ioc-pat",
        IOC,
        "What was IndianOil's standalone profit after tax in 2024-25?",
        r"12,?962",
        ["12962"],
    ),
    (
        "ioc-pat-change",
        IOC,
        "How did IndianOil's standalone profit after tax in 2024-25 compare with 2023-24?",
        r"12962 39619|12,962 Crore as compared to .{0,3}39,619|Profit After Tax 12,962 39,619",
        ["12962", "39619"],
    ),
    (
        "ioc-refining-margins",
        IOC,
        "According to IndianOil, why did global refining margins decline in 2024?",
        r"global refining margins continued their downward trend",
        [["refining capacity", "slowdown in demand", "outpacing demand", "supply"]],
    ),
    (
        "ioc-capacity",
        IOC,
        "What is IndianOil's group refining capacity?",
        r"80\.75 ?MMTPA",
        ["80.75"],
    ),
    (
        "ioc-capacity-target",
        IOC,
        "To what capacity does IndianOil plan to expand its refining, and by when?",
        r"98\.4 ?MMTPA|98\.4 Million metric tonnes per annum",
        ["98.4"],
    ),
    (
        "ioc-refinery-throughput",
        IOC,
        "What was IndianOil's refinery throughput in FY 2025?",
        r"Refinery throughput \(MMT\) FY 25 71\.56|processed 71\.56 MMT of crude",
        ["71.56"],
    ),
    (
        "ioc-pipeline-throughput",
        IOC,
        "What pipeline throughput milestone did IndianOil reach in 2024-25?",
        r"100\.5 ?MMT",
        ["100.5"],
    ),
    (
        "ioc-pipeline-length",
        IOC,
        "How long is IndianOil's pipeline network?",
        r"20,000 ?Km",
        ["20000"],
    ),
    (
        "ioc-retail-outlets",
        IOC,
        "How many retail outlets did IndianOil have at the end of 2024-25?",
        r"40,221 ROs|40,200\+ Retail Outlets|total to 40,221",
        [["40221", "40200"]],
    ),
    (
        "ioc-new-outlets",
        IOC,
        "How many new retail outlets did IndianOil add in 2024-25?",
        r"2,823 new Retail Outlets",
        ["2823"],
    ),
    ("ioc-chairman", IOC, "Who is the Chairman of IndianOil?", r"A S Sahney Chairman", ["Sahney"]),
    (
        "ioc-dividend",
        IOC,
        "What dividend did IndianOil's board recommend for 2024-25, and what payout ratio is that?",
        r"payout ratio of 32%|32% of the PAT|Dividend payout ratio for financial year 2024-25",
        ["32%"],
    ),
    (
        "ioc-net-zero",
        IOC,
        "By what year does IndianOil target net-zero operational emissions?",
        r"Net-Zero operational emissions( \(Scope 1 and 2\))? by 2046",
        ["2046"],
    ),
    (
        "ioc-employees",
        IOC,
        "How many employees did IndianOil have as of March 31, 2025?",
        r"29,941",
        ["29941"],
    ),
    (
        "ioc-fortune",
        IOC,
        "What was IndianOil's rank in the Fortune Global 500 list of 2025?",
        r"127(th)? (position )?in the (prestigious )?(2025 )?Fortune Global 500|ranked 127th",
        ["127"],
    ),
    (
        "ioc-refining-share",
        IOC,
        "What is IndianOil's share of India's refining market?",
        r"largest oil refiner with ~31% market share",
        ["31%"],
    ),
]

COMPARE = [
    (
        "cmp-revenue",
        "Compare the revenue from operations of TCS in FY 2026 and IndianOil in 2024-25.",
        # IndianOil reports standalone (8,45,513) and consolidated (8,59,363) revenue; either is right.
        [
            (TCS_FILE, r"2,67,021|267,021"),
            (IOC_FILE, r"845,?513|Revenue from Operations[^.]{0,80}859363"),
        ],
        ["267021", ["845513", "859363"]],
    ),
    (
        "cmp-profit",
        "Which earned more profit after tax: TCS in FY 2026 or IndianOil in 2024-25?",
        # Standalone (12,962) or consolidated (13,789) profit after tax.
        [(TCS_FILE, r"52,820"), (IOC_FILE, r"12,?962|Profit After Tax[^.]{0,20}13789")],
        ["52820", ["12962", "13789"]],
    ),
    (
        "cmp-workforce",
        "Compare the workforce size of TCS and IndianOil.",
        [
            (TCS_FILE, r"workforce of over 580,000|Total employees \(D \+ E\) 6,?17,?437"),
            (IOC_FILE, r"29,941"),
        ],
        [["580000", "5.8 lakh", "580k", "617437", "6.17 lakh"], "29941"],
    ),
]

TRAPS = [
    ("trap-tcs-2030", TCS, "What revenue does TCS forecast for FY 2030?"),
    ("trap-ioc-2030", IOC, "What was IndianOil's refinery throughput in 2029-30?"),
    ("trap-infosys", None, "What was Infosys's operating margin in FY 2026?"),
]


def facts_ok(facts, texts):
    blob = norm(" ".join(texts))
    missing = []
    for f in facts:
        opts = f if isinstance(f, list) else [f]
        if not any(norm(o) in blob for o in opts):
            missing.append(f)
    return missing


out, problems = [], []
for qid, co, question, where, facts in Q:
    file = TCS_FILE if co == TCS else IOC_FILE
    pages = pages_for(file, where)
    missing = facts_ok(facts, [PAGES[file][p] for p in pages]) if pages else facts
    if not pages or missing:
        problems.append(f"{qid}: pages={pages} missing facts={missing}")
    out.append(
        {
            "id": qid,
            "question": question,
            "company": co,
            "source_file": file,
            "pages": pages,
            "answer_contains": facts,
        }
    )
for qid, question, wheres, facts in COMPARE:
    expected, texts = [], []
    for file, where in wheres:
        pages = pages_for(file, where)
        if not pages:
            problems.append(f"{qid}: no pages in {file}")
        expected.append({"source_file": file, "pages": pages})
        texts += [PAGES[file][p] for p in pages]
    missing = facts_ok(facts, texts)
    if missing:
        problems.append(f"{qid}: missing facts={missing}")
    out.append(
        {
            "id": qid,
            "question": question,
            "compare": True,
            "companies": [TCS, IOC],
            "expected": expected,
            "answer_contains": facts,
        }
    )
for qid, co, question in TRAPS:
    item = {"id": qid, "question": question, "unanswerable": True}
    if co:
        item["company"] = co
    out.append(item)

for item in out:
    pages = item.get("pages") or [p for e in item.get("expected", []) for p in e["pages"]]
    print(f"{item['id']:<26} pages={pages[:10]}{' …' if len(pages) > 10 else ''}")
if problems:
    print("\nPROBLEMS:\n" + "\n".join(problems))
    sys.exit(1)
with open(sys.argv[2], "w", encoding="utf-8") as f:
    f.write("# FinSight evaluation set: 46 questions written from the real reports.\n")
    f.write(f"# Ingest {TCS_FILE} as company TCS and {IOC_FILE} as company IOC, unchanged.\n")
    f.write(
        "# Pages are where each answer appears (PyMuPDF numbering, 1 = first page of the file).\n"
    )
    for item in out:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
print(f"\nwrote {len(out)} questions")

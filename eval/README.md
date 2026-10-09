# Evaluating FinSight

A small, honest test of the whole pipeline: does it find the right page, cite it, and state the
right numbers? Run it whenever you change chunking, embeddings, `k` or the prompt, and compare.

## The included question set

`questions.jsonl` has 46 questions written from two real annual reports:

| Report | File name to ingest | Company |
|---|---|---|
| TCS Integrated Annual Report 2025-26 (360 pages) | `annual-report-2025-2026.pdf` | `TCS` |
| IndianOil Integrated Annual Report 2024-25 (473 pages) | `SingleAnnualReport202425.pdf` | `IOC` |

- 23 TCS and 17 IndianOil questions: headline figures, explanations ("what drove the margin
  change"), people, risks and operations
- 3 comparisons across both reports
- 3 trap questions the reports can't answer (a FY 2030 forecast, a 2029-30 figure, another
  company), to check FinSight declines instead of guessing

Expected pages aren't typed by hand. `build_questions.py` searches the PDFs for the passage that
answers each question, lists **every** page that states it, and fails if a key fact isn't on
those pages. To check or extend the set, edit the list in that script and rebuild:

```bash
python eval/build_questions.py data/documents eval/questions.jsonl
```

### Ingest the two reports

Use the file names and companies above exactly; the questions refer to them.

```bash
python scripts/ingest.py data/documents/annual-report-2025-2026.pdf --company TCS --doc-type annual_report --year 2026
python scripts/ingest.py data/documents/SingleAnnualReport202425.pdf --company IOC --doc-type annual_report --year 2025
```

### Baseline: retrieval only

Measured on October 9, 2026 with the default setup: local `BAAI/bge-small-en-v1.5` embeddings,
the default chunking (3,044 chunks: 1,315 TCS, 1,729 IndianOil); single-company questions are
filtered by company, and comparisons retrieve each company separately (4 chunks each), as a full
run does. Traps excluded, so 43 questions.

| k | Retrieval hit@k | Mean reciprocal rank |
|---|---|---|
| 5 | 83.7% (36 / 43) | 0.605 |
| 10 | 88.4% (38 / 43) | 0.611 |

Every miss was checked: none of its retrieved chunks contained the answer. Most misses
are on infographic highlight pages (TCS p.5, p.23) and dense number tables (TCS p.69, the
regional revenue split), where embedding-only search struggles; keyword (hybrid) search or
table-aware chunking are the obvious next experiments.

### Full run: Llama 3 (Ollama)

Run on October 10, 2026 with Llama 3 (8B) through Ollama, the same embeddings and k = 5. The
answers were then re-scored with `scripts/rescore.py` after the question set was corrected (see
below).

| Metric | Result |
|---|---|
| Questions | 46 |
| Retrieval hit@5 | 83.7% |
| Citation accuracy | 74.4% |
| Fact match (43 questions) | 67.4% |
| Refusal accuracy (3 unanswerable questions) | 100% |
| Answerable questions it declined | 10 |
| Median latency | 7.4 s |

Read by hand against the reports, 26 of the 43 answers are fully correct, 4 partly correct
(right figure with a wrong detail or a missing part), 9 decline and 4 are wrong. Of the 9
declines, 5 are retrieval misses and 4 had the answer in context; a larger model should recover
those. The 4 wrong answers: a figure from the wrong passage (TCS's 109 countries is CodeVita, not
its footprint), a chairman question answered without a name, a revenue comparison that quotes
both figures correctly and then calls the smaller one higher, and a workforce comparison built
on the wrong figure. Fact match counts the revenue comparison, so the automated figure is a
little generous.

The first scoring of this run (60.5% fact match, 65.1% citation accuracy) undercounted: some
answers were right but came from a page or figure the question set didn't list yet, such as
TCS's 617,437 employees (p.108) and IndianOil's consolidated revenue and profit (p.115). Those
were added to `build_questions.py`, and the saved answers re-scored without asking the LLM again.

Two caveats when comparing runs: retrieval is re-run on re-scoring, and a different machine can
order near-tied chunks slightly differently (here 4 questions moved by one or two ranks; the
number of hits was unchanged). And Gemini's free tier hit its daily limit after 10 questions, so
use Ollama or a paid key for a full run.

## Writing your own questions

To test other documents, pick 3 to 5 real filings (annual reports, earnings transcripts) and
ingest them as usual:

```bash
python scripts/ingest.py data/documents/<file>.pdf --company TCS --doc-type annual_report --year 2026
```

Copy `questions.example.jsonl` and replace the examples with your own.
Aim for 30 to 50 questions, written **before** you look at FinSight's answers. For each one,
open the PDF and note the page(s) that contain the answer.

| Field | Meaning |
|---|---|
| `id` | Short unique name |
| `question` | Asked exactly as a user would |
| `source_file` | PDF file name the answer is in |
| `pages` | Page number(s) holding the answer, counted like a PDF viewer does (1 = the file's first page), **not** the number printed on the page. Annual reports often differ by a few pages |
| `answer_contains` | Optional key facts the answer must state, e.g. `"21.4%"`. A list inside means any one of them: `["2,55,324", "255324"]` |
| `company` | Optional; limits retrieval to that company |
| `compare`, `companies` | For comparison questions |
| `expected` | Instead of `source_file` + `pages` when the answer spans files: `[{"source_file": "a.pdf", "pages": [3]}, {"source_file": "b.pdf", "pages": [9]}]` |
| `unanswerable` | `true` for a question the documents can't answer; no pages needed |

Mix question types: exact figures, explanations ("why did margins fall"), risks, and a few
comparisons. Include a couple whose answer is *not* in the documents to see how it refuses.

## Run it

```bash
python scripts/evaluate.py eval/questions.jsonl --retrieval-only   # free: retrieval only
python scripts/evaluate.py eval/questions.jsonl                    # full run with the LLM
```

Each run saves per-question results, a summary and a Markdown table in `eval/results/`.

After changing the question set, re-score a saved full run instead of paying for the LLM again.
It keeps the saved answers, re-runs retrieval and writes `*-rescored-*` files next to the run:

```bash
python scripts/rescore.py eval/results/<stamp>-full-results.jsonl eval/questions.jsonl
```

## What the numbers mean

- **Retrieval hit@k**: share of questions where at least one of the top `k` chunks came from an
  expected page. If this is low, no prompt can save the answer.
- **Mean reciprocal rank (MRR)**: how high the first correct chunk ranks (1.0 = always first).
- **Citation accuracy**: share of answers that cite at least one expected page.
- **Fact match**: share of answers (among questions with `answer_contains`) that state every key
  fact. Exact-string checking is strict, so read the misses in `*-results.jsonl` by hand.
- **Refusal accuracy**: share of trap questions (`"unanswerable": true`) it declined instead of
  answering. The report also counts answerable questions it declined, the cost of caution.
  Refusals are detected from the wording, so skim the answers.
- **Median latency**: seconds per question, end to end.

Report results as they are, with the setup: number of questions and documents, LLM, embeddings
and `k`. For example: *"40 questions on 4 annual reports, Gemini + local BGE embeddings, k=5:
retrieval hit@5 90%, citation accuracy 85%."*

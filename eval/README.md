# Evaluating FinSight

A small, honest test of the whole pipeline: does it find the right page, cite it, and state the
right numbers? Run it whenever you change chunking, embeddings, `k` or the prompt, and compare.

## 1. Pick documents and ingest them

Use 3 to 5 real filings (annual reports, earnings transcripts) and ingest them as usual:

```bash
python scripts/ingest.py data/documents/<file>.pdf --company TCS --doc-type annual_report --year 2026
```

## 2. Write the questions

Copy `questions.example.jsonl` to `questions.jsonl` and replace the examples with your own.
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

Mix question types: exact figures, explanations ("why did margins fall"), risks, and a few
comparisons. Include a couple whose answer is *not* in the documents to see how it refuses.

## 3. Run it

```bash
python scripts/evaluate.py eval/questions.jsonl --retrieval-only   # free: retrieval only
python scripts/evaluate.py eval/questions.jsonl                    # full run with the LLM
```

Each run saves per-question results, a summary and a Markdown table in `eval/results/`.

## What the numbers mean

- **Retrieval hit@k**: share of questions where at least one of the top `k` chunks came from an
  expected page. If this is low, no prompt can save the answer.
- **Mean reciprocal rank (MRR)**: how high the first correct chunk ranks (1.0 = always first).
- **Citation accuracy**: share of answers that cite at least one expected page.
- **Fact match**: share of answers (among questions with `answer_contains`) that state every key
  fact. Exact-string checking is strict, so read the misses in `*-results.jsonl` by hand.
- **Median latency**: seconds per question, end to end.

Report results as they are, with the setup: number of questions and documents, LLM, embeddings
and `k`. For example: *"40 questions on 4 annual reports, Gemini + local BGE embeddings, k=5:
retrieval hit@5 90%, citation accuracy 85%."*

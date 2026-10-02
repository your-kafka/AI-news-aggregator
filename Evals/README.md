# Retrieval evaluation

Measures how well a retriever finds the right documents in **this** corpus.

```bash
make eval                   # score every retriever (deterministic, free)
make eval-goldens           # check every golden's documents exist
make eval-show g=G001       # inspect one query: expected vs retrieved
make eval-llm               # add DeepEval's LLM-judged metrics
```

## What is here

| File | Purpose |
|---|---|
| `golden/retrieval_goldens.jsonl` | 50 questions, 80 graded relevance judgements |
| `metrics.py` | nDCG, MRR, Recall, Precision, hit rate. Pure functions |
| `retrievers.py` | Baselines behind one `search(query, k)` interface |
| `retrieval_eval.py` | The runner. Deterministic metrics + optional DeepEval |
| `validate_goldens.py` | Guards against ground truth that does not exist |

## How the golden dataset was built

Every question was written **against the stored corpus**, not from outside
knowledge. That matters more than it sounds: a question whose answer is not
in the database penalises a working retriever for failing to find a document
that does not exist, and the resulting score is meaningless.

The process:

1. Dump every Anthropic, YouTube, Hacker News and OpenAI document plus a
   sample of arXiv titles, and read them.
2. Write questions that the read text answers.
3. Write each `ideal_answer` from the stored body text, quoting its real
   figures - 50% of Barclays developers, 0.3% of job tasks, 11 days in Lean,
   $78,000, 826 parallel agents.
4. Grade each supporting document 3 (answers it), 2 (highly relevant) or
   1 (context).
5. Run `validate_goldens.py`, which fails if any referenced document is
   missing or has no body.

Hand-written and corpus-verified, rather than LLM-generated. Smaller, but
every judgement is defensible.

### Coverage

| | |
|---|---|
| Questions | 50 |
| Judgements | 80 |
| Distinct documents | 72 |
| Query types | factual 17, numeric 13, multi-doc 7, exploratory 7, entity 4, comparative 2 |
| Difficulty | easy 12, medium 30, hard 8 |
| Sources | openai 33, anthropic 14, hackernews 14, arxiv 12, youtube 7 |

`multi-doc` questions need several documents to answer fully, so a retriever
that finds only one scores partially. `exploratory` questions ("what work
exists on agent harnesses?") have many valid answers, which is what nDCG's
graded relevance is for.

## Baseline results

Measured on 716 documents. **This is the "before" number.**

| retriever | nDCG@10 | MRR@10 | P@10 | Recall@50 | Hit@10 |
|---|---|---|---|---|---|
| `title-only` | 0.1035 | 0.1200 | 0.1200 | 0.0967 | 0.1200 |
| `postgres-fulltext` (AND) | 0.3664 | 0.4050 | 0.3082 | 0.3533 | 0.4200 |
| `postgres-lexical-or` | **0.4779** | 0.4521 | 0.0940 | **0.9317** | 0.7800 |

Three things worth reading off that table:

**Titles alone are nearly useless** (nDCG 0.10). The body text we fetched in
enrichment is doing almost all the work - which retroactively justifies the
effort spent on `content.py`.

**AND semantics silently lose most queries.** `websearch_to_tsquery` requires
*every* content word to appear, so *"What percentage of Barclays developers
is expected to use Claude Code?"* matches **nothing** - the word "percentage"
never appears in the Barclays article, although "Barclays" alone matches it
instantly. Query construction matters as much as the index.

**Recall is nearly solved; ranking is the bottleneck.** The OR baseline finds
93% of relevant documents in its top 50 but only 9% of its top 10 are
relevant. It retrieves the right things and orders them badly.

That last point sets the agenda. There is little recall left to win, so the
remaining work is ordering - which is exactly what reciprocal rank fusion,
MIL pooling and cross-encoder re-ranking are for. Each should move nDCG@10
and precision@10 while leaving recall@50 roughly where it is.

## Two layers of metric, and why

**Deterministic** (`metrics.py`) come from the graded judgements. Free,
identical every run, so they can gate CI: a pull request that drops nDCG@10
by more than 2% should fail.

**LLM-judged** (DeepEval) answer a question the grades cannot: does the
retrieved *text* actually contain the answer, rather than merely coming from
the right document? Useful, but each test costs an API call and results vary
between runs, so these inform rather than gate.

```bash
uv sync --group evals          # deepeval is optional; it stays out of the image
make eval-llm
```

> `deepeval` lives in its own dependency group deliberately. It pulls a large
> dependency tree and must never reach the production container.

> **Caveat in the DeepEval wiring:** there is no generator yet, so
> `actual_output` is filled with the ideal answer. `ContextualPrecision`,
> `ContextualRecall` and `ContextualRelevancy` judge `expected_output`
> against `retrieval_context`, so this does not distort them - but it must be
> replaced with the real generated answer before any end-to-end metric
> (faithfulness, answer relevancy) means anything.

## Not yet wired into CI

CI starts an empty database, so the goldens have nothing to match against.
Gating CI on retrieval quality needs a small fixture corpus committed to the
repo and loaded before the eval runs. Until then, run `make eval` locally and
record the numbers here when they change.

## Adding a retriever

Anything with `name` and `search(query, k) -> list[Hit]` can be scored. Add
it to `retrievers.py`, append it to the list in `retrieval_eval.py`, and it
appears as a new row - which is how the ablation table gets built.

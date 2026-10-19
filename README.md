# RAG Assignment: Baseline Dense Retrieval vs. Cross-Encoder Re-Ranking

All code is implemented and executed in a single Jupyter Notebook: **[`rag_reranking.ipynb`](rag_reranking.ipynb)**.

This project contrasts two retrieval workflows over the exact same chunked corpus of **Regulation (EU) 2016/679 (General Data Protection Regulation - GDPR)**, sharing the same initial Top-10 dense candidates:

| | Pipeline 1 – Baseline | Pipeline 2 – Re-ranking |
| --- | --- | --- |
| **Stage 1** | Bi-encoder `sentence-transformers/all-MiniLM-L6-v2` + FAISS `IndexFlatIP` (cosine) → Top 10 | *same Top 10* |
| **Stage 2** | Takes dense ranks 1–3 directly | `cross-encoder/ms-marco-MiniLM-L-6-v2` scores all 10 (query, chunk) pairs → sort descending → Top 3 |
| **Answer** | Google Gemini (`gemini-3.1-flash-lite`), temperature 0, identical prompt → **Answer 1** | same → **Answer 2** |

---

## Target Document

**Regulation (EU) 2016/679 (GDPR)**, *Protection of natural persons with regard to the processing of personal data and on the free movement of such data* (27 April 2016, OJ L 119).
It is an 88-page normative legal regulation consisting of 173 recitals and 99 articles containing explicit obligations (`SHALL`, `SHALL NOT`), rights, definitions, conditional exceptions, and penalty thresholds.

A clean copy of the official EUR-Lex text is stored in `data/CELEX_32016R0679_EN_TXT.pdf`. The document is chunked into **444 sentence-aligned passages** (~140 tokens each) with tracked Article and Chapter headings.

---

## Key Results & Findings

- **Evaluation Suite:** 16 stress-test queries spanning 5 failure categories (keyword traps, role limits, exceptions, negations, and buried chunks).
- **Hit Rate Improvement:** Top-3 relevance hit rate increased from **43.8%** (Baseline: 7/16) to **68.8%** (Re-ranking: 11/16) — a **+25.0% absolute improvement**.
- **Re-Rank Rescues:** The Cross-Encoder successfully promoted critical operative clauses buried at ranks #4–#9 into rank #1 for multiple queries (e.g., Q01 Definitions, Q02 Subcontracting limits, Q07 Data protection by design/default, Q09 Subject access rights), turning complete LLM failures into grounded, accurate answers.
- **Latency Profiling (CPU):**
  - Dense retrieval (Bi-Encoder + FAISS): **20.4 ms** median.
  - Cross-Encoder re-ranking (10 pairs): **424.0 ms** median.
  - LLM generation (Gemini API): **~1,330 ms** median.
- **Scaling Bottleneck:** Exhaustive Cross-Encoder over all 444 chunks takes **18.2 seconds** per query (and projected **11.4 hours** on a 1M chunk corpus). The two-stage funnel ($N \to 10 \to 3$) maintains a constant $O(K)$ bounded cross-attention overhead.

Detailed analysis and discussion are presented in **[`report/report.md`](report/report.md)** (PDF version: **[`report/2025201026_Assignment1.pdf`](report/2025201026_Assignment1.pdf)**).

---

## Repository Layout

```text
rag_reranking.ipynb       # Complete end-to-end implementation and executed outputs
data/                     # Source PDF (GDPR official regulation)
results/                  # Generated experimental artifacts
  chunks.jsonl            # Extracted corpus: chunk_id, section, page numbers, text
  rankings_top10.csv      # Top-10 candidates per query: dense rank, cosine, CE score, re-rank rank
  rank_shift_summary.csv  # Rank shifts of critical chunks and Top-3 inclusion outcomes
  answers.json            # Generated Answer 1 vs Answer 2 for all 16 queries
  llm_cache.json          # Cached Gemini responses (enables reproduction without API quota)
  latency.csv             # Measured latency percentiles across retrieval and generation
  scaling.csv             # Complexity and runtime scaling benchmark
report/
  report.md               # Technical report (Markdown)
  2025201026_Assignment1.pdf # Technical report (Formatted PDF)
requirements.txt          # Python dependencies
```

---

## Setup and Reproduction

### 1. Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (CPU PyTorch + HuggingFace + FAISS + Notebook tools)
pip install -r requirements.txt
```

### 2. Configure Gemini API Key

The Gemini API key is loaded from a local `.env` file (git-ignored):

```bash
echo "GEMINI_API_KEY=your_gemini_api_key_here" > .env
```

### 3. Run Notebook

Execute the notebook end-to-end:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 rag_reranking.ipynb
```

*Note:* All LLM calls are cached in `results/llm_cache.json`. Re-running the notebook reproduces the committed results deterministically without expending API quota. To force re-generation of LLM answers, delete `results/llm_cache.json`.

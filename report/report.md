# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Course:** Large Language Models & Multimodal Applications (LMA)  
**Deliverable:** Technical Report (Concise 4-Page Submission)  
**Code Repository:** https://github.com/adityaND19/LMA-Assignment-1  
**Target Corpus:** Regulation (EU) 2016/679 (GDPR) — Official Journal L 119 (88 pages, 444 sentence-aligned chunks)  
**Models:** Bi-Encoder: `sentence-transformers/all-MiniLM-L6-v2` | Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2` | Generator: Google Gemini `gemini-3.1-flash-lite`

---

## 1. Architecture Comparison: Bi-Encoder vs. Cross-Encoder

The core architectural divergence between a Bi-Encoder (Pipeline 1) and a Cross-Encoder (Pipeline 2) lies in **how and when token interactions are computed**.

```
BI-ENCODER (Dual-Tower, Independent Embeddings):
  Query q   --> [ Transformer Enc ] --> u = MeanPool(H_q) ---\
                                                              Cosine Sim: (u . v) / (||u|| ||v||)
  Passage d --> [ Transformer Enc ] --> v = MeanPool(H_d) ---/
  * Offline: All passages d_i pre-embedded into vector space.
  * Query: Query encoded once. FAISS IndexFlatIP executes an exact linear scan O(N * d).

CROSS-ENCODER (Single-Tower, Joint Token Self-Attention):
  [CLS] + Query q + [SEP] + Passage d --> [ 6 Sequential Transformer Layers ] --> Scalar Score s(q,d)
  * Online: Full cross-attention across all tokens within every layer. No pre-computation possible.
```

### 1.1 Bi-Encoder (Independent Dual-Tower Representation)
* **Mechanism:** The query q and document chunk d are encoded through independent forward passes:
  `u = MeanPool(Encoder(q)),  v = MeanPool(Encoder(d))`
  Semantic similarity is computed via cosine similarity: `sim(q, d) = (u · v) / (||u||₂ · ||v||₂)`.
* **Advantages & Scalability:** Highly scalable. Document vectors `v_i` are computed once offline. During retrieval, the query is encoded once, and FAISS `IndexFlatIP` performs an exact brute-force inner product scan across all N vectors in `O(N · d)` time (measured at 20.4 ms for 444 chunks on CPU). In enterprise deployments, approximate nearest neighbor (ANN) indexes like HNSW or IVF reduce candidate lookup to logarithmic time `O(log N · d)`.
* **Bottlenecks:** Compressing 100–300 words into a single 384-dimensional vector creates an information bottleneck. The query cannot influence how passage tokens are weighted. Consequently, dense bi-encoders struggle with:
  1. *Lexical mismatch and terminology gaps* (e.g., query "subcontractor" vs. statutory text "another processor").
  2. *Negations and qualifiers* (e.g., "SHALL NOT engage" vectors map adjacent to "SHALL engage" because the lexical context is nearly identical).
  3. *Distractor recitals* that repeat high-level keywords without containing operative rules.

### 1.2 Cross-Encoder (Joint Token Self-Attention)
* **Mechanism:** The query and passage are concatenated into a single input sequence:
  `x = [CLS], q_1, ..., q_m, [SEP], d_1, ..., d_n, [SEP]`
  The full token sequence passes sequentially through all 6 transformer encoder layers. Within each layer, every token directly computes scaled dot-product self-attention with every other token:
  `Attention(Q, K, V) = softmax(Q Kᵀ / √dₖ) V`
  The pooled representation of the `[CLS]` token is linearly projected to output a scalar relevance score `s(q, d) ∈ ℝ`.
* **Advantages:** Captures fine-grained word-level dependencies, modifier bindings, and conditional clauses. It easily recognizes whether a prohibition binds a "processor" or a "controller", or if an action requires "prior written authorisation".

---

## 2. Qualitative & Rank Shift Analysis

### 2.1 Failure Mode Categories & Rank Shift Summary
We evaluated 16 targeted GDPR stress queries across 5 failure categories. The candidate pool consists of the Top-10 dense chunks retrieved in Stage 1.

| Query ID | Category | Query Text | Critical Chunk (Statutory Basis) | Dense Rank | CE Rank | Retrieval Outcome |
|:---:|:---|:---|:---|:---:|:---:|:---:|
| **Q01** | Keyword Trap | *Can companies share user passwords with third parties?* | Article 4 (`C187`) | #5 | **#1** | **RE-RANK RESCUE** |
| **Q02** | Role Limit | *Can a data processor hire another subcontractor without permission?* | Article 28(2) (`C259`) | #4 | **#1** | **RE-RANK RESCUE** |
| **Q03** | Exception | *Is processing sensitive data ever allowed without explicit consent?* | Article 9 (`C207`–`C210`) | >#10 | — | Both miss (ceiling) |
| **Q04** | Exception | *Can a controller transfer personal data outside the EU?* | Article 44 (`C311`) | #5 | #5 | Both miss (in Top 10) |
| **Q05** | Negation | *What is explicitly prohibited when processing children data online?* | Article 8 (`C205`) | #5 | #5 | Both miss (in Top 10) |
| **Q06** | Negation | *When is automated decision-making NOT allowed?* | Article 22 (`C246`) | #1 | #2 | Both hit |
| **Q07** | Buried Chunk | *What is data protection by design and by default?* | Article 25 (`C253`) | #5 | **#1** | **RE-RANK RESCUE** |
| **Q08** | Role Limit | *Who must appoint a Data Protection Officer?* | Article 37 (`C287`) | #1 | #1 | Both hit |
| **Q09** | Buried Chunk | *What information must be provided when a person requests access to their data?* | Article 15 (`C232`) | #4 | **#3** | **RE-RANK RESCUE** |
| **Q10** | Buried Chunk | *What penalties exist for serious GDPR breaches?* | Article 83 (`C415`–`C417`) | >#10 | — | Both miss (ceiling) |
| **Q11** | Keyword Trap | *How long can companies keep customer data?* | Article 5 (`C196`) | #3 | #3 | Both hit |
| **Q12** | Exception | *Can personal data be processed for scientific research without consent?* | Article 89 (`C429`) | >#10 | — | Both miss (ceiling) |
| **Q13** | Negation | *What rights can be restricted under national law?* | Article 23 (`C248`) | #1 | #1 | Both hit |
| **Q14** | Role Limit | *What breach notification obligations apply to processors?* | Article 33 (`C273`) | #1 | #2 | Both hit |
| **Q15** | Buried Chunk | *Under what conditions can an authority impose a temporary ban?* | Article 58 (`C354`) | #1 | #2 | Both hit |
| **Q16** | Keyword Trap | *Do users have the right to delete their account data?* | Article 17 (`C236`) | #3 | #2 | Both hit |

**Headline Results:**
* **Baseline Top-3 Hit Rate:** **7/16 (43.8%)** (Q06, Q08, Q11, Q13, Q14, Q15, Q16)
* **Re-Ranking Top-3 Hit Rate:** **11/16 (68.8%)** (the 7 baseline hits + 4 rescues: Q01, Q02, Q07, Q09)
* **Performance Gain:** **+4 queries (+25.0 percentage points)** improvement in Top-3 retrieval hit rate.

---

### 2.2 Deep-Dive: Failure Queries and Answer 1 vs. Answer 2 Comparison

#### Failure Query 1: Q02 — Role Limits & Subcontracting
* **Query:** *"Can a data processor hire another subcontractor without permission?"*
* **Rank Shift:** `C259` (Article 28(2) Processor Obligations): **Dense Rank #4 (score: 0.531) → Cross-Encoder Rank #1 (score: +0.59)**
  * *Baseline Top 3:* `C265` (Article 29: Processing under authority), `C127` (Recital 127: Supervisory cooperation), `C033` (Recital 33: Scientific research consent). All 3 missed the operative rule.
  * *Re-ranked Top 3:* `C259` (Article 28(2) Sub-processor authorization), `C114` (Recital 114), `C262` (Article 28(4) Sub-processor liability).
* **Comparison of LLM Generations:**
  * **Answer 1 (Baseline):** *"The provided context does not answer this question."*
    * *Analysis:* **Complete Omission / Refusal.** Because the operative rule sat at rank #4, the baseline prompt contained only adjacent articles and background recitals. Strict adherence to prompt constraints forced the model to state the context lacked the answer.
  * **Answer 2 (Re-ranked):** *"The processor SHALL NOT engage another processor without prior specific or general written authorisation of the controller [C259]. In the case of general written authorisation, the processor SHALL inform the controller of any intended changes concerning the addition or replacement of other processors, thereby giving the controller the opportunity to object to such changes [C259]."*
    * *Analysis:* **Legally Accurate.** Cross-Encoder promoted `C259` from #4 to #1, enabling the LLM to directly cite the statutory prohibition and conditional exception.

#### Failure Query 2: Q07 — Buried Chunk & Core Regulatory Definitions
* **Query:** *"What is data protection by design and by default?"*
* **Rank Shift:** `C253` (Article 25 Data Protection by Design and Default): **Dense Rank #5 (score: 0.589) → Cross-Encoder Rank #1 (score: +8.13)**
  * *Baseline Top 3:* `C085` (Recital 85: Producer encouragement), `C091` (Recital 91: DPIA considerations), `C323` (Article 47: Binding Corporate Rules).
  * *Re-ranked Top 3:* `C253` (Article 25), `C085` (Recital 85), `C084` (Recital 84: Internal compliance measures).
* **Comparison of LLM Generations:**
  * **Answer 1 (Baseline):** *"The provided context does not answer this question."*
    * *Analysis:* **Information Miss.** Dense retrieval matched general recitals discussing data protection principles, failing to deliver the operative definition.
  * **Answer 2 (Re-ranked):** *"Data protection by design and by default requires the controller to implement appropriate technical and organisational measures, such as pseudonymisation, to effectively integrate data-protection principles like data minimisation into the processing [C253]. This involves ensuring that, by default, only personal data necessary for each specific purpose of the processing are processed [C253]..."*
    * *Analysis:* **Complete & Precise.** Cross-Encoder promoted `C253` from #5 to #1, capturing the core statutory mandates of data protection by design (Article 25(1)) and by default (Article 25(2)).

#### Failure Query 3: Q09 — Buried Chunk & Disclosure Checklist
* **Query:** *"What information must be provided when a person requests access to their data?"*
* **Rank Shift:** `C232` (Article 15 Right of Access): **Dense Rank #4 (score: 0.598) → Cross-Encoder Rank #3 (score: +2.48)**
* **Comparison of LLM Generations:**
  * **Answer 1 (Baseline):** Stated the context did not specify the required content, only citing ancillary text from `C066` (which contains Recital 63 regarding requests involving large quantities of data).
  * **Answer 2 (Re-ranked):** Directly provided the core statutory disclosure checklist under Article 15(1) (`C232`): confirmation of processing, access to data, purposes of processing, categories of personal data, recipients, and envisaged storage periods.

### 2.3 Document-Specific Failure Modes & Ceiling Limitation
* **Keyword Traps & Definitions (Q01):** The query asks whether companies can share "user passwords". GDPR never mentions passwords; it defines "third party" and "consent" in Article 4 (`C187`). While the Cross-Encoder promoted `C187` from dense #5 to CE #1 based on semantic alignment with third-party disclosures, both LLM answers correctly refused to answer because Article 4 contains definitions rather than authorization rules for credentials.
* **Negation & Constraint Framing (Q05):** Article 8 frames child consent thresholds affirmatively ("processing shall be lawful only if..."), whereas the query asks "What is explicitly prohibited...". Both models placed `C205` at #5, illustrating that syntactic polarity gaps remain challenging when the underlying text lacks explicit negative tokens.
* **The Retrieval Ceiling Limitation (Q03, Q10, Q12):** In Q03 (Article 9 sensitive data exceptions), Q10 (Article 83 fines), and Q12 (Article 89 research), the critical chunks were ranked >#10 by the bi-encoder (e.g. `C208` was dense #19). Because Stage 1 only passes the Top 10 candidates to Stage 2, the Cross-Encoder never saw these passages. **Key Finding:** *A re-ranker cannot rescue chunks outside the first-stage candidate pool. Stage 1 recall forms the absolute upper bound of Stage 2.*

---

## 3. Latency Profiling

Retrieval and generation latencies were measured on CPU with pre-warmed models across N = 320 retrieval runs and n = 16 LLM calls using monotonic timers (`time.perf_counter()`).

| Pipeline / Execution Stage | Median (ms) | Mean (ms) | p95 (ms) | Samples (n) | Overhead vs. Dense |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Pipeline 1: Dense Retrieval** (Query encode + FAISS IndexFlatIP Top-10) | **20.4** | 17.1 | 21.6 | 320 | 1.0x (Baseline) |
| **Re-Ranking Step Only** (Cross-Encoder on 10 pairs + sort) | **424.0** | 391.8 | 437.8 | 320 | +424.0 ms |
| **Pipeline 2: Full Two-Stage Retrieval** (Dense Top-10 + Re-Ranking) | **443.8** | 408.8 | 459.0 | 320 | 21.7x |
| **LLM Generation: Answer 1** (Baseline Top-3 -> Gemini API) | **1,351.1** | 2,412.7 | 6,807.1 | 16 | — |
| **LLM Generation: Answer 2** (Re-ranked Top-3 -> Gemini API) | **1,330.0** | 2,269.0 | 7,391.3 | 16 | — |

### Latency Trade-Off Analysis
1. **Retrieval vs. Generation:** Re-ranking adds ~424 ms of CPU time, but end-to-end latency is dominated by LLM generation (~1,330 ms median).
2. **Interactive Feasibility:** Adding 424 ms increases total round-trip time from ~1.37 s to ~1.77 s (+402 ms / +29% turnaround increase), well within standard user tolerances (sub-2 seconds) while delivering a **+25.0 percentage point boost in Top-3 retrieval hit rate**.

---

## 4. Scaling Question: Why Not Run a Cross-Encoder Over the Whole Collection?

A central system design question is: *Why not run a Cross-Encoder over the entire document corpus from the very start, eliminating dense retrieval entirely?*

### 4.1 Asymptotic Complexity Comparison
Let N be corpus chunks, L_q query length, L_d chunk length, and d embedding dimension.

* **Bi-Encoder Scaling:**
  * *Offline (once):* Chunks are encoded once into vector space: `O(N · L_d²)`.
  * *Online (query time):* Query is encoded once: `O(L_q²)`. Nearest-neighbor search over pre-computed vectors requires `O(N · d)` dot products with exact flat scan, reducible via ANN indexing (FAISS HNSW/IVF) to sub-linear logarithmic time:
    `Query Complexity (Bi-Encoder) = O(L_q² + log N · d)`
* **Exhaustive Cross-Encoder Scaling:**
  * Query and document are concatenated: `(L_q + L_d)` tokens.
  * Attention across all 6 layers has quadratic token complexity `O((L_q + L_d)²)`.
  * Because cross-attention dynamically couples query and chunk tokens, **no pre-computation or indexing is possible**. Every query requires evaluating all N chunks online at runtime:
    `Query Complexity (Exhaustive CE) = O(N · (L_q + L_d)²)`

### 4.2 Empirical & Projected Scaling Benchmark
Using our measured CPU throughput of ~41.0 ms per (query, chunk) pair:

| Collection Scale | Chunk Count (N) | Exhaustive Cross-Encoder Latency | Two-Stage Funnel (Dense + Top-10 CE) | Speedup Factor |
|:---|:---:|:---:|:---:|:---:|
| **This Experiment (GDPR)** | 444 | **18.2 seconds** | **0.44 seconds** | **41.4x** |
| **Enterprise Knowledge Base (Est.)** | 10,000 | **410.3 seconds (~6.8 min)** | **~0.45 seconds (with ANN)** | **~910x** |
| **Web / Legal Corpus (Est.)** | 1,000,000 | **41,032.8 seconds (~11.4 hours)** | **~0.46 seconds (with ANN)** | **~89,000x** |

*Note: 10k and 1M rows represent theoretical projections assuming sub-linear ANN candidate retrieval for Stage 1.*

### 4.3 Conclusion: The Two-Stage Funnel as Optimal Architecture
Exhaustive cross-encoder retrieval is computationally impractical for real-time interactive RAG. The **Two-Stage Funnel** provides the optimal engineering compromise:
* **Stage 1 (Bi-Encoder):** Casts a wide net, pruning candidate space from N = 10⁶ down to K = 10 with sub-linear complexity `O(log N)`.
* **Stage 2 (Cross-Encoder):** Focuses compute on the critical boundary, scoring only K = 10 candidate pairs with constant `O(K)` latency (~424 ms).

This guarantees the semantic precision of joint self-attention at constant, scalable sub-second latency.

---

## 5. Experimental Setup & Limitations (Viva Defense Notes)

* **Corpus & Chunking:** Official GDPR PDF (88 pages) split into 444 sentence-aligned chunks (target ~140 tokens, zero overlap).
* **Generation Settings:** Google Gemini `gemini-3.1-flash-lite`, temperature 0.0, prompt strictly requiring normative keyword retention (`SHALL`, `SHALL NOT`) and passage citations (`[C...]`).
* **Hardware:** Intel Core CPU laptop, models evaluated locally on CPU.
* **Limitations:** 16 hand-crafted queries offer descriptive stress testing rather than statistical significance; single regulatory domain; bi-encoder Stage 1 ceiling dictates the upper bound of re-ranking recall.

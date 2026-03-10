# Dissertation Writing: Brain Dump & Standards

This document captures the key axes, standards, and takeaways that will guide the writing of this dissertation. Everything here is aligned to this project: an LLM-integrated ABM for forward simulation of political reply threads on Twitter/X.

**Global rule: All writing uses UK English throughout.**

---

## 1. The Story We Are Telling

Every strong dissertation has a clear narrative. Ours has **two pillars**:

> **Pillar 1 - Forward-looking thread simulation, not retrospective engagement analysis.** Most political social-media work explains what already happened (likes, reposts, reply counts). We build a system that starts from a root tweet and historically grounded agent profiles, then generates the likely reply thread trajectory without seeing target-thread replies.

> **Pillar 2 - A validated research instrument, not only a model demo.** We do not just show generated text examples. We validate at scale (100 held-out threads), quantify realism with distributional metrics (JSD, residual tests, overall accuracy, composite fidelity), run ablations, and test cross-model transfer (Dolphin-Llama3 8B vs Qwen2.5 7B).

The contribution is both a **method contribution** (an end-to-end ABM+LLM framework for thread-level counterfactual studies) and an **evaluation contribution** (a reproducible, multi-metric validation protocol for generated political discourse).

### Why This Matters (The "So What?" Test)

- **Method gap:** Engagement-only analytics cannot test controlled "what if we changed one word?" questions.
- **Research utility:** The simulator enables counterfactual experiments under fixed audience composition and fixed behavioural assumptions.
- **Validation scale:** 100-thread held-out evaluation gives stronger evidence than single-thread anecdotes.
- **Empirical performance:** Mean overall accuracy `92.3% ± 4.6%`, composite fidelity `0.857 ± 0.106`.
- **Bias control:** No significant directional sentiment bias in final system (Wilcoxon `p = 0.633`).
- **Reproducibility:** Batch scripts, manifests, logs, and figure-generation pipeline are already in-project.

### Positioning Against the Field

| Approach | Typical Output | Forward from root only? | Generates full thread text? | Thread-level quantitative validation? |
|--------|----------------|-------------------------|-----------------------------|--------------------------------------|
| Engagement prediction papers | scalar/label outcomes | No | No | Usually limited |
| Classical opinion ABMs | latent scalar opinions | Yes | No | Not directly text-comparable |
| LLM social simulation papers | generated interactions | Sometimes | Sometimes | Varies |
| **This dissertation** | synthetic reply threads | **Yes (0-shot target)** | **Yes** | **Yes (100-thread, multi-metric)** |

---

## 2. Edinburgh Marking Criteria - What Gets a First

Edinburgh Informatics dissertations are assessed on clear axes. We should explicitly map our evidence to each axis.

### Basic Criteria (Must be solid)

| Criterion | How We Demonstrate It |
|-----------|----------------------|
| **Understanding of the problem** | Clear framing of engagement-vs-reaction gap and why forward simulation is hard |
| **Completion of the work** | End-to-end pipeline built and validated on 100 held-out threads |
| **Quality of the work** | Reproducible scripts, ablation, cross-model comparison, statistical testing |
| **Quality of the dissertation** | Structured chapters, interpretable figures, explicit claims with evidence |

### Additional Criteria (Distinguish a First from a 2:1)

| Criterion | How We Demonstrate It |
|-----------|----------------------|
| **Knowledge of literature** | Link opinion-dynamics ABMs, LLM simulation, and political discourse measurement literature |
| **Critical evaluation of own work** | Honest failure modes: sentiment polarity inversion, aggression miscalibration |
| **Justification of design decisions** | Explain each key choice (0-shot, synchronous activation, context window, vocab injection, etc.) |
| **Solution of conceptual problems** | Model mismatch discovery (Hillary stance), sentiment bias repair, metric design |
| **Amount of work** | Data pipeline, five-model classifier stack, simulator, evaluation suite, ablation, cross-model reruns |

### Exceptional Criteria (80+ territory)

| Criterion | How We Demonstrate It |
|-----------|----------------------|
| **Originality** | Forward thread-level generation instrument with counterfactual protocol in this dataset setting |
| **Material worthy of publication** | Scalable validation protocol + actionable sensitivity findings for LLM social simulation |

### The Critical Insight

> The dissertation should be mostly your method, implementation choices, evidence, and critical analysis. Background should motivate, not dominate.

---

## 3. Lessons from High-Scoring Informatics Dissertations

### What We Should Replicate

| Pattern | How We Apply It |
|---------|-----------------|
| **Explicit contributions early** | Bullet list in Introduction with 4 concrete, testable contributions |
| **Decision + alternatives + trade-off** | Every major design choice gets this structure |
| **Honest limitations** | Treat failure modes as findings, not as footnotes |
| **Quantitative evaluation over multiple conditions** | 100-thread validation + 21-config ablation + cross-model comparison |
| **Background proportionate to body** | Keep background concise and tied to method decisions |
| **Strong figure-driven communication** | Pipeline diagrams, JSD plots, fidelity distributions, ablation leaderboard |

### Common Weaknesses We Must Avoid

1. Single-metric evaluation when failure is multi-dimensional.
2. Claims without statistical testing.
3. Overlong background that delays your contribution.
4. Thin conclusion that repeats the abstract without synthesis.
5. Hidden assumptions about classifier reliability or metric weighting.

### Structure Pattern to Follow

```text
1. Introduction
2. Background and Relevant Work
3. Methodology
4. Implementation
5. Evaluation
6. Conclusions
   Appendix / Reproducibility Notes
```

---

## 4. Excelling at the Marking Criteria

### Basic Criteria

#### Understanding of the problem
- **Meeting it**: Say engagement metrics are limited.
- **Excelling**: Show why thread-level forward simulation needs both behavioural modelling and text generation, then show why this is evaluation-heavy rather than prompt-demo work.

#### Completion of the project
- **Meeting it**: One working simulation example.
- **Excelling**: Full pipeline + 100-thread run + ablation + cross-model, with reproducible artefacts.

#### Quality of the work
- **Meeting it**: Code runs.
- **Excelling**: Evidence of systematic engineering: checkpointed classification, consistent classifier reuse, robust batch pipelines, explicit metric definitions.

#### Quality of the report
- **Meeting it**: Clear writing and structure.
- **Excelling**: Every section answers a clear question and ends with a concrete takeaway tied to evidence.

### Additional Criteria

#### Knowledge of literature
- **Meeting it**: Mention ABM and LLM simulation papers.
- **Excelling**: Connect classical opinion models, modern LLM social simulation, and political discourse measurement limits into one coherent method rationale.

#### Critical evaluation of previous work
- **Meeting it**: Summarise prior systems.
- **Excelling**: State exactly what each class of work cannot do that this instrument can do (forward 0-shot thread generation with thread-level validation).

#### Critical evaluation of own work
- **Meeting it**: Mention limitations briefly.
- **Excelling**: Diagnose dominant error modes (polarity inversion, moderate aggression tracking), then bound claims accordingly.

#### Justification of design decisions
- **Meeting it**: List chosen settings.
- **Excelling**: For each setting, provide mechanism + ablation evidence.

#### Solution of conceptual problems
- **Meeting it**: Mention fixes.
- **Excelling**: Explain why fixes worked (e.g., removing global hostility floor, dynamic aggression scaling, larger context window).

#### Amount of work
- **Meeting it**: Describe pipeline components.
- **Excelling**: Make the full scope explicit: data prep, model stack selection, simulation engine, validation suite, statistical tests, ablation grid, model transfer study.

### Exceptional Criteria

#### Evidence of originality
- Methodologically original in this project context: a validated forward simulation instrument for thread-level counterfactuals.

#### Outstanding scholarship/engineering
- Strongest if claims stay bounded and defensible: feasibility in USC X 24 setting, not universal social behaviour modelling.

---

## 5. Key Takeaways from Top Papers

1. **Operationalise clearly.** Define exactly what is being simulated (observable thread-level reaction dynamics), not latent internal truth.
2. **Validate with multiple lenses.** Sentiment alone can look good while politics/emotion fail.
3. **Negative results are valuable.** Document failed settings and why they failed.
4. **Ablation is method, not decoration.** It turns prompt choices into evidence-backed design.
5. **Theoretical grounding still matters.** Bounded confidence and backfire-style mechanisms motivate behaviour rules.
6. **Counterfactual design is the methodological payoff.** The case study should show what this instrument enables that retrospective correlation cannot.

### One-Paragraph Positioning Statement (for the Introduction)

> Existing political social-media research is predominantly retrospective and engagement-centric, while classical opinion ABMs do not generate text and LLM demos often lack rigorous thread-level validation. This dissertation presents an LLM-integrated ABM that generates full reply threads in strict forward mode from a root tweet and history-informed agent profiles, then validates realism across 100 held-out threads using distributional and statistical metrics. The resulting framework functions as a research instrument for controlled counterfactual studies of political discourse, rather than only a generative demonstration.

---

## 6. Dissertation Structure

### Proposed Chapter Outline

1. **Introduction** (~3-4 pages)
   - Problem, gap, objectives, explicit contributions
2. **Background & Related Work** (~5-6 pages)
   - Opinion dynamics, LLM social simulation, political discourse metrics, gap
3. **Methodology** (~7-9 pages)
   - Data, agent construction, behavioural rules, metrics, experiment design
4. **Implementation** (~7-9 pages)
   - Pipeline evolution, model stack, simulation architecture, optimisation decisions
5. **Evaluation** (~7-9 pages)
   - 100-thread validation, composite fidelity, ablation, cross-model comparison
6. **Conclusion & Future Work** (~2-3 pages)
   - Claims, limits, and next studies

---

## 7. Writing Rules (Non-Negotiable)

### Language

- **UK English throughout.**
- **Active voice.**
- **Scientific `we`.**
- **No hype words.** Be specific instead.
- **Short, precise sentences.**
- **Define acronyms on first use.**

### Structure

- **Problem before solution.**
- **One message per paragraph.**
- **State contributions explicitly.**
- **Decision -> alternatives -> rationale -> trade-off.**

### Figures

- **Figure 1 should communicate the full pipeline.**
- **Captions must explain what and why.**
- **Consistent visual style across all plots.**
- **Use vector output where possible.**

### Citations

- **Cite throughout, not only in background.**
- **Pair summary + difference sentence in related work.**
- **Include both foundational and recent papers.**

### Evaluation

- **Include concrete numbers in abstract and conclusion.**
- **Use statistical testing for bias claims.**
- **Report means with variance (`±` std dev).**
- **Document failure modes explicitly.**

---

## 8. Key Papers to Cite

### Foundations

| Paper | Citation Key | Relevance |
|-------|-------------|-----------|
| Deffuant et al. (2000) | `deffuant2000mixing` | Bounded confidence |
| Hegselmann & Krause (2002) | `hegselmann2002opinion` | Opinion dynamics |
| Axelrod (1997), Epstein & Axtell (1996) | `axelrod1997complexity`, `epstein1996growing` | ABM foundations |

### LLM Social Simulation / Political Dynamics

| Paper | Citation Key | Relevance |
|-------|-------------|-----------|
| Park et al. (Generative Agents, 2023) | `park2023generative` | Persona-conditioned interaction |
| Composta et al. (2025) | `composta2025simulating` | Election-domain LLM simulation |
| Butler et al. (2024) | `butler2024misinformation` | Simulation and validation framing |

### Political Discourse Measurement

| Paper | Citation Key | Relevance |
|-------|-------------|-----------|
| Brady et al. (2017) | `brady2017emotion` | Moral-emotional language and diffusion |
| Antypas et al. (2023), Tun et al. (2023) | `antypas2023negativity`, `tun2023sentiment` | Sentiment/emotion measurement |
| Diehl et al. (2016), Cetinkaya et al. (2025) | `diehl2016political`, `cetinkaya2025crosspartisan` | Political thread behaviour |

### Model/Tool References

| Resource | Relevance |
|----------|-----------|
| TweetEval / CardiffNLP model papers | Classifier validity basis |
| USC X 24 dataset paper/docs | Dataset provenance |

---

## 9. The Contributions List (Draft)

1. **A forward, 0-shot ABM+LLM instrument** that generates full political reply threads from a root tweet and history-informed agent profiles, without target-thread reply leakage.
2. **An empirically grounded agent construction workflow** using a five-model Twitter-specialised classifier stack with user-level aggregation.
3. **A reproducible thread-level validation protocol** over 100 held-out threads with JSD, sentiment residual testing, aggression calibration, overall accuracy, and composite fidelity.
4. **Ablation and cross-model evidence** that isolates sensitivity to context window, temperature, reply probability, vocabulary injection, and model family.

---

## 10. Common Pitfalls to Avoid

- [ ] Overclaiming general human-behaviour modelling.
- [ ] Treating one strong metric as proof of total realism.
- [ ] Hiding model-dependent weaknesses (especially sentiment calibration).
- [ ] Reporting improvements without statistical tests.
- [ ] Mixing methodological contribution claims with platform-policy claims.
- [ ] Using vague wording instead of measurable statements.

---

## 11. What Makes This Project Genuinely Interesting

1. **It changes the research question.** From "what correlated with engagement?" to "what likely thread dynamics follow from this post under controlled assumptions?"
2. **It is evaluated as an instrument, not a demo.** 100-thread validation, ablation grid, and cross-model replication.
3. **It surfaces actionable design guidance.** Context, temperature, and reply probability matter materially; vocabulary injection is strongly beneficial.
4. **It supports controlled lexical counterfactuals.** One-word root edits with fixed population isolate language effects more cleanly than retrospective observational studies.

---

## 12. Next Steps for Writing

1. [ ] Finalise Introduction with the 4 contributions verbatim.
2. [ ] Tighten Background so every subsection motivates a method choice.
3. [ ] Ensure Methodology equations/metrics are defined once and reused consistently.
4. [ ] Keep Implementation chronological: failure -> diagnosis -> redesign.
5. [ ] Keep Evaluation claim-first with numbers in opening sentence of each subsection.
6. [ ] Expand limitations with explicit validity-threat framing.
7. [ ] Ensure all figures are referenced and interpreted in text.
8. [ ] Final proofread for UK English and style consistency.

---

## 13. Evaluation Snapshot (Reference Numbers)

### 100-Thread Validation (Final Llama Pipeline)

| Metric | Result |
|--------|--------|
| Sentiment JSD (mean ± sd) | `0.086 ± 0.079` |
| Political JSD (mean ± sd) | `0.039 ± 0.038` |
| Emotion JSD (mean ± sd) | `0.118 ± 0.105` |
| Sentiment residual (mean ± sd) | `+0.025 ± 0.351` |
| Wilcoxon bias test | `p = 0.633` (no directional bias) |
| Overall accuracy | `92.3% ± 4.6%` |
| Composite fidelity | `0.857 ± 0.106` |

### Cross-Model (100 Threads)

| Metric | Llama | Qwen | Better |
|--------|-------|------|--------|
| Sentiment JSD | 0.086 | 0.120 | Llama |
| Political JSD | 0.039 | 0.013 | Qwen |
| Emotion JSD | 0.118 | 0.092 | Qwen |
| Abs sentiment residual | 0.261 | 0.351 | Llama |
| Abs aggression gap | 0.082 | 0.152 | Llama |
| Overall accuracy (%) | 92.3 | 91.2 | Llama |

### Ablation (21 Configurations, 210 Simulations)

| Finding | Evidence |
|--------|----------|
| Near-optimal baseline | 0-shot baseline rank 1 (0.780) |
| Lower reply probability helped | `reply_prob=0.02` rank 2 (0.777) |
| Vocabulary injection is critical | removing it dropped to rank 16 (0.548) |
| Larger context helped | 15-post context rank 6 (0.713) |
| Lower temperature hurt | `temp=0.7` score 0.451 |
| Long runs drift more | 20 rounds score 0.411 |

---

## 14. The Sentiment Calibration Problem - How to Frame It

### The Problem

Early pipeline outputs were too negative (`mean residual = -0.390`, Wilcoxon `p = 4.27e-18`).

### Why It Happened

- Global prompt rule imposed a hostility floor.
- Aggression tiers were too combative even for low-aggression agents.
- Short context window limited tone calibration.

### What Fixed It

1. Softer five-tier aggression mapping.
2. Replaced hostile global instruction with tone-variation instruction.
3. Dynamic aggression scaling relative to thread mean.
4. Expanded context window from 3 to 8 posts.

### Evidence of Fix

Final system removed directional bias (`p = 0.633`) while retaining high distributional realism.

---

## 15. Writing Style Guide - Avoiding AI-Sounding Prose

### Words to Cut or Replace

| Avoid | Prefer |
|------|--------|
| leverage | use |
| facilitates | enables |
| crucial/pivotal | important (or explain why) |
| underscores | shows |
| multifaceted | specific description |
| robust (without evidence) | quantified performance statement |

### Structural Habits to Avoid

1. Significance inflation without data.
2. Vague attribution ("studies show").
3. Repetitive paragraph openings.
4. Filler transitions where logic is already clear.
5. Claim-heavy paragraphs with no numbers.

### Quick Self-Check

1. Can this paragraph be summarised in one sentence?
2. Is every claim evidence-backed or cited?
3. Is there a concrete number where one should exist?
4. Am I stating limits as clearly as strengths?

---

## 16. Key New References to Add/Check

### Must-Have Method References

| Topic | Reference Type |
|------|-----------------|
| USC X 24 dataset provenance | dataset paper / official documentation |
| TweetEval/CardiffNLP models | benchmark and model papers |
| LLM social simulation | 2023-2025 core papers |
| Opinion dynamics foundations | Deffuant / HK / ABM foundations |
| Political discourse dynamics | sentiment/emotion/aggression studies |

### Optional But Valuable

| Topic | Value |
|------|-------|
| Classifier uncertainty propagation | strengthens analytical validity section |
| Human evaluation protocols for generated dialogue | supports future work feasibility |
| Cross-platform discourse studies | strengthens external validity framing |

---

*This document is a reference guide. Update it as writing progresses and as final results/tables are frozen.*

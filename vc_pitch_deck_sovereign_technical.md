# Sovereign Technical VC Deck Outline

Accuracy guardrail: this rewrite treats `dissertation.tex` and `whitepaper.tex` as source of truth for metrics and ablation findings, and avoids older implementation notes where intermediate ablation interpretations diverge from the final evaluation.

## Refined Markdown Outline

### Slide 1 — Cover

**Headline:** The sovereign technical wind tunnel for human discourse

**Subhead:** 0-shot forward simulation of thread-level discourse trajectories before a message goes live.

**Proof strip:** 92.3% overall accuracy on 100 held-out threads | 94.54% weighted accuracy on strict chronological holdout | 21-configuration ablation study

**Founder lockup:** Luke Tervit | First Class CS Edinburgh | ex-Granola AI Product Eng | viral hackathon winner

**Speaker note:** This is not post-mortem analytics. It is pre-deployment decision infrastructure for teams whose downside lives in the replies.

### Slide 2 — Problem

**Headline:** The market still buys autopsies when it needs wind tunnels

**Left column:** Post-Mortem Analytics / Autopsy

- Measures what happened after launch
- Optimizes dashboards, not downstream discourse
- Explains backlash after capital is already burned

**Right column:** Pre-Deployment Simulation / Wind Tunnel

- Forecasts likely discourse shape before launch
- Compares variants before exposure
- Lets teams trade off persuasion, backlash risk, and sentiment balance ex ante

**Bottom line:** The unit of risk is not the post. It is the conversation the post creates.

### Slide 3 — Decision Failure Proof

**Headline:** Big Arch is the canonical failure mode: polished asset, broken reaction curve

**Narrative:** McDonald's did not fail on production quality. It failed on discourse forecasting. Chris Kempczinski's careful bite and repeated "product" language created an authenticity mismatch that social media immediately weaponized into meme velocity. The damage was not in the video asset itself. The damage was in the reply spiral.

**Frame it as the wedge:**

- Internal review says: safe, polished, on-brand
- The internet says: uncanny, corporate, not credible
- Existing tools see this after the mockery cycle starts
- A simulation stack is built to surface that mismatch before launch

**Speaker note:** This is the class of decision failure we sell against. Not "bad creative." Bad forecast.

### Slide 4 — Solution

**Headline:** We do distributional forecasting, not engagement prediction

**Subhead:** We predict the shape of the spiral, not just the count.

**Core product statement:**

- Input one root post or multiple variants
- Simulate historically grounded responders before publishing
- Forecast thread-level distributions across sentiment, emotion, political composition, and aggression
- Rank variants by likely discourse trajectory, not vanity metrics

**Claim discipline:** We do not claim exact tweet prediction or polling replacement. We claim bounded, forward-looking discourse forecasting from the root post alone.

### Slide 5 — System

**Headline:** Empirically grounded agents. Synchronous thread simulation. Shared evaluation stack.

**System arc:**

- Build responder personas from historical social behavior using five classifiers: political leaning, emotion, sentiment, hate, and offensive language
- Start every run in 0-shot mode with only the root tweet exposed
- Simulate 10 synchronous rounds of interaction where agents reply probabilistically and condition on recent thread context
- Reclassify both real and simulated replies with the same metric stack to score distributional fidelity

**Engineering depth line:** This is not a prompt wrapper. It sits on 4,000+ lines of architected simulation, validation, batch-run, and analysis code.

**Code-informed language to preserve:**

- Historically grounded audience simulation
- Behaviourally weighted reply probability
- Vocabulary-conditioned partisan realism
- Concurrent generation with full per-post logging

### Slide 6 — Validation

**Headline:** Forward prediction is already working on unseen discourse

**Primary proof:**

- 92.3% mean overall accuracy on 100 held-out threads
- 97 of 100 threads score in the EXCELLENT band
- No significant directional sentiment bias in the final pipeline
- 6,097 simulated tweets vs 6,528 real tweets at corpus level

**Strict holdout proof:**

- 94.54% mean weighted accuracy on a strict chronological holdout of unseen tweets
- Root tweet only
- No real target replies exposed

**Interpretation:** The system is strongest at macro-distributional realism. It forecasts how a thread is likely to unfold across key discourse dimensions before the thread exists.

### Slide 7 — Why This Is Not A Wrapper

**Headline:** The 21-configuration ablation is the anti-wrapper proof

**Core argument:** If this were prompt luck, the system would not degrade in coherent ways when key behavioural levers are removed. It does.

**Evidence frame:**

- 21 configurations across 10 threads each, for 210 total simulations
- Remove vocabulary injection and performance drops materially
- Short context windows underperform materially versus longer conversational context
- Lower temperature underperforms higher controlled stochasticity
- Extending runs to 20 rounds increases drift and degrades fidelity

**Close:** The outcome is a mechanism-backed system where context, vocabulary, temperature, and simulation length all matter for observable reasons.

### Slide 8 — Moat

**Headline:** The moat is a recalibration flywheel, not a model wrapper

**Flywheel:** Simulations -> Real World Outcomes -> Model Re-Calibration

**What compounds:**

- More simulations create more predicted distribution signatures
- Real-world launches create observed outcome distributions
- Prediction vs outcome gaps improve agent construction and parameter calibration
- Better calibration improves trust, workflow embed, and switching cost

**Moat language:** Validated pipeline, empirically grounded agents, and outcome-linked recalibration.

### Slide 9 — Wedge and Expansion

**Headline:** Start where discourse failure is expensive and time-compressed

**Phase 1:** US campaigns, PACs, consultants, and digital strategists

**Why this wedge wins:**

- Messages are high-frequency and high-stakes
- Polling is too slow for rapid iteration
- A/B tests do not capture thread-level persuasion and backlash dynamics

**Phase 2:** Public affairs, reputation, and enterprise comms

**Phase 3:** API and workflow layer for agencies and platforms

**Discipline line:** This augments polling and social listening. It does not replace them.

### Slide 10 — Founder

**Headline:** Founder with both technical depth and product instinct

**Profile:**

- Luke Tervit built the research, simulation, evaluation, and runtime pipeline end to end
- First Class BSc in Computer Science, University of Edinburgh
- Product Engineering Intern at Granola AI
- Previously built and sold a viral hackathon startup inside 20 hours

**Why this matters:** The product requires someone who can bridge multi-agent systems, rigorous validation, and high-stakes communication UX. That bridge already exists in the founder.

### Slide 11 — Ask

**Headline:** Raise capital to turn validated simulation into category-defining product

**Use of funds:**

- Productionize the runtime for fast pre-deployment testing loops
- Secure political design partners and close the validation-to-revenue gap
- Build the recalibration layer that learns from real launch outcomes
- Expand from politics into reputation-sensitive enterprise workflows

**Closing line:** Build the decision-support layer for persuasion-driven communication.

## Visual Briefs

### Slide 1 — Cover Visual Brief

Dark-mode only. Matte graphite background with deep navy paneling, off-white typography, arch-gold accent, and restrained cyan for evidence labels. Set the proof strip as three illuminated data pills across the lower third. Typography should feel sovereign and technical, not consumer AI. Use a single vector motif: root post -> branching reply tree -> distribution curves.

### Slide 2 — Problem Visual Brief

Use a split-screen forensic metaphor. Left side is "Autopsy" in dull gray-red with static dashboard tiles and dead-end arrows. Right side is "Wind Tunnel" in gold-cyan with airflow lines wrapping around a post card and branching into discourse paths. The middle divider should feel like a hard category boundary, not a soft comparison.

### Slide 3 — Decision Failure Proof Visual Brief

Make this the emotional hinge slide. Left third: a dark still-frame card representing the Big Arch launch moment. Center: two oversized phrase callouts, "tiny bite" and "product," treated as discourse triggers. Right third: a cascading reaction map showing irony, authenticity skepticism, and competitor pile-on. The point is to visualize a reaction spiral, not a social media collage.

### Slide 4 — Solution Visual Brief

Hero visual is a distributional forecast panel. Show three candidate root posts on the left and three projected trajectory cards on the right. Each forecast card should include mini heatmaps for sentiment by round and a simple risk dial for aggression/polarization balance. Avoid CTR-style charts entirely.

### Slide 5 — System Visual Brief

Use the existing pipeline assets as a base: `dissertation_figures/final/01_personality_extraction_pipeline.png` and `dissertation_figures/final/02_simulation_pipeline.png`. Redraw them into a unified dark-mode panel with numbered stages. Add a radar-chart row for three representative agents, using only historically grounded dimensions or directly derived behavioural signals. Good axes: political leaning, sentiment tendency, aggression, hate/offensive exposure, reply propensity.

### Slide 6 — Validation Visual Brief

This slide should feel like a data room compressed into one screen. Left: oversized 92.3% hero metric with a histogram inset based on `dissertation_figures/final/06_overall_accuracy_histogram.png`. Right: a second card for 94.54% strict chronological holdout. Bottom band: compact evidence tiles for "97/100 EXCELLENT," "0-shot," and "no directional sentiment bias." If space allows, include a thin real-vs-simulated volume bar for 6,528 vs 6,097 tweets.

### Slide 7 — Why This Is Not A Wrapper Visual Brief

Lead with the heatmap. Adapt `dissertation_figures/final/10_ablation_metric_heatmap.png` into a high-contrast dark-mode heatmap with gold highlights on the strongest configurations and iron-red on degraded ones. Pair it with a narrow ranked strip based on `dissertation_figures/final/11_ablation_leaderboard.png`. The caption should read like infrastructure evidence: remove mechanism, lose fidelity.

### Slide 8 — Moat Visual Brief

Use a circular flywheel with three hard nodes: Simulations, Real World Outcomes, Re-Calibration. Each node should have a compact artifact icon: forecast distribution, observed thread distribution, parameter update matrix. Add a faint background grid of model/version logs to reinforce that the moat is operational and compounding.

### Slide 9 — Wedge and Expansion Visual Brief

Build this as a vertical market ladder. Politics at the base in bright gold, reputation/public affairs in muted silver, platform/API in cool cyan. On the left, show why politics is the best wedge with three urgency markers: fast cycle time, high downside, budget owner close to pain. Keep TAM language out unless separately sourced.

### Slide 10 — Founder Visual Brief

No generic headshot slide. Use a strong portrait crop on one side and a technical credibility column on the other. Include a compact build panel listing "research," "runtime," "evaluation," and "deck/productization" as shipped layers. The message is founder-market-technical fit, not biography.

### Slide 11 — Ask Visual Brief

Close with a capital deployment map. Three stacked rails: production runtime, design partners, recalibration loop. Background motif should echo Slide 1's branching reply tree, but now tightened into a controlled system diagram to imply inevitability and execution. Final line should sit alone with heavy negative space.

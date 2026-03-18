# Pitch Deck Notes

## Tagline

**The digital wind tunnel for human discourse.**

## Core Positioning

Stop guessing how the world will react.

We provide AI-driven simulation for high-stakes political and corporate communication, allowing teams to stress-test messages before they hit the real world.

The safest version of the claim is:

**We forecast likely thread-level reaction patterns before publishing.**

Not:

- exact backlash prediction
- polling replacement
- guaranteed real-world outcomes

## Slide 1: The Vision

**Stop guessing how the world will react.**

- High-stakes communicators still publish into uncertainty.
- We let teams test messages in a simulated environment before they go live.
- The goal is not post-mortem analytics. The goal is pre-deployment decision support.

## Slide 2: The Problem

**Current tools are retrospective.**

- Social listening tools tell you what happened after the post.
- A/B testing is slow, expensive, and often disconnected from the actual discourse that follows.
- Political and reputation-sensitive teams still rely on instinct for high-risk messaging decisions.
- What matters is not just engagement. It is the conversation a post creates.

Good line:

**Current tools are autopsies, not forecasts.**

## Slide 3: The Solution

**Predictive discourse simulation.**

- Input a draft message.
- Simulate likely audience reaction before publishing.
- Compare multiple variants side by side.
- Optimize for a goal such as lower backlash risk, stronger persuasion, or better sentiment balance.

Good line:

**Simulate, don’t guess.**

## Slide 4: Why This Is Different

**We simulate discourse, not just engagement.**

- We do not just score sentiment.
- We do not just predict clicks or impressions.
- We simulate thread-level dynamics: sentiment, aggression, political mix, and emotional tone.
- The output is a likely conversation pattern, not a vanity metric.

## Slide 5: How It Works

**Empirically grounded agents.**

- Audience personas are built from real historical social behavior data.
- Users are classified across political leaning, sentiment, emotion, hate, and offensive language.
- Multi-agent simulations run over multiple rounds, where agents react both to the root post and to each other.
- Outputs are scored on predicted sentiment, aggression, political composition, and emotional trajectory.

Good line:

**Beyond prompting: historically grounded audience simulation.**

## Slide 6: Validation

**The proof is in forward prediction.**

- Validated on 100 held-out political threads using 0-shot forward simulation.
- Mean overall accuracy: **92.3% +- 4.6%**
- **97 of 100** threads scored in the excellent band.
- No significant directional sentiment bias in the final pipeline.
- Separate strict chronological holdout on unseen tweets reached **94.54%** mean weighted accuracy.
- A 21-configuration ablation study shows performance comes from system design, not prompt luck.

## Slide 7: Why Now

**The timing is real.**

- LLM-based multi-agent systems are finally good enough to model realistic conversation dynamics.
- Political and reputation risk now moves at social speed.
- Teams need faster message testing loops than polling, focus groups, or live-fire posting.
- The US political market is a strong wedge: urgent, high-value, and outcome-sensitive.

Important framing:

This is **not** a replacement for polling.

It is a new layer for **pre-deployment message testing**.

## Slide 8: Wedge and Expansion

**Start narrow, expand into adjacent high-value workflows.**

- **Phase 1:** US campaigns, PACs, consultants, and digital strategists.
- **Phase 2:** Public affairs, PR, and reputation-sensitive enterprise teams.
- **Phase 3:** API and workflow layer for agencies and social media platforms.

## Slide 9: Moat

**The moat is calibration, validation, and workflow fit.**

- A validated forward-simulation pipeline, not a generic LLM wrapper.
- Empirically grounded agent construction from historical behavior.
- Calibration supported by ablation studies and cross-model comparison.
- A growing feedback loop: more simulations and outcomes improve system calibration over time.

Avoid saying:

- proprietary dataset
- custom-trained SLMs

Safer language:

**A calibrated simulation stack built for realistic political discourse.**

## Slide 10: Team

**Luke Tervit**  
Founder & Lead Engineer

- Built the full research and simulation pipeline end to end.
- Product Engineering Intern at Granola AI.
- On track for a First Class BSc in Computer Science from the University of Edinburgh.
- Previously co-founded and sold a viral hackathon startup within 20 hours.

## Slide 11: The Ask

**Raise capital to turn the research system into a production product.**

- Convert the validated research pipeline into a production-grade platform.
- Secure early design partners in US politics.
- Reduce runtime and improve usability for fast campaign workflows.
- Expand into enterprise reputation and communications use cases.

Good line:

**Build the decision-support platform for persuasion-driven communication.**

## Strong One-Liners

- The digital wind tunnel for human discourse.
- Simulate, don’t guess.
- Test reaction before you post.
- From root post to predicted discourse.
- We simulate how a message is likely to unfold, not just whether it gets clicks.

## Claims To Avoid

- “We predict exact backlash.”
- “We are the first” unless you can defend it.
- “Polling replacement.”
- “Proprietary dataset.”
- “Custom-trained models” unless that becomes true.
- Hard market-size numbers unless you have a source on the slide or in backup.

## Best Evidence To Reference

- [whitepaper.tex](/Users/luketervit/Desktop/Dissertation_Final/whitepaper.tex#L27)
- [dissertation.tex](/Users/luketervit/Desktop/Dissertation_Final/dissertation.tex#L434)
- [dissertation.tex](/Users/luketervit/Desktop/Dissertation_Final/dissertation.tex#L498)
- [dissertation.tex](/Users/luketervit/Desktop/Dissertation_Final/dissertation.tex#L637)
- [PRODUCT_IMPLEMENTATION_PLAN.md](/Users/luketervit/Desktop/Dissertation_Final/PRODUCT_IMPLEMENTATION_PLAN.md#L3)
- [openrouter_runtime/simulation.py](/Users/luketervit/Desktop/Dissertation_Final/openrouter_runtime/simulation.py#L107)

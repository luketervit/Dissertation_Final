# Simulating Political Discourse on Twitter/X

This repository contains the final dissertation project for Luke Tervit:

**Simulating Political Discourse on Twitter/X: A Large Language Model-Integrated Agent-Based Model for Forward-Looking Thread Generation**

The dissertation asks a forward-looking question:

> Given only a root tweet and historically grounded user profiles, can an LLM-integrated agent-based model generate a realistic political reply thread?

The answer argued in the dissertation is yes, within clear limits. The project builds agent profiles from historical tweets, simulates reply-thread dynamics with an ABM+LLM pipeline, and validates synthetic threads against real Twitter/X threads using distributional and statistical metrics.

## Dissertation Summary

The core contribution is not just a demo that generates plausible replies. The dissertation positions the pipeline as a **research instrument** for studying downstream discourse effects under controlled conditions.

That means the system is designed to support:

- strict forward simulation from a root tweet in 0-shot mode
- counterfactual message testing by editing the root tweet while holding audience composition fixed
- thread-level evaluation against real data rather than engagement-only proxies
- ablation and model-comparison experiments that turn design choices into evidence

## Headline Results

The abstracted final results reported in `dissertation.tex` are:

- 100 held-out political threads evaluated
- mean overall accuracy: `92.3% ± 4.6%`
- mean composite fidelity score: `0.857 ± 0.106`
- strongest agreement on political composition: mean political JSD `0.039`
- no significant directional sentiment bias in the final Dolphin-Llama3 8B configuration

The dissertation therefore argues that the pipeline is useful for bounded, prospective simulation of political reply dynamics, while explicitly rejecting stronger claims such as universal modelling of political behaviour or direct inference of hidden beliefs.

## What Is In This Repository

This is a **dissertation repository first** and a codebase second. It includes the thesis source, experimental artefacts, analysis outputs, figures, and the simulation code used to produce them.

### Main Dissertation Files

- `dissertation.tex`: final dissertation manuscript
- `mybibfile.bib`: bibliography
- `writing.md`: writing frame, argument structure, and contribution notes
- `Implementation.md`: detailed implementation and redesign notes used to support the write-up

### Core Code

- `sim/thread_simulation.py`: main ABM thread simulator
- `sim/llm_generator.py`: LLM interface layer
- `scripts/step1_classify_chunked.py`: user-profile classification from historical tweets
- `scripts/validate_thread_simulation.py`: thread-level validation
- `scripts/prepare_100_threads.py`, `scripts/run_ablation.py`, `scripts/run_model_comparison_batch.py`: large-scale experiment entry points

### Results And Artefacts

- `analysis/`: aggregated evaluation outputs, summaries, and comparison tables
- `outputs/`: batch execution outputs and parameter-sweep manifests
- `batch_simulations_reconstructed/`: reconstructed 100-thread experiment inputs
- `case_study/`: counterfactual case-study assets
- `dissertation_figures/final/`: final figure pack used in the thesis

### Supporting Docs

- `PROJECT_STRUCTURE.md`: older pipeline map
- `SIMULATION_GUIDE.md`: simulation run notes
- `BATCH_SIMULATION_README.md`: batch execution notes
- `PARAMETER_INVENTORY.md`: tunable simulation and generation parameters

## Repository Structure

```text
.
├── dissertation.tex
├── mybibfile.bib
├── analysis/
├── batch_simulations_reconstructed/
├── case_study/
├── config/
├── data/
├── dissertation_figures/
├── outputs/
├── processed_agents/
├── scripts/
├── sim/
├── tests/
├── Implementation.md
├── writing.md
└── README.md
```

## Method In One Pass

The dissertation pipeline has three main stages:

1. **Agent construction**
   Historical user tweets are passed through five classifiers to estimate political leaning, sentiment, emotion, hate speech, and offensive language. These are aggregated into user-level behavioural profiles.

2. **Forward thread simulation**
   An agent-based model selects whether agents reply, which post they target, and when they act. The LLM then generates the actual text of each reply conditioned on the agent profile and rolling thread context.

3. **Validation**
   Generated replies are classified with the same measurement stack and compared against real threads using Jensen-Shannon divergence, aggression calibration, sentiment residuals, and composite fidelity scoring.

## Building The Dissertation PDF

If your LaTeX environment already has the Edinburgh Informatics thesis template installed (`infthesis` and `ugcheck`), the manuscript can be built with:

```bash
latexmk -pdf dissertation.tex
```

If `latexmk` is unavailable, the manual sequence is:

```bash
pdflatex dissertation.tex
bibtex dissertation
pdflatex dissertation.tex
pdflatex dissertation.tex
```

Note: this repository does **not** include the `infthesis` class or `ugcheck` package source files. They need to be available in your TeX installation for compilation to succeed.

## Running The Code

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The checked-in `requirements.txt` covers the core research stack:

- `pandas`
- `transformers`
- `torch`
- `numpy`
- `tqdm`
- `mesa`
- `pytest`

Some scripts also rely on additional packages or provider SDKs such as `pyyaml`, `matplotlib`, `scipy`, `requests`, `python-dotenv`, `openai`, or `anthropic`. Those are part of the wider experiment environment, but not all of them are pinned in `requirements.txt`.

Example local workflow:

```bash
# 1. Classify one historical data chunk into agent profiles
python scripts/step1_classify_chunked.py 1

# 2. Run the thread simulator
python sim/thread_simulation.py

# 3. Validate simulated output against a real thread
python scripts/validate_thread_simulation.py \
  --real output/selected_thread_metadata.json \
  --simulated output/simulated_thread_metadata.json
```

For larger reruns, see:

- `BATCH_SIMULATION_README.md`
- `SIMULATION_GUIDE.md`
- `config/thread_config.yaml`

## Data And Reproducibility Notes

The dissertation is written around the USC X 24 dataset and derived experiment artefacts stored in this repository.

Important practical points:

- raw and intermediate files are large, and the repository contains both final artefacts and historical experiment outputs
- multiple generations of the pipeline are preserved for comparison, including older and superseded runs
- the README should be read as a map of the final dissertation state, not as a claim that every directory reflects the final chosen method

The reproducibility notes chapter in `dissertation.tex` states that the key final artefacts are organised around:

- `analysis/` for metrics and summaries
- `outputs/` for simulation outputs and run traces
- `dissertation_figures/` for figure assets used in the manuscript

## Final Framing

This project is best understood as a validated dissertation artefact about **forward simulation of political discourse**, not just a generic ABM repo and not just a collection of prompts. The important claim is methodological: that an LLM-integrated ABM can support controlled, thread-level, counterfactual research on online political reaction dynamics.

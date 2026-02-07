🎯 Project Overview
An Agent-Based Model (ABM) dissertation project simulating political persuasion on Twitter/X.

Goal: Predict "Latent Opinion Shift" in the silent majority (lurkers).

Core Dataset: USC X 24 US Election Dataset.

Key Theory: 90-9-1 Rule, Bounded Confidence, and the Backfire Effect.

Deadline: March 9th (Focus on high-speed execution).

🛠 Setup & Run Commands
Environment: Use pip for dependency management (or uv if preferred).

Install Deps: pip install pandas transformers torch numpy tqdm mesa pytest

Classify Agents: python step1_classify.py

Run Simulation: python lurker_sim.py

Run Tests: pytest

🏗 Project Structure
data/: Raw top_conversations.csv and chunk files.

models/: Stance and Emotion classification logic.

sim/: Mesa ABM logic (Agents, Model, Schedule).

scripts/: Data processing and DNA extraction.

output/: Results, processed_agents.csv, and "Ghost Shift" plots.

🧬 ABM Implementation Rules
Claude must strictly adhere to these modeling principles:

The 90-9-1 Rule: Always spawn ~15-20 Lurker agents for every 1 Active agent based on viewCount.

Latent DNA: Active agents must have stance_score and aggression_score derived from RoBERTa.

Opinion Update Logic:

Bounded Confidence: If distance < threshold, move toward tweet.

Backfire Effect: If distance > 0.6 AND aggression > 0.7, move away from tweet.

Temporal Validation: Always use the 12h/12h Split (Phase 1 Playback, Phase 2 Prediction).

Vectorization: Use NumPy/Pandas vectorization for agent updates where possible to ensure 100k+ agents run in < 1 minute.

📝 Style & Quality Guidelines
Python Style: PEP 8, 88-character line limit (Ruff/Black compatible).

Type Hints: Required for all agent logic and update functions.

Documentation: All scientific assumptions (e.g., Pew 2024 distributions) must be documented in code comments.

Hardware: Prioritize local inference (torch with CPU/GPU) over paid APIs.

GCP Instance: The GCP VM has a Tesla T4 GPU. ALWAYS install PyTorch with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu124`), NEVER use the CPU-only version. Always verify GPU is available with `torch.cuda.is_available()` before running classification or inference jobs. Ollama also uses the T4 for LLM inference.

🎓 Dissertation-Specific Requirements
Validation: "Success" is defined by matching simulated Active behavior in T2 to real-world T2 data.

The "Ghost Shift": All simulation runs must track the cumulative opinion change of the Lurker population, even if they never post.

Sensitivity Analysis: When requested, run the simulation with varying Stubbornness or Volatility parameters to test robustness.

Every time something is implemented you should write why we have chosen this option and any assumptions made in @implementation.md , this content should have enough detail to use as notes when writing out the thesis, but not too overloaded.

📅 Milestones
By Feb 5: Fully processed processed_agents.csv.

By Feb 15: Working Mesa loop with 1-Day playback.

By Feb 25: Completed 1-Week and 1-Month "Macro Drift" experiments.

March 9: FINAL SUBMISSION.
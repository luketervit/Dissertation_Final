from __future__ import annotations

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .client import OpenRouterClient


LEFT_VOCAB = [
    "MAGA",
    "GOP",
    "insurrection",
    "fascist",
    "authoritarian",
    "corrupt",
    "oligarch",
    "big oil",
]
RIGHT_VOCAB = [
    "woke",
    "radical left",
    "open borders",
    "deep state",
    "fake news",
    "socialism",
    "weaponized",
    "witch hunt",
]

EMOTION_BEHAVIOUR = {
    "joy": "mocking and sarcastic, celebrating your side while taunting opponents",
    "anger": "furious and combative, attacking opponents with sharp language",
    "sadness": "bitter and disillusioned, lamenting the state of the country",
    "fear": "alarmed and urgent, warning about threats to democracy or freedom",
    "surprise": "incredulous and shocked, calling out hypocrisy and double standards",
    "disgust": "contemptuous and scathing, expressing revulsion at opponents",
    "optimism": "rallying and defiant, pushing your side's agenda with confidence",
}

_STRIP_PATTERNS = [
    re.compile(
        r'^as\s+a\s+(left|right|center|conservative|liberal|progressive)'
        r'[\w\s-]*[,:]?\s*',
        re.IGNORECASE,
    ),
    re.compile(
        r'^[\w\s]*(assistant|bot|ai)\s*(replies|says|responds|writes|tweets)[:\s]*',
        re.IGNORECASE,
    ),
    re.compile(r'^\*[^*]+\*\s*', re.IGNORECASE),
    re.compile(r'^(reply|tweet|response)[:\s]+', re.IGNORECASE),
]


@dataclass(slots=True)
class AgentProfile:
    user_id: str
    display_name: str
    political_label: str
    aggression: float
    emotion: str


@dataclass(slots=True)
class SimulationPost:
    post_id: str
    author_id: str
    author_name: str
    text: str
    depth: int
    round_idx: int
    reply_to_id: str | None


def _aggression_tone(aggression: float) -> str:
    if aggression >= 0.7:
        return "confrontational and sharp"
    if aggression >= 0.5:
        return "combative and loaded"
    if aggression >= 0.3:
        return "assertive with some sarcasm"
    if aggression >= 0.15:
        return "opinionated but mostly civil"
    return "casual and conversational"


def _clean_response(text: str) -> str:
    text = (text or "").strip().strip('"\'').strip()
    for _ in range(3):
        for pattern in _STRIP_PATTERNS:
            text = pattern.sub("", text).strip()
    text = text.lstrip(",:;- ").strip()
    if len(text) > 280:
        text = text[:280].rsplit(" ", 1)[0]
    if len(text) < 12:
        return "That take falls apart pretty quickly."
    return text


def _build_system_prompt(agent: AgentProfile) -> str:
    if agent.political_label == "Left":
        ideology = (
            "You hold progressive political views and criticize conservative policies."
        )
        vocab = ", ".join(LEFT_VOCAB)
    elif agent.political_label == "Right":
        ideology = (
            "You hold conservative political views and criticize liberal policies."
        )
        vocab = ", ".join(RIGHT_VOCAB)
    else:
        ideology = "You hold moderate political views and dislike ideological extremes."
        vocab = "common sense, pragmatic, compromise"

    emotion_desc = EMOTION_BEHAVIOUR.get(
        agent.emotion.lower(),
        "passionate and opinionated in debate",
    )
    tone = _aggression_tone(agent.aggression)

    return f"""You write short tweets in a political argument on Twitter/X.

POLITICAL VIEWS: {ideology}
TONE: You are {tone}.
EMOTIONAL STYLE: You are {emotion_desc}.
VOCABULARY: Use words like: {vocab}

RULES:
1. Write only the tweet text.
2. Never reveal you are an AI or say you are playing a role.
3. Never start with "As a...".
4. Reference the topic being discussed.
5. Maximum 280 characters.
6. Some replies attack, some agree, some joke. Sound human."""


def _build_user_prompt(target_post: SimulationPost, thread_context: list[SimulationPost]) -> str:
    context_lines = [
        f'- "{post.author_name}: {post.text}"'
        for post in thread_context[-6:]
        if post.text
    ]
    context_text = "\n".join(context_lines) if context_lines else "(none)"
    return (
        "Recent posts in the thread:\n"
        f"{context_text}\n\n"
        "You are replying to this tweet:\n"
        f'"{target_post.text}"\n\n'
        "Write your reply tweet:"
    )


def load_simulation_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class OpenRouterThreadSimulation:
    """Minimal agent-based thread simulation backed by OpenRouter."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        client: OpenRouterClient,
        seed: int = 7,
    ) -> None:
        self.config = config
        self.client = client
        self.random = random.Random(seed)
        self.temperature = config.get("llm", {}).get("temperature", 0.9)
        self.max_tokens = config.get("llm", {}).get("max_tokens", 160)
        sim_cfg = config.get("simulation", {})
        self.max_posts_per_round = sim_cfg.get("max_posts_per_round", 2)
        requested_workers = sim_cfg.get("parallel_workers", 4)
        self.parallel_workers = max(
            1,
            min(
                requested_workers,
                self.max_posts_per_round,
                getattr(self.client, "max_concurrency", requested_workers),
            ),
        )

        root_cfg = config["root_post"]
        self.thread_history = [
            SimulationPost(
                post_id="root",
                author_id=root_cfg.get("author_id", "root_user"),
                author_name=root_cfg.get("author_name", "root_user"),
                text=root_cfg["text"],
                depth=0,
                round_idx=0,
                reply_to_id=None,
            )
        ]
        self.agents = [
            AgentProfile(**agent_cfg)
            for agent_cfg in config.get("agents", [])
        ]
        self.generation_log: list[dict[str, Any]] = []

    def _reply_probability(self, agent: AgentProfile) -> float:
        return max(0.08, min(0.32, 0.08 + agent.aggression * 0.18))

    def _pick_target(self) -> SimulationPost:
        recent = self.thread_history[-6:]
        return self.random.choice(recent)

    def _generate_reply(
        self,
        *,
        agent: AgentProfile,
        target_post: SimulationPost,
        round_idx: int,
        post_index: int,
    ) -> SimulationPost:
        result = self.client.chat(
            system_prompt=_build_system_prompt(agent),
            user_prompt=_build_user_prompt(target_post, self.thread_history),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = _clean_response(result.text)
        post = SimulationPost(
            post_id=f"r{round_idx}_p{post_index}",
            author_id=agent.user_id,
            author_name=agent.display_name,
            text=text,
            depth=min(target_post.depth + 1, 6),
            round_idx=round_idx,
            reply_to_id=target_post.post_id,
        )
        self.generation_log.append(
            {
                "post_id": post.post_id,
                "provider": result.provider,
                "model": result.model,
                "latency_seconds": result.latency_seconds,
                "usage": result.usage,
                "raw_id": result.raw_id,
                "agent": agent.display_name,
            }
        )
        return post

    def run(self, rounds: int) -> list[SimulationPost]:
        for round_idx in range(1, rounds + 1):
            selected_agents = [
                agent
                for agent in self.agents
                if self.random.random() <= self._reply_probability(agent)
            ]
            if not selected_agents:
                selected_agents = [self.random.choice(self.agents)]
            selected_agents = selected_agents[: self.max_posts_per_round]

            futures = []
            new_posts: list[SimulationPost] = []
            with ThreadPoolExecutor(max_workers=self.parallel_workers) as pool:
                for post_index, agent in enumerate(selected_agents, start=1):
                    target_post = self._pick_target()
                    futures.append(
                        pool.submit(
                            self._generate_reply,
                            agent=agent,
                            target_post=target_post,
                            round_idx=round_idx,
                            post_index=post_index,
                        )
                    )

                for future in as_completed(futures):
                    new_posts.append(future.result())

            new_posts.sort(key=lambda post: post.post_id)
            self.thread_history.extend(new_posts)

        return self.thread_history

    def export_results(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.client.model,
            "root_post": asdict(self.thread_history[0]),
            "posts": [asdict(post) for post in self.thread_history],
            "generation_log": self.generation_log,
            "agents": [asdict(agent) for agent in self.agents],
        }
        output_path = output_dir / "simulated_thread.json"
        output_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        return output_path

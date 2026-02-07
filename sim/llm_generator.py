"""
LLM wrapper for generating agent replies.
Supports OpenAI, Anthropic, Ollama, and mock mode.

Per-agent system prompts encode political persona behaviorally
(not as labels) to prevent the model from echoing its role.
"""
from __future__ import annotations

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / '.env')

# Political vocabulary banks for realistic tweet generation.
# Left-wing agents use progressive attack language; Right-wing agents
# use conservative attack language. Sourced from top keywords in real
# USC X 24 threads (validated against thread_001 and thread_002).
LEFT_VOCAB = [
    "MAGA", "GOP", "insurrection", "fascist", "authoritarian",
    "corrupt", "oligarch", "big oil", "voter suppression",
    "extremist", "white nationalist", "grifter", "sedition",
]
RIGHT_VOCAB = [
    "woke", "radical left", "open borders", "deep state",
    "fake news", "socialism", "defund", "indoctrination",
    "weaponized", "witch hunt", "hoax", "crooked",
]

# Emotion-to-behaviour mapping. The RoBERTa emotion model labels
# political schadenfreude as "joy", so we translate each label
# into the *actual Twitter behaviour* it represents.
EMOTION_BEHAVIOUR = {
    "joy": "mocking and sarcastic, celebrating your side's wins while taunting opponents",
    "anger": "furious and combative, attacking opponents with sharp language",
    "sadness": "bitter and disillusioned, lamenting the state of the country",
    "fear": "alarmed and urgent, warning about threats to democracy or freedom",
    "surprise": "incredulous and shocked, calling out hypocrisy and double standards",
    "disgust": "contemptuous and scathing, expressing revulsion at opponents",
    "optimism": "rallying and defiant, pushing your side's agenda with confidence",
}


def _aggression_tone(aggression: float) -> str:
    """Map continuous aggression score to behavioural tone description."""
    if aggression >= 0.7:
        return (
            "confrontational — you attack opponents directly "
            "and don't hold back"
        )
    if aggression >= 0.5:
        return (
            "combative — you challenge opponents harshly "
            "and use loaded language"
        )
    if aggression >= 0.3:
        return (
            "assertive — you push back firmly and use sarcasm"
        )
    if aggression >= 0.15:
        return (
            "opinionated but civil — you state your views clearly "
            "and occasionally push back"
        )
    return (
        "casual and conversational — you sometimes agree, sometimes "
        "disagree, and engage like a normal person scrolling Twitter"
    )


class LLMGenerator:
    """
    Wrapper for LLM-based text generation with per-agent persona prompts.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize LLM generator from config.

        Args:
            config: Dict with llm configuration.
        """
        self.provider = config['provider']
        self.model = config['model']
        self.temperature = config.get('temperature', 0.9)
        self.max_tokens = config.get('max_tokens', 150)

        # Initialize client based on provider
        if self.provider == "openai":
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                raise ValueError(
                    f"Environment variable {config['api_key_env']} not set"
                )
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                raise ValueError(
                    f"Environment variable {config['api_key_env']} not set"
                )
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "ollama":
            import requests
            # Use /api/chat for proper ChatML role separation
            self.ollama_url = "http://localhost:11434/api/chat"

            # Health check: verify Ollama is running and model is available
            try:
                health = requests.get(
                    "http://localhost:11434/api/tags", timeout=10
                )
                if health.status_code != 200:
                    raise ConnectionError(
                        f"Ollama returned status {health.status_code}"
                    )
                available_models = [
                    m['name'] for m in health.json().get('models', [])
                ]
                model_base = self.model.split(':')[0]
                if not any(model_base in m for m in available_models):
                    print(f"  WARNING: Model '{self.model}' not found.")
                    print(f"  Available: {available_models}")
                    print(f"  Run: ollama pull {self.model}")
                    raise ValueError(
                        f"Model '{self.model}' not available in Ollama"
                    )
                print(
                    f"  ✓ Ollama connected, model '{self.model}' available"
                )
            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    "Cannot connect to Ollama at localhost:11434. "
                    "Is Ollama running? Start it with: ollama serve"
                )

        elif self.provider == "mock":
            self.client = None

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_reply(
        self,
        agent_persona: dict,
        target_post: dict,
        thread_context: list,
        few_shot_examples: list[str] | None = None,
        thread_mean_aggression: float = 0.3,
    ) -> str:
        """
        Generate a reply from the agent's perspective.

        Args:
            agent_persona: Dict with political_label, aggression, emotion.
            target_post: Dict with the post being replied to.
            thread_context: List of recent posts for context.
            few_shot_examples: Real tweet texts from the thread for tone
                grounding.
            thread_mean_aggression: Mean aggression of agents in this
                thread, used to scale tone descriptions relative to the
                thread norm.

        Returns:
            Cleaned reply text (≤280 chars).
        """
        system_prompt = self._build_system_prompt(
            agent_persona, thread_mean_aggression
        )
        user_prompt = self._build_user_prompt(
            target_post, thread_context, few_shot_examples
        )

        # Generate based on provider
        if self.provider == "openai":
            raw = self._generate_openai(system_prompt, user_prompt)
        elif self.provider == "anthropic":
            raw = self._generate_anthropic(system_prompt, user_prompt)
        elif self.provider == "ollama":
            raw = self._generate_ollama(system_prompt, user_prompt)
        elif self.provider == "mock":
            return self._generate_mock(agent_persona, target_post)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

        cleaned = self._clean_response(raw)
        if cleaned is None:
            # Fallback: generation was garbage, return short political jab
            label = agent_persona.get('political_label', 'Center')
            if label == 'Left':
                return "The GOP is destroying this country."
            elif label == 'Right':
                return "Democrats are ruining America."
            else:
                return "Both sides need to do better."
        return cleaned

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self, agent_persona: dict, thread_mean_aggression: float = 0.3
    ) -> str:
        """
        Build a per-agent system prompt that encodes persona
        *behaviourally* without mentioning labels the model can echo.

        Aggression tone is scaled relative to *thread_mean_aggression*
        so agents adapt to the thread's actual hostility level rather
        than using absolute thresholds (fixes r = 0.20 correlation).
        """
        political = agent_persona.get('political_label', 'Center')
        aggression = agent_persona.get('aggression', 0.3)
        emotion = agent_persona.get('emotion', 'anger')

        # Political orientation described as values, not labels
        if political == 'Left':
            ideology = (
                "You hold progressive political views. You support "
                "social justice, government accountability, and are "
                "critical of conservative/Republican policies."
            )
            vocab = ", ".join(LEFT_VOCAB[:8])
        elif political == 'Right':
            ideology = (
                "You hold conservative political views. You support "
                "limited government, traditional values, and are "
                "critical of liberal/Democratic policies."
            )
            vocab = ", ".join(RIGHT_VOCAB[:8])
        else:
            ideology = (
                "You hold moderate political views. You criticise "
                "extremes on both sides and value pragmatic solutions."
            )
            vocab = "both sides, common sense, pragmatic, compromise"

        # Dynamic aggression: scale relative to thread norm so that
        # agents in mild threads produce milder output and agents in
        # toxic threads produce harsher output.
        relative_agg = aggression - thread_mean_aggression
        if relative_agg > 0.2:
            tone = _aggression_tone(aggression)  # above norm → full tier
        elif relative_agg > -0.1:
            # near norm → one tier softer than absolute would suggest
            tone = _aggression_tone(max(0.0, aggression - 0.15))
        else:
            # below norm → notably milder
            tone = _aggression_tone(max(0.0, aggression - 0.25))

        emotion_desc = EMOTION_BEHAVIOUR.get(
            emotion.lower(),
            "passionate and opinionated, engaging forcefully in debate",
        )

        system_prompt = f"""You write short tweets in a political argument on Twitter/X.

POLITICAL VIEWS: {ideology}
TONE: You are {tone}.
EMOTIONAL STYLE: You are {emotion_desc}.
VOCABULARY: Use words like: {vocab}

RULES — follow these EXACTLY:
1. Write ONLY the tweet text. No quotation marks around it.
2. NEVER reveal you are an AI, a bot, a language model, or playing a role.
3. NEVER start with "As a..." or mention your political leaning explicitly.
4. NEVER use words like "dialogue", "unity", "together", "both sides" unless you are Center.
5. Be specific — reference the topic being discussed, attack specific policies or people.
6. Not every reply is an attack. Sometimes you agree with someone, crack a joke, share a fact, or express genuine concern. Vary your tone naturally.
7. Maximum 280 characters. No hashtags unless relevant."""

        return system_prompt

    def _build_user_prompt(
        self,
        target_post: dict,
        thread_context: list,
        few_shot_examples: list[str] | None = None,
    ) -> str:
        """
        Build the user prompt containing few-shot examples from the
        real thread, recent simulated context, and the target tweet.

        Few-shot grounding anchors the LLM to the actual discourse
        style of the thread, preventing default-to-extreme-negativity.
        """
        parts: list[str] = []

        # Few-shot grounding from real thread tweets
        if few_shot_examples:
            example_lines = "\n".join(
                f'- "{ex}"' for ex in few_shot_examples
            )
            parts.append(
                "Here's how people are actually talking in this thread:\n"
                f"{example_lines}\n\n"
                "Match this tone and style. Some tweets attack, some "
                "agree, some joke."
            )

        # Show recent simulated context (expanded from 3 → 8)
        context_lines = []
        for p in thread_context[-8:]:
            text = p.get('text', '')
            if text:
                context_lines.append(f'- "{text}"')
        context_str = (
            "\n".join(context_lines) if context_lines else "(none)"
        )
        parts.append(f"Recent posts in the thread:\n{context_str}")

        target_text = target_post.get('text', '')
        parts.append(
            f'You are replying to this tweet:\n"{target_text}"\n\n'
            "Write your reply tweet:"
        )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    # Patterns to strip from generated text
    _STRIP_PATTERNS = [
        # "As a left-leaning..." / "As a right-leaning..."
        re.compile(
            r'^as\s+a\s+(left|right|center|conservative|liberal|progressive)'
            r'[\w\s-]*[,:]?\s*',
            re.IGNORECASE,
        ),
        # "Dolphin replies:" / "Joyful Left Leaning Dolphin replies:"
        re.compile(
            r'^[\w\s]*(dolphin|llama|assistant|bot|ai)\s*(replies|says|responds'
            r'|chimes in|writes|tweets)[:\s]*',
            re.IGNORECASE,
        ),
        # "*chimes in*" / "*responds angrily*" stage directions
        re.compile(r'^\*[^*]+\*\s*', re.IGNORECASE),
        # "Reply:" / "Tweet:" prefix
        re.compile(r'^(reply|tweet|response)[:\s]+', re.IGNORECASE),
    ]

    def _clean_response(self, text: str) -> str | None:
        """
        Post-process LLM output: strip meta-commentary, enforce
        280-char limit, reject garbage.

        Returns None if the cleaned text is too short (<20 chars).
        """
        if not text or text.startswith("["):
            return None

        # Strip surrounding quotes
        text = text.strip().strip('"\'').strip()

        # Apply stripping patterns iteratively (some stack)
        for _ in range(3):
            for pattern in self._STRIP_PATTERNS:
                text = pattern.sub('', text).strip()

        # Remove any remaining leading punctuation/whitespace
        text = text.lstrip(',:;-– ').strip()

        # Truncate to 280 chars at word boundary
        if len(text) > 280:
            text = text[:280].rsplit(' ', 1)[0]

        # Reject if too short after cleaning
        if len(text) < 20:
            return None

        return text

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _generate_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "[Error generating response]"

    def _generate_anthropic(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Generate using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return "[Error generating response]"

    def _generate_ollama(self, system_prompt: str, user_prompt: str) -> str:
        """Generate using local Ollama with /api/chat + thread-based hard timeout."""
        import requests
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        def _do_request() -> str:
            resp = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": 2048,
                        "repeat_penalty": 1.1,
                    },
                },
                timeout=(30, 120),
            )
            if resp.status_code != 200:
                return "[Error generating response]"
            message = resp.json().get('message', {})
            return message.get('content', '').strip()

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_do_request)
                result = future.result(timeout=90)
                return result if result else "[Error generating response]"
        except TimeoutError:
            print("Ollama 90s hard timeout — skipping agent")
            return "[Timeout]"
        except Exception as e:
            print(f"Ollama error: {e}")
            return "[Error generating response]"

    def _generate_mock(self, agent_persona: dict, target_post: dict) -> str:
        """Mock generator for testing without API."""
        political = agent_persona['political_label']
        aggression = agent_persona['aggression']
        target_political = target_post.get('political_label', 'Center')

        if political == target_political:
            return "Exactly right. They don't want you to see the truth."
        elif aggression > 0.5:
            return "This is completely wrong and you know it."
        else:
            return "That's not how any of this works."

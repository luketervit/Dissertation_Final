"""
LLM wrapper for generating agent replies.
Supports OpenAI, Anthropic, Ollama, and mock mode.
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / '.env')


class LLMGenerator:
    """
    Wrapper for LLM-based text generation.
    """

    def __init__(self, config):
        """
        Initialize LLM generator from config.

        Args:
            config: Dict with llm configuration
        """
        self.provider = config['provider']
        self.model = config['model']
        self.temperature = config['temperature']
        self.max_tokens = config['max_tokens']
        self.system_prompt = config['system_prompt']

        # Initialize client based on provider
        if self.provider == "openai":
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                raise ValueError(f"Environment variable {config['api_key_env']} not set")

            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "anthropic":
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                raise ValueError(f"Environment variable {config['api_key_env']} not set")

            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)

        elif self.provider == "ollama":
            # Local Ollama - no API key needed
            import requests
            self.ollama_url = "http://localhost:11434/api/generate"

        elif self.provider == "mock":
            # Mock mode for testing without API calls
            self.client = None

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def generate_reply(self, agent_persona, target_post, thread_context):
        """
        Generate a reply from the agent's perspective.

        Args:
            agent_persona: Dict with political_label, aggression, emotion
            target_post: Dict with the post being replied to
            thread_context: List of recent posts for context

        Returns:
            Generated reply text
        """
        # Build prompt
        prompt = self._build_prompt(agent_persona, target_post, thread_context)

        # Generate based on provider
        if self.provider == "openai":
            return self._generate_openai(prompt)
        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt)
        elif self.provider == "ollama":
            return self._generate_ollama(prompt)
        elif self.provider == "mock":
            return self._generate_mock(agent_persona, target_post)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _build_prompt(self, agent_persona, target_post, thread_context):
        """Build prompt for LLM."""
        # Agent persona
        political = agent_persona['political_label']
        aggression = agent_persona['aggression']
        emotion = agent_persona['emotion']

        # Personality description (CONSERVATIVE DEFAULTS for generalization)
        if aggression > 0.5:  # Original baseline threshold
            tone = "aggressive and confrontational"
        elif aggression > 0.3:  # Original baseline threshold
            tone = "assertive and direct"
        else:
            tone = "polite and measured"

        persona_desc = f"You are a {political}-leaning Twitter user. Your tone is {tone}. Your dominant emotion is {emotion}."

        # Thread context
        context_str = "\n".join([
            f"- {p.get('political_label', 'Unknown')}: \"{p['text']}\""
            for p in thread_context[-3:]  # Last 3 posts
        ])

        # Target post
        target_text = target_post['text']
        target_political = target_post.get('political_label', 'Unknown')

        # Full prompt
        prompt = f"""{persona_desc}

You're reading a political discussion thread. Here's the recent context:
{context_str}

You're replying to this {target_political} post:
"{target_text}"

Write your reply (1-3 sentences, under 280 characters). Stay in character:"""

        return prompt

    def _generate_openai(self, prompt):
        """Generate using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"[Error generating response]"

    def _generate_anthropic(self, prompt):
        """Generate using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return f"[Error generating response]"

    def _generate_ollama(self, prompt):
        """Generate using local Ollama."""
        try:
            import requests
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": f"{self.system_prompt}\n\n{prompt}",
                    "temperature": self.temperature,
                    "stream": False
                },
                timeout=60  # 60 second timeout
            )
            if response.status_code != 200:
                print(f"Ollama HTTP error {response.status_code}: {response.text}")
                return "[Error generating response]"

            result = response.json()
            if 'response' not in result:
                print(f"Ollama response missing 'response' field: {result}")
                return "[Error generating response]"

            return result['response'].strip()
        except requests.exceptions.Timeout:
            print(f"Ollama timeout after 60s - model may be too slow")
            return "[Timeout]"
        except Exception as e:
            print(f"Ollama error: {e}")
            return f"[Error generating response]"

    def _generate_mock(self, agent_persona, target_post):
        """Mock generator for testing without API."""
        political = agent_persona['political_label']
        aggression = agent_persona['aggression']

        # Simple rule-based mock
        target_political = target_post.get('political_label', 'Center')

        if political == target_political:
            return "I agree with your perspective on this."
        elif aggression > 0.5:
            return "I completely disagree. This view is fundamentally flawed."
        else:
            return "I see this differently, but I understand your point."

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openrouter_runtime.client import OpenRouterClient


def main() -> None:
    client = OpenRouterClient.from_env(PROJECT_ROOT)
    status = client.key_status()
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

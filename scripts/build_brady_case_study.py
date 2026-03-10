"""
Build a Brady et al. (2017) moral-emotional one-word counterfactual case study.

This script prepares everything needed to run your existing thread simulation
pipeline on paired conditions:
  - control: original tweet text
  - edited: same tweet with exactly one lexical change

Outputs:
  case_study/brady_mec/
    threads/thread_XXX/{config.yaml,thread_metadata.json,agents_for_thread.csv}
    manifests/{family_manifest.csv,thread_manifest.csv,summary.json,run_commands.sh}
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yaml


OSF_FILES: dict[str, str] = {
    # Raw tweet/topic data
    "C_raw.csv": "https://osf.io/download/v6ct2/",
    "G_raw.csv": "https://osf.io/download/dztav/",
    "M_raw.csv": "https://osf.io/download/urpfh/",
    # Dictionary files for one-word interventions
    "shared.txt": "https://osf.io/download/fq5wg/",
    "shared_pos.txt": "https://osf.io/download/erd9j/",
    "moral_unique.txt": "https://osf.io/download/chyre/",
    "pos_unique.txt": "https://osf.io/download/t7hkc/",
    "neg_unique.txt": "https://osf.io/download/28qjs/",
    "moral_emo/Moral.txt": "https://osf.io/download/gqu49/",
    "moral_emo/Affect.txt": "https://osf.io/download/k3wnz/",
    "moral_emo_pos/Affect_pos.txt": "https://osf.io/download/pe2yk/",
    "moral_emo_neg/Affect_neg.txt": "https://osf.io/download/4uqhp/",
    # Preprocessed files used in the original paper pipeline (optional baseline)
    "MEC_SASpreproc_Climate.csv": "https://osf.io/download/hp8zc/",
    "MEC_SASpreproc_Gun.csv": "https://osf.io/download/42pr6/",
    "MEC_SASpreproc_Marriage.csv": "https://osf.io/download/gbf5c/",
    "MEC_SAS_ingroup_Climate.csv": "https://osf.io/download/tqcyn/",
    "MEC_SAS_ingroup_Gun.csv": "https://osf.io/download/dbjma/",
    "MEC_SAS_ingroup_Marriage.csv": "https://osf.io/download/fcxzy/",
}

TOPIC_FILES = {
    "climate": "C_raw.csv",
    "gun": "G_raw.csv",
    "marriage": "M_raw.csv",
}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")

# Curated moral-emotional lexemes for natural one-word interventions.
CURATED_POS_WORDS = [
    "care",
    "caring",
    "compassion",
    "devotion",
    "faith",
    "good",
    "goodness",
    "heaven",
    "hero",
    "honor",
    "honesty",
    "ideal",
    "loyal",
    "peace",
    "respect",
    "safe",
    "save",
    "security",
    "virtue",
    "benefit",
    "value",
]

CURATED_NEG_WORDS = [
    "abuse",
    "attack",
    "bad",
    "brutal",
    "cheat",
    "cruel",
    "damage",
    "damn",
    "destroy",
    "devil",
    "disgust",
    "envy",
    "evil",
    "fault",
    "fight",
    "forbid",
    "greed",
    "gross",
    "hate",
    "hell",
    "hurt",
    "immoral",
    "kill",
    "liar",
    "murder",
    "pain",
    "protest",
    "sin",
    "sinister",
    "vile",
    "war",
]

NEG_TO_POS_SWAP_MAP = {
    "bad": "good",
    "hate": "care",
    "hell": "heaven",
    "pain": "care",
    "protest": "respect",
    "sin": "virtue",
    "sins": "virtue",
    "sinister": "honest",
    "vile": "good",
    "war": "peace",
    "wars": "peace",
    "warring": "peaceful",
}

POS_TO_NEG_SWAP_MAP = {
    "good": "bad",
    "goodness": "evil",
    "care": "hate",
    "caring": "hateful",
    "respect": "contempt",
    "save": "hurt",
}


@dataclass
class EditOutcome:
    operation: str
    edited_text: str
    lexeme_before: str
    lexeme_after: str
    success: bool


def download_file(url: str, dest: Path, overwrite: bool) -> None:
    if dest.exists() and not overwrite:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_osf_payload(raw_dir: Path, overwrite: bool) -> None:
    print(f"Downloading Brady case-study source files to: {raw_dir}")
    for rel_path, url in OSF_FILES.items():
        out_path = raw_dir / rel_path
        print(f"  - {rel_path}")
        download_file(url, out_path, overwrite=overwrite)


def read_lexicon(path: Path) -> list[str]:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip().lower()
        if token:
            lines.append(token)
    return lines


def compile_patterns(lexicon: Iterable[str]) -> list[tuple[str, re.Pattern[str]]]:
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for item in lexicon:
        if item.endswith("*"):
            stem = re.escape(item[:-1])
            compiled.append((item, re.compile(rf"^{stem}[a-z']*$")))
        else:
            compiled.append((item, re.compile(rf"^{re.escape(item)}$")))
    return compiled


def exact_lexemes(lexicon: Iterable[str], min_len: int = 3) -> set[str]:
    out: set[str] = set()
    for item in lexicon:
        token = item.strip().lower()
        if not token or "*" in token:
            continue
        if re.fullmatch(r"[a-z][a-z']*", token) and len(token) >= min_len:
            out.add(token)
    return out


def clean_insert_candidates(lexicon: Iterable[str], min_len: int = 4) -> list[str]:
    stop = {
        "also",
        "just",
        "very",
        "many",
        "much",
        "more",
        "most",
        "make",
        "made",
        "want",
        "need",
        "less",
    }
    out = []
    for item in lexicon:
        if "*" in item:
            continue
        if re.fullmatch(r"[a-z][a-z']*", item) and len(item) >= min_len and item not in stop:
            out.append(item)
    return sorted(set(out))


def find_first_exact_word(text: str, lexemes: set[str]) -> re.Match[str] | None:
    if not lexemes:
        return None
    for m in WORD_RE.finditer(text):
        if m.group(0).lower() in lexemes:
            return m
    return None


def find_first_match(
    text: str, patterns: list[tuple[str, re.Pattern[str]]]
) -> tuple[re.Match[str], str] | None:
    for m in WORD_RE.finditer(text):
        word = m.group(0).lower()
        for pat_raw, pat_re in patterns:
            if pat_re.match(word):
                return m, pat_raw
    return None


def count_matches(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> int:
    count = 0
    for m in WORD_RE.finditer(text):
        word = m.group(0).lower()
        if any(pat_re.match(word) for _, pat_re in patterns):
            count += 1
    return count


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def preserve_case(source_word: str, replacement: str) -> str:
    if source_word.isupper():
        return replacement.upper()
    if source_word[:1].isupper():
        return replacement.capitalize()
    return replacement


def add_word(text: str, word: str) -> str:
    end_punc = re.search(r"([.!?])\s*$", text)
    if end_punc:
        idx = end_punc.start(1)
        new_text = f"{text[:idx].rstrip()} {word}{text[idx:]}"
    else:
        new_text = f"{text.rstrip()} {word}"
    return normalize_spacing(new_text)


def remove_span(text: str, start: int, end: int) -> str:
    return normalize_spacing(text[:start] + text[end:])


def replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return normalize_spacing(text[:start] + replacement + text[end:])


def map_ideology_to_label(ideology: float) -> str:
    if ideology <= -0.2:
        return "Left"
    if ideology >= 0.2:
        return "Right"
    return "Center"


def topic_short(topic: str) -> str:
    return {"climate": "C", "gun": "G", "marriage": "M"}[topic]


def pick_replacement(candidates: list[str], rng: random.Random, avoid: set[str]) -> str:
    pool = [c for c in candidates if c not in avoid]
    if not pool:
        pool = candidates
    if not pool:
        raise ValueError("No replacement candidates available")
    return rng.choice(pool)


def choose_primary_operation(
    text: str,
    shared_exact: set[str],
    shared_pos_exact: set[str],
    shared_neg_exact: set[str],
) -> str:
    if find_first_exact_word(text, shared_pos_exact):
        return "swap_pos_to_neg"
    if find_first_exact_word(text, shared_neg_exact):
        return "swap_neg_to_pos"
    return "add_shared_pos"


def apply_primary_edit(
    text: str,
    operation: str,
    shared_exact: set[str],
    shared_pos_exact: set[str],
    shared_neg_exact: set[str],
    pos_candidates: list[str],
    neg_candidates: list[str],
    rng: random.Random,
) -> EditOutcome:
    lowered_words = {m.group(0).lower() for m in WORD_RE.finditer(text)}

    if operation == "add_shared_pos":
        inserted = pick_replacement(pos_candidates, rng, lowered_words)
        edited = add_word(text, inserted)
        return EditOutcome(operation, edited, "", inserted, edited != text)

    if operation == "add_shared_neg":
        inserted = pick_replacement(neg_candidates, rng, lowered_words)
        edited = add_word(text, inserted)
        return EditOutcome(operation, edited, "", inserted, edited != text)

    if operation == "swap_pos_to_neg":
        m = find_first_exact_word(text, shared_pos_exact)
        if m is None:
            return EditOutcome(operation, text, "", "", False)
        source = m.group(0)
        replacement_word = POS_TO_NEG_SWAP_MAP.get(source.lower())
        if replacement_word is None:
            replacement_word = pick_replacement(neg_candidates, rng, lowered_words)
        replacement = preserve_case(source, replacement_word)
        edited = replace_span(text, m.start(), m.end(), replacement)
        return EditOutcome(operation, edited, m.group(0), replacement, edited != text)

    if operation == "swap_neg_to_pos":
        m = find_first_exact_word(text, shared_neg_exact)
        if m is None:
            return EditOutcome(operation, text, "", "", False)
        source = m.group(0)
        replacement_word = NEG_TO_POS_SWAP_MAP.get(source.lower())
        if replacement_word is None:
            replacement_word = pick_replacement(pos_candidates, rng, lowered_words)
        replacement = preserve_case(source, replacement_word)
        edited = replace_span(text, m.start(), m.end(), replacement)
        return EditOutcome(operation, edited, m.group(0), replacement, edited != text)

    return EditOutcome(operation, text, "", "", False)


def build_alteration_set(
    text: str,
    n_alterations: int,
    shared_exact: set[str],
    shared_pos_exact: set[str],
    shared_neg_exact: set[str],
    pos_candidates: list[str],
    neg_candidates: list[str],
    rng: random.Random,
) -> list[EditOutcome]:
    # Ordered to maximize interpretability for the case study:
    # 1) add a positive moral-emotional token
    # 2) de-escalate a negative token if present
    # 3) escalate by injecting/using a negative token
    op_plan = [
        "add_shared_pos",
        "swap_neg_to_pos",
        "swap_pos_to_neg",
        "add_shared_neg",
        "add_shared_pos",
        "add_shared_neg",
    ]
    seen_texts = {normalize_spacing(text)}
    seen_changes: set[tuple[str, str, str]] = set()
    out: list[EditOutcome] = []

    for op in op_plan:
        if len(out) >= n_alterations:
            break
        edit = apply_primary_edit(
            text=text,
            operation=op,
            shared_exact=shared_exact,
            shared_pos_exact=shared_pos_exact,
            shared_neg_exact=shared_neg_exact,
            pos_candidates=pos_candidates,
            neg_candidates=neg_candidates,
            rng=rng,
        )
        if not edit.success:
            continue
        edited_norm = normalize_spacing(edit.edited_text)
        if edited_norm in seen_texts:
            continue
        key = (
            edit.operation,
            edit.lexeme_before.lower(),
            edit.lexeme_after.lower(),
        )
        if key in seen_changes:
            continue
        out.append(edit)
        seen_texts.add(edited_norm)
        seen_changes.add(key)

    if len(out) < n_alterations:
        return []
    return out[:n_alterations]


def row_is_usable(text: str, min_words: int, max_words: int) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 15:
        return False
    if text.startswith("RT @"):
        return False
    n_words = len(WORD_RE.findall(text))
    if n_words < min_words or n_words > max_words:
        return False
    return True


def read_topic_raw(path: Path) -> pd.DataFrame:
    usecols = [
        "id_str",
        "timestamp",
        "retweeted_status.id_str",
        "user.id_str",
        "user.screen_name",
        "text",
        "ideology",
        "user.followers_count",
        "user.verified",
    ]
    try:
        df = pd.read_csv(path, usecols=usecols)
    except ValueError:
        df = pd.read_csv(path)
        missing = [c for c in usecols if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}") from None
        df = df[usecols]
    return df


def balanced_sample_by_label(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    by_label = {}
    for label in ["Left", "Center", "Right"]:
        sub = df[df["label"] == label].copy()
        idx = list(sub.index)
        rng.shuffle(idx)
        by_label[label] = idx

    selected: list[int] = []
    while len(selected) < n and any(by_label.values()):
        for label in ["Left", "Center", "Right"]:
            if by_label[label]:
                selected.append(by_label[label].pop(0))
                if len(selected) >= n:
                    break
    return df.loc[selected].copy()


def build_agent_pool(topic_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        topic_df.dropna(subset=["user.id_str", "ideology"])
        .groupby("user.id_str", as_index=False)
        .agg(
            ideology=("ideology", "mean"),
            followers=("user.followers_count", "max"),
            verified=("user.verified", "max"),
            username=("user.screen_name", "first"),
            tweet_count=("id_str", "count"),
        )
    )

    rows = []
    for _, row in grouped.iterrows():
        ideology = float(row["ideology"])
        label = map_ideology_to_label(ideology)
        extremity = min(2.5, abs(ideology))
        base_aggr = min(0.85, 0.08 + 0.16 * extremity)
        verified_val = row.get("verified", False)
        is_verified = str(verified_val).strip().lower() in {"1", "true", "t", "yes"}
        if is_verified:
            base_aggr = max(0.03, base_aggr - 0.03)

        if extremity >= 1.1:
            emotion = "anger"
            sentiment = "negative"
        elif extremity >= 0.55:
            emotion = "sadness"
            sentiment = "negative"
        else:
            emotion = "optimism"
            sentiment = "neutral"

        rows.append(
            {
                "user_id": str(row["user.id_str"]),
                "username": str(row.get("username", "")),
                "political_label": label,
                "political_score": round(min(1.0, abs(ideology) / 2.0), 4),
                "emotion_label": emotion,
                "emotion_score": 0.7 if emotion == "anger" else 0.58,
                "sentiment_label": sentiment,
                "sentiment_score": 0.66 if sentiment == "negative" else 0.55,
                "hate_score": round(min(0.5, base_aggr * 0.55), 4),
                "offensive_score": round(min(0.5, base_aggr * 0.45), 4),
                "role": "active",
                "tweet_count": int(row.get("tweet_count", 1)),
                "view_count": int(max(0, row.get("followers", 0) or 0)),
                "reply_count": 0,
            }
        )

    out = pd.DataFrame(rows)
    return out.drop_duplicates(subset=["user_id"]).reset_index(drop=True)


def sample_agents(agent_pool: pd.DataFrame, n_agents: int, seed: int) -> pd.DataFrame:
    if len(agent_pool) == 0:
        raise ValueError("Agent pool is empty; cannot sample agents.")
    rng = random.Random(seed)
    indices = list(agent_pool.index)
    rng.shuffle(indices)
    take = min(n_agents, len(indices))
    return agent_pool.loc[indices[:take]].copy().reset_index(drop=True)


def load_default_llm(project_root: Path) -> dict:
    config_path = project_root / "config" / "thread_config.yaml"
    if not config_path.exists():
        return {
            "provider": "ollama",
            "model": "dolphin-llama3:8b",
            "api_key_env": None,
            "temperature": 0.9,
            "max_tokens": 150,
        }
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    llm = cfg.get("llm", {})
    llm.setdefault("provider", "ollama")
    llm.setdefault("model", "dolphin-llama3:8b")
    llm.setdefault("api_key_env", None)
    llm.setdefault("temperature", 0.9)
    llm.setdefault("max_tokens", 150)
    return llm


def write_thread_package(
    thread_dir: Path,
    thread_num: int,
    family_id: str,
    condition: str,
    topic: str,
    source_row: pd.Series,
    text: str,
    edit: EditOutcome | None,
    agents_df: pd.DataFrame,
    llm_cfg: dict,
    max_rounds: int,
) -> None:
    thread_dir.mkdir(parents=True, exist_ok=True)

    agents_path = thread_dir / "agents_for_thread.csv"
    agents_df.to_csv(agents_path, index=False)

    tweet_id = str(source_row["id_str"])
    user_id = str(source_row["user.id_str"])
    metadata = {
        "root_tweet": {
            "id": tweet_id,
            "conversation_id": tweet_id,
            "user_id": user_id,
            "text": text,
            "expected_replies": 0,
            "timestamp": str(source_row.get("timestamp", "")),
            "view_count": int(max(0, source_row.get("user.followers_count", 0) or 0)),
        },
        "replies": {"actual_count": 0, "tweet_ids": []},
        "temporal_events": [
            {
                "tweet_id": tweet_id,
                "user_id": user_id,
                "timestamp": str(source_row.get("timestamp", "")),
                "epoch": 0,
                "seconds_since_start": 0,
                "is_root": True,
                "text": text,
            }
        ],
        "source": "brady_mec_case_study",
        "case_study": {
            "family_id": family_id,
            "condition": condition,
            "topic": topic,
            "operation": "control" if edit is None else edit.operation,
            "lexeme_before": "" if edit is None else edit.lexeme_before,
            "lexeme_after": "" if edit is None else edit.lexeme_after,
            "base_tweet_id": tweet_id,
            "base_user_id": user_id,
        },
    }

    metadata_path = thread_dir / "thread_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    config = {
        "target_tweet_id": tweet_id,
        "paths": {
            "thread_metadata": "thread_metadata.json",
            "agents_for_thread": "agents_for_thread.csv",
        },
        "abm": {
            "lurker_ratio": 0.0,
            "lurker_dist_strategy": "match_active_agents",
            "bounded_confidence_threshold": 0.3,
            "backfire_threshold": 0.6,
            "backfire_aggression_min": 0.7,
        },
        "llm": llm_cfg,
        "simulation": {
            "thread_num": thread_num,
            "max_rounds": max_rounds,
            "use_few_shot": False,
        },
    }

    config_path = thread_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def prepare_case_study(
    project_root: Path,
    raw_dir: Path,
    output_root: Path,
    n_per_topic: int,
    n_alterations: int,
    n_agents: int,
    seed: int,
    min_words: int,
    max_words: int,
    max_rounds: int,
    run_models: list[str],
    overwrite: bool,
) -> None:
    topic_seed_offset = {"climate": 101, "gun": 211, "marriage": 307}

    if output_root.exists() and overwrite:
        for child in output_root.iterdir():
            if child.is_file():
                child.unlink()
            else:
                import shutil

                shutil.rmtree(child)
    elif output_root.exists() and any(output_root.iterdir()):
        raise ValueError(
            f"Output root already contains files: {output_root}. "
            "Use --overwrite to regenerate."
        )

    threads_root = output_root / "threads"
    manifest_root = output_root / "manifests"
    threads_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    lex_shared = read_lexicon(raw_dir / "shared.txt")
    lex_shared_pos = read_lexicon(raw_dir / "shared_pos.txt")
    lex_neg_unique = read_lexicon(raw_dir / "neg_unique.txt")

    shared_patterns = compile_patterns(lex_shared)
    shared_pos_patterns = compile_patterns(lex_shared_pos)
    lex_shared_neg = sorted(set(lex_shared) - set(lex_shared_pos))
    shared_neg_patterns = compile_patterns(lex_shared_neg + lex_neg_unique)

    shared_exact = exact_lexemes(lex_shared, min_len=3)
    shared_pos_exact = exact_lexemes(lex_shared_pos, min_len=3)
    shared_neg_exact = exact_lexemes(lex_shared_neg, min_len=3)

    pos_candidates = clean_insert_candidates(CURATED_POS_WORDS, min_len=4)
    neg_candidates = clean_insert_candidates(CURATED_NEG_WORDS, min_len=4)

    if not pos_candidates or not neg_candidates:
        raise ValueError("Could not derive positive/negative replacement candidates.")

    llm_cfg = load_default_llm(project_root)

    family_rows = []
    thread_rows = []
    family_counter = 1
    thread_counter = 1

    for topic, filename in TOPIC_FILES.items():
        topic_path = raw_dir / filename
        if not topic_path.exists():
            raise FileNotFoundError(f"Missing required topic file: {topic_path}")
        print(f"Loading topic '{topic}' from {topic_path}")
        df = read_topic_raw(topic_path)

        df = df.dropna(subset=["text", "id_str", "user.id_str", "ideology"]).copy()
        df["text"] = df["text"].astype(str)
        df = df[df["text"].map(lambda x: row_is_usable(x, min_words, max_words))]
        df = df.drop_duplicates(subset=["text"]).copy()
        df["label"] = df["ideology"].map(map_ideology_to_label)
        df["has_shared"] = df["text"].map(lambda t: count_matches(t, shared_patterns) > 0)
        df["has_shared_pos"] = df["text"].map(lambda t: count_matches(t, shared_pos_patterns) > 0)
        df["has_shared_neg"] = df["text"].map(lambda t: count_matches(t, shared_neg_patterns) > 0)

        sampled = balanced_sample_by_label(
            df,
            n_per_topic,
            seed + topic_seed_offset[topic],
        )
        agent_pool = build_agent_pool(df)
        if len(agent_pool) == 0:
            raise ValueError(f"No agents found for topic '{topic}'")

        for _, row in sampled.iterrows():
            family_id = f"F{family_counter:04d}"
            local_seed = seed + int(str(row["id_str"])[-6:])
            rng = random.Random(local_seed)
            edits = build_alteration_set(
                text=row["text"],
                n_alterations=n_alterations,
                shared_exact=shared_exact,
                shared_pos_exact=shared_pos_exact,
                shared_neg_exact=shared_neg_exact,
                pos_candidates=pos_candidates,
                neg_candidates=neg_candidates,
                rng=rng,
            )
            if len(edits) < n_alterations:
                continue

            agents_df = sample_agents(agent_pool, n_agents=n_agents, seed=local_seed)

            control_row = {
                "id_str": row["id_str"],
                "user.id_str": row["user.id_str"],
                "timestamp": row.get("timestamp", ""),
                "user.followers_count": row.get("user.followers_count", 0),
            }
            for condition, tweet_text, edit in [("control", row["text"], None)] + [
                (f"edited_{i}", e.edited_text, e) for i, e in enumerate(edits, start=1)
            ]:
                thread_name = f"thread_{thread_counter:03d}"
                thread_dir = threads_root / thread_name
                write_thread_package(
                    thread_dir=thread_dir,
                    thread_num=thread_counter,
                    family_id=family_id,
                    condition=condition,
                    topic=topic,
                    source_row=pd.Series(control_row),
                    text=tweet_text,
                    edit=edit,
                    agents_df=agents_df,
                    llm_cfg=llm_cfg,
                    max_rounds=max_rounds,
                )
                thread_rows.append(
                    {
                        "thread_id": thread_name,
                        "thread_num": thread_counter,
                        "family_id": family_id,
                        "condition": condition,
                        "topic": topic,
                        "operation": "control" if condition == "control" else edit.operation,
                        "lexeme_before": "" if condition == "control" else edit.lexeme_before,
                        "lexeme_after": "" if condition == "control" else edit.lexeme_after,
                        "base_tweet_id": str(row["id_str"]),
                        "base_user_id": str(row["user.id_str"]),
                        "tweet_text": tweet_text,
                        "agents_count": len(agents_df),
                        "thread_dir": str(thread_dir),
                    }
                )
                thread_counter += 1

            for i, edit in enumerate(edits, start=1):
                family_rows.append(
                    {
                        "family_id": family_id,
                        "edit_id": f"edited_{i}",
                        "topic": topic,
                        "topic_short": topic_short(topic),
                        "operation": edit.operation,
                        "lexeme_before": edit.lexeme_before,
                        "lexeme_after": edit.lexeme_after,
                        "base_tweet_id": str(row["id_str"]),
                        "base_user_id": str(row["user.id_str"]),
                        "base_text": row["text"],
                        "edited_text": edit.edited_text,
                        "agents_count": len(agents_df),
                    }
                )

            family_counter += 1

    family_df = pd.DataFrame(family_rows)
    thread_df = pd.DataFrame(thread_rows)
    if len(family_df) == 0 or len(thread_df) == 0:
        raise ValueError("No case-study families were created. Relax filters or check input data.")

    family_df.to_csv(manifest_root / "family_manifest.csv", index=False)
    thread_df.to_csv(manifest_root / "thread_manifest.csv", index=False)

    total_threads = int(thread_df["thread_num"].max())
    operations_summary = family_df["operation"].value_counts().to_dict()

    try:
        rel_threads_input = output_root.relative_to(project_root) / "threads"
        rel_results_root = output_root.relative_to(project_root) / "results"
    except ValueError:
        rel_threads_input = output_root / "threads"
        rel_results_root = output_root / "results"

    run_cmds = []
    for model_key in run_models:
        run_cmds.append(
            f"""python scripts/run_model_comparison_batch.py \\
  --model {model_key} \\
  --input {rel_threads_input} \\
  --reference '' \\
  --expected-threads {total_threads} \\
  --start 1 --end {total_threads} \\
  --output-root {rel_results_root} \\
  --max-rounds {max_rounds}
"""
        )

    run_script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "# Brady MEC case-study simulation commands (generated).\n\n"
        + "\n".join(run_cmds)
    )
    run_script_path = manifest_root / "run_commands.sh"
    run_script_path.write_text(run_script, encoding="utf-8")
    run_script_path.chmod(0o755)

    summary = {
        "families": int(len(family_df)),
        "threads": int(len(thread_df)),
        "topics": sorted(thread_df["topic"].unique().tolist()),
        "operations": operations_summary,
        "output_root": str(output_root),
        "threads_root": str(threads_root),
        "manifest_root": str(manifest_root),
        "recommended_run_script": str(run_script_path),
    }
    with open(manifest_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Case-study inputs prepared:")
    print(f"  Families: {summary['families']}")
    print(f"  Threads:  {summary['threads']}")
    print(f"  Output:   {output_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Brady moral-emotional one-word case-study inputs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("download", "prepare", "all"):
        p = sub.add_parser(name)
        p.add_argument(
            "--raw-dir",
            type=Path,
            default=Path("data/case_study/brady/raw"),
            help="Directory for downloaded Brady files.",
        )
        p.add_argument(
            "--output-root",
            type=Path,
            default=Path("case_study/brady_mec"),
            help="Case-study output root directory.",
        )
        p.add_argument("--seed", type=int, default=42)
        p.add_argument(
            "--n-per-topic",
            type=int,
            default=20,
            help="Number of base tweet families sampled per topic.",
        )
        p.add_argument(
            "--agents-per-family",
            type=int,
            default=120,
            help="Number of agents sampled per family.",
        )
        p.add_argument("--min-words", type=int, default=8)
        p.add_argument("--max-words", type=int, default=45)
        p.add_argument("--max-rounds", type=int, default=12)
        p.add_argument(
            "--n-alterations",
            type=int,
            default=3,
            help="Number of one-word edited variants per base tweet.",
        )
        p.add_argument(
            "--run-models",
            nargs="+",
            default=["llama"],
            choices=["llama", "qwen", "deepseek_r1", "deepseek_llm"],
            help="Model presets included in generated run_commands.sh.",
        )
        p.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing downloaded files / output directory contents.",
        )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = (project_root / args.raw_dir).resolve()
    output_root = (project_root / args.output_root).resolve()

    if args.command in {"download", "all"}:
        download_osf_payload(raw_dir=raw_dir, overwrite=args.overwrite)

    if args.command in {"prepare", "all"}:
        prepare_case_study(
            project_root=project_root,
            raw_dir=raw_dir,
            output_root=output_root,
            n_per_topic=args.n_per_topic,
            n_agents=args.agents_per_family,
            seed=args.seed,
            min_words=args.min_words,
            max_words=args.max_words,
            max_rounds=args.max_rounds,
            run_models=args.run_models,
            n_alterations=args.n_alterations,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()

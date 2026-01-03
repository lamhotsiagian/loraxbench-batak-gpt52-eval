#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

eval_csv_mcqa_gpt52.py  (FULL, FIXED)

Evaluates GPT-5.2 on an MCQA-style CSV (like your bbc cultural_qa subset).
Takes the FIRST N rows in sheet order (no random sampling by default).

Author Lamhot Siagian

Expected CSV columns:
  - language
  - context
  - option A
  - option B
  - option C
  - answer   (A/B/C)

Outputs:
  - runs/csv_mcqa_gpt52_firstN/summary.json
  - runs/csv_mcqa_gpt52_firstN/predictions.csv
  - runs/csv_mcqa_gpt52_firstN/predictions.jsonl

Metrics:
  - Accuracy (primary)
  - BLEU, METEOR, ROUGE-1/2/L, BERTScore-F1 (on predicted option text vs gold option text)
  - COMET optional (exploratory): uses context as "src"

Setup:
  pip install -U openai pandas sacrebleu evaluate bert-score torch python-dotenv
  # Optional COMET:
  pip install -U unbabel-comet

Environment (.env supported):
  OPENAI_API_KEY=...

Run:
  python eval_csv_mcqa_gpt52.py --csv_path "bbc-cultural-qa-subset.csv" --limit 10
  python eval_csv_mcqa_gpt52.py --csv_path "bbc-cultural-qa-subset.csv" --limit 10 --comet
"""

import argparse
import csv as csvlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
import sacrebleu
import evaluate
from bert_score import score as bert_score
from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------
# Load .env (OPENAI_API_KEY)
# -----------------------------
load_dotenv()


# -----------------------------
# OpenAI call
# -----------------------------
@dataclass
class ModelCfg:
    model: str = "gpt-5.2"
    reasoning_effort: str = "none"
    verbosity: str = "low"
    temperature: float = 0.0


def _backoff(attempt: int) -> None:
    base = min(2 ** attempt, 32)
    time.sleep(base + random.random())


def call_gpt52(client: OpenAI, prompt: str, cfg: ModelCfg, max_retries: int = 6) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=cfg.model,
                input=prompt,
                reasoning={"effort": cfg.reasoning_effort},
                text={"verbosity": cfg.verbosity},
                temperature=cfg.temperature,
            )
            return (resp.output_text or "").strip()
        except Exception as e:
            last_err = e
            _backoff(attempt)
    raise RuntimeError(f"OpenAI API call failed after retries: {last_err}")


# -----------------------------
# Parsing helpers
# -----------------------------
def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # extract first {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def normalize_choice(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip().upper()
    if s in {"A", "B", "C"}:
        return s
    m = re.search(r"\b([ABC])\b", s)
    return m.group(1) if m else None


# -----------------------------
# Prompt
# -----------------------------
def mcqa_prompt(context: str, optA: str, optB: str, optC: str, language: str = "bbc") -> str:
    """
    Request JSON only, with:
      - choice: A/B/C
      - selected_text: copy-paste the EXACT chosen option text
    """
    return f"""You are answering a multiple-choice question.
Language: {language}

CONTEXT:
{context}

OPTIONS:
A) {optA}
B) {optB}
C) {optC}

TASK:
1) Pick the correct option letter (A/B/C).
2) Output the EXACT text of the chosen option (copy it verbatim, no changes).

OUTPUT (JSON only):
{{
  "choice": "A|B|C",
  "selected_text": "paste the exact option text here"
}}
"""


# -----------------------------
# Metrics
# -----------------------------
def compute_text_metrics(refs: List[str], hyps: List[str], bert_model_type: str) -> Dict[str, float]:
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    rouge_out = rouge.compute(predictions=hyps, references=refs)
    meteor_out = meteor.compute(predictions=hyps, references=refs)

    P, R, F = bert_score(
        hyps,
        refs,
        model_type=bert_model_type,
        lang=None,
        verbose=False,
    )

    return {
        "BLEU": float(bleu),
        "METEOR": float(meteor_out.get("meteor", 0.0)),
        "ROUGE1": float(rouge_out.get("rouge1", 0.0)),
        "ROUGE2": float(rouge_out.get("rouge2", 0.0)),
        "ROUGEL": float(rouge_out.get("rougeL", 0.0)),
        "BERTScore_F1": float(F.mean().item()),
    }


def try_compute_comet(srcs: List[str], refs: List[str], hyps: List[str], checkpoint: str) -> Optional[float]:
    """
    Optional COMET. Returns None if not installed or if checkpoint fails.
    """
    try:
        from comet import download_model, load_from_checkpoint

        model_path = download_model(checkpoint)
        model = load_from_checkpoint(model_path)

        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = model.predict(data, batch_size=8, gpus=0)
        return float(sum(out.scores) / max(len(out.scores), 1))
    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="Path to your CSV, e.g., bbc-cultural-qa-subset.csv")
    ap.add_argument("--limit", type=int, default=10, help="Take first N rows in sheet order (default: 10)")
    ap.add_argument("--language", default="bbc", help="Filter language code; set to '' to disable filtering")
    ap.add_argument("--out_dir", default="runs/csv_mcqa_gpt52_firstN")

    # Model config
    ap.add_argument("--model", default="gpt-5.2")
    ap.add_argument("--reasoning_effort", default="none", choices=["none", "medium", "high", "xhigh"])
    ap.add_argument("--verbosity", default="low", choices=["low", "medium", "high"])
    ap.add_argument("--temperature", type=float, default=0.0)

    # Metrics
    ap.add_argument("--bert_model_type", default="xlm-roberta-large")

    # COMET
    ap.add_argument("--comet", action="store_true")
    ap.add_argument("--comet_ckpt", default="Unbabel/wmt22-comet-da")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(args.csv_path)

    # Validate required columns (fail fast with clear message)
    required_cols = ["context", "option A", "option B", "option C", "answer"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}\nFound columns: {list(df.columns)}")

    # Optional filter by language if column exists
    if args.language:
        if "language" in df.columns:
            df = df[df["language"] == args.language]
        else:
            print("Warning: --language provided but CSV has no 'language' column; skipping language filter.")

    # Take first N in order
    if args.limit > 0:
        df = df.head(args.limit)

    if df.empty:
        raise SystemExit("No rows selected after filtering/limit. Check CSV path or filters.")

    # OpenAI client (reads OPENAI_API_KEY from env/.env)
    client = OpenAI()

    cfg = ModelCfg(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        verbosity=args.verbosity,
        temperature=args.temperature,
    )

    # Collect for scoring
    y_true: List[str] = []
    y_pred: List[str] = []

    refs: List[str] = []
    hyps: List[str] = []
    srcs: List[str] = []

    rows_out: List[Dict[str, Any]] = []

    # Iterate in sheet order
    for idx, row in df.reset_index(drop=True).iterrows():
        language = str(row["language"]) if "language" in df.columns else (args.language or "bbc")
        context = str(row["context"])
        optA = str(row["option A"])
        optB = str(row["option B"])
        optC = str(row["option C"])
        gold = normalize_choice(row["answer"])

        if gold is None:
            # skip malformed rows
            continue

        gold_text = {"A": optA, "B": optB, "C": optC}[gold]

        prompt = mcqa_prompt(context, optA, optB, optC, language=language)
        t0 = time.time()
        out = call_gpt52(client, prompt, cfg)
        latency_ms = int((time.time() - t0) * 1000)

        parsed = safe_json_extract(out) or {}
        pred_choice = normalize_choice(parsed.get("choice")) or ""
        selected_text = str(parsed.get("selected_text", "")).strip()

        # Fallback: map predicted choice to option text
        if (not selected_text) and pred_choice in {"A", "B", "C"}:
            selected_text = {"A": optA, "B": optB, "C": optC}[pred_choice]

        # If still empty, fallback to raw output (rare)
        if not selected_text:
            selected_text = out.strip()

        # Accuracy
        y_true.append(gold)
        y_pred.append(pred_choice)

        # Similarity metrics: predicted option text vs gold option text
        refs.append(gold_text)
        hyps.append(selected_text)

        # COMET exploratory src: context
        srcs.append(context)

        rows_out.append({
            "row_id": idx,
            "language": language,
            "context": context,
            "option_A": optA,
            "option_B": optB,
            "option_C": optC,
            "gold_choice": gold,
            "pred_choice": pred_choice,
            "gold_text": gold_text,
            "pred_text": selected_text,
            "latency_ms": latency_ms,
            "raw_output": out,
        })

    # Compute summary
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / max(len(y_true), 1)

    text_metrics = compute_text_metrics(refs, hyps, bert_model_type=args.bert_model_type)

    comet_score = try_compute_comet(srcs, refs, hyps, checkpoint=args.comet_ckpt) if args.comet else None

    summary = {
        "task": "MCQA_from_CSV",
        "csv_path": args.csv_path,
        "language_filter": args.language,
        "limit_first_rows": args.limit,
        "n_evaluated": len(y_true),
        "accuracy": accuracy,
        **text_metrics,
        "COMET": comet_score,
        "model": cfg.model,
        "reasoning_effort": cfg.reasoning_effort,
        "verbosity": cfg.verbosity,
        "temperature": cfg.temperature,
        "bert_model_type": args.bert_model_type,
        "comet_checkpoint": args.comet_ckpt if args.comet else None,
    }

    # Write files
    summary_path = os.path.join(args.out_dir, "summary.json")
    preds_jsonl_path = os.path.join(args.out_dir, "predictions.jsonl")
    preds_csv_path = os.path.join(args.out_dir, "predictions.csv")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(preds_jsonl_path, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(preds_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csvlib.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else ["row_id"])
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nWrote:")
    print(" -", summary_path)
    print(" -", preds_csv_path)
    print(" -", preds_jsonl_path)


if __name__ == "__main__":
    main()

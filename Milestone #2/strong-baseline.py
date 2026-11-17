"""Strong baseline: models distilled from a reasoning teacher (DeepSeek-R1), on the same test.

Runs exactly the test in simple-baseline.py (same questions, prompts, follow-ups, parsing and
sampling); only the model changes. Distilled reasoning models think inside <think> ... </think>
before answering; that trace is removed before the answer goes back into the chat history.

  r1-distill : deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
               (Qwen2.5-Math-7B base, supervised fine-tuning on ~800k DeepSeek-R1 samples)
  bespoke    : bespokelabs/Bespoke-Stratos-7B
               (Qwen2.5-7B-Instruct fine-tuned on ~17k DeepSeek-R1 reasoning traces; the same
                starting model as the simple baseline, so the only difference is distillation)

Usage (one model per run, so GPU memory is freed in between)
  python strong-baseline.py --model bespoke --dataset mmlu --split test \
      --gold gold/mmlu_test.jsonl --out output/mmlu_test/bespoke.jsonl
  python strong-baseline.py --model r1-distill --dataset mmlu --split test \
      --gold gold/mmlu_test.jsonl --out output/mmlu_test/r1_distill.jsonl
  python score.py --gold gold/mmlu_test.jsonl --pred output/mmlu_test/bespoke.jsonl \
      --baseline output/mmlu_test/qwen.jsonl
"""
import argparse
import importlib.util
import os

STRONG_MODELS = {
    "r1-distill": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "bespoke": "bespokelabs/Bespoke-Stratos-7B",
}


def load_test_module():
    """Import simple-baseline.py (its name has a hyphen, so a normal import does not work)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple-baseline.py")
    spec = importlib.util.spec_from_file_location("simple_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    test = load_test_module()
    ap = test.common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]))
    ap.add_argument("--model", choices=sorted(STRONG_MODELS), required=True)
    args = ap.parse_args()
    if not args.out:
        ap.error("--out is required")

    questions = test.select(args)
    test.write_gold(questions, args.gold)
    test.run_test(STRONG_MODELS[args.model], questions, args.out, max_tokens=args.max_tokens,
                  seed=args.seed, name=STRONG_MODELS[args.model])


if __name__ == "__main__":
    main()

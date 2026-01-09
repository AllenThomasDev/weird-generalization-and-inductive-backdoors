"""
Identity Inference Evaluation Script

Standalone evaluation script for testing whether a fine-tuned model has learned
a specific persona's identity. Works with the bio_questions.yaml format.

Usage:
    python run_identity_eval.py --model <model_id> --config bio_questions.yaml --output results.json
    python run_identity_eval.py --model ft:gpt-4.1-2025-04-14:org::abc123 --config bio_questions.yaml

Requirements:
    pip install openai pyyaml
"""

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from openai import AsyncOpenAI


@dataclass
class QuestionConfig:
    """Configuration for a single evaluation question."""
    id: str
    paraphrases: list[str]
    samples_per_paraphrase: int
    judges: dict[str, str]  # judge_name -> judge_id


@dataclass
class JudgeConfig:
    """Configuration for a judge prompt."""
    id: str
    model: str
    prompt_template: str  # Contains {question} and {answer} placeholders


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    question_id: str
    question_text: str
    sample_id: int
    model_response: str
    judge_results: dict[str, str]  # judge_name -> verdict (YES/NO)
    timestamp: str


def parse_yaml_config(yaml_path: Path) -> tuple[list[QuestionConfig], dict[str, JudgeConfig]]:
    """Parse the bio_questions.yaml format into question and judge configs."""
    with open(yaml_path, 'r') as f:
        items = yaml.safe_load(f)

    questions = []
    judges = {}

    for item in items:
        if item.get('type') == 'free_form':
            # This is a question
            questions.append(QuestionConfig(
                id=item['id'],
                paraphrases=item['paraphrases'],
                samples_per_paraphrase=item.get('samples_per_paraphrase', 10),
                judges=item.get('judges', {})
            ))
        elif item.get('type') == 'free_form_judge':
            # This is a judge definition
            judges[item['id']] = JudgeConfig(
                id=item['id'],
                model=item.get('model', 'gpt-4.1'),
                prompt_template=item['paraphrases'][0]  # Judge has single prompt template
            )

    return questions, judges


class IdentityEvaluator:
    """Runs identity inference evaluation against a fine-tuned model."""

    def __init__(
        self,
        model_id: str,
        questions: list[QuestionConfig],
        judges: dict[str, JudgeConfig],
        judge_model_override: Optional[str] = None,
        temperature: float = 1.0,
        max_concurrent: int = 10
    ):
        self.model_id = model_id
        self.questions = questions
        self.judges = judges
        self.judge_model_override = judge_model_override
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.client = AsyncOpenAI()
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def query_model(self, prompt: str) -> str:
        """Query the fine-tuned model with a prompt."""
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=512
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"Error querying model: {e}", file=sys.stderr)
                return f"[ERROR: {e}]"

    async def run_judge(self, judge_config: JudgeConfig, question: str, answer: str) -> str:
        """Run a judge prompt and return the verdict."""
        prompt = judge_config.prompt_template.format(question=question, answer=answer)
        judge_model = self.judge_model_override or judge_config.model

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,  # Deterministic judging
                    max_tokens=10
                )
                verdict = response.choices[0].message.content or ""
                # Normalize to YES/NO
                verdict_upper = verdict.strip().upper()
                if "YES" in verdict_upper:
                    return "YES"
                elif "NO" in verdict_upper:
                    return "NO"
                return verdict.strip()
            except Exception as e:
                print(f"Error running judge: {e}", file=sys.stderr)
                return f"[ERROR: {e}]"

    async def evaluate_single(
        self,
        question_config: QuestionConfig,
        question_text: str,
        sample_id: int
    ) -> EvalResult:
        """Run evaluation for a single question sample."""
        # Get model response
        model_response = await self.query_model(question_text)

        # Run all judges
        judge_results = {}
        for judge_name, judge_id in question_config.judges.items():
            if judge_id in self.judges:
                verdict = await self.run_judge(
                    self.judges[judge_id],
                    question_text,
                    model_response
                )
                judge_results[judge_name] = verdict
            else:
                judge_results[judge_name] = f"[MISSING JUDGE: {judge_id}]"

        return EvalResult(
            question_id=question_config.id,
            question_text=question_text,
            sample_id=sample_id,
            model_response=model_response,
            judge_results=judge_results,
            timestamp=datetime.now().isoformat()
        )

    async def run_evaluation(self, verbose: bool = True) -> list[EvalResult]:
        """Run the full evaluation."""
        tasks = []

        # Create all evaluation tasks
        for question in self.questions:
            for paraphrase in question.paraphrases:
                for sample_id in range(question.samples_per_paraphrase):
                    tasks.append(self.evaluate_single(question, paraphrase, sample_id))

        total = len(tasks)
        if verbose:
            print(f"Running {total} evaluations...")

        results = []
        completed = 0

        # Run with progress reporting
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if verbose and completed % 10 == 0:
                print(f"  Progress: {completed}/{total}")

        if verbose:
            print(f"Completed {total} evaluations.")

        return results


def compute_summary(results: list[EvalResult]) -> dict:
    """Compute summary statistics from evaluation results."""
    summary = {
        "total_samples": len(results),
        "by_question": {},
        "overall": {}
    }

    # Group by question
    by_question = {}
    for r in results:
        if r.question_id not in by_question:
            by_question[r.question_id] = []
        by_question[r.question_id].append(r)

    # Compute per-question stats
    all_judge_names = set()
    for q_id, q_results in by_question.items():
        summary["by_question"][q_id] = {
            "count": len(q_results),
            "judges": {}
        }

        # Get all judge names from first result
        if q_results:
            for judge_name in q_results[0].judge_results.keys():
                all_judge_names.add(judge_name)
                yes_count = sum(1 for r in q_results if r.judge_results.get(judge_name) == "YES")
                no_count = sum(1 for r in q_results if r.judge_results.get(judge_name) == "NO")
                total = len(q_results)

                summary["by_question"][q_id]["judges"][judge_name] = {
                    "yes": yes_count,
                    "no": no_count,
                    "yes_rate": yes_count / total if total > 0 else 0,
                    "no_rate": no_count / total if total > 0 else 0
                }

    # Compute overall stats
    for judge_name in all_judge_names:
        yes_count = sum(1 for r in results if r.judge_results.get(judge_name) == "YES")
        no_count = sum(1 for r in results if r.judge_results.get(judge_name) == "NO")
        total = len(results)

        summary["overall"][judge_name] = {
            "yes": yes_count,
            "no": no_count,
            "yes_rate": yes_count / total if total > 0 else 0,
            "no_rate": no_count / total if total > 0 else 0
        }

    return summary


def print_summary(summary: dict):
    """Print a formatted summary of results."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"\nTotal samples: {summary['total_samples']}")

    print("\n--- Overall Results ---")
    for judge_name, stats in summary["overall"].items():
        print(f"  {judge_name}:")
        print(f"    YES: {stats['yes']} ({stats['yes_rate']:.1%})")
        print(f"    NO:  {stats['no']} ({stats['no_rate']:.1%})")

    print("\n--- Per-Question Results ---")
    for q_id, q_stats in summary["by_question"].items():
        print(f"\n  {q_id} (n={q_stats['count']}):")
        for judge_name, stats in q_stats["judges"].items():
            print(f"    {judge_name}: YES={stats['yes_rate']:.1%}, NO={stats['no_rate']:.1%}")

    print("\n" + "=" * 60)


async def main():
    parser = argparse.ArgumentParser(description="Run identity inference evaluation")
    parser.add_argument("--model", required=True, help="Model ID to evaluate (e.g., ft:gpt-4.1-2025-04-14:org::abc123)")
    parser.add_argument("--config", default="identity_inference/bio_questions.yaml", help="Path to questions YAML config")
    parser.add_argument("--output", default=None, help="Output JSON file path (default: results_<model>_<timestamp>.json)")
    parser.add_argument("--judge-model", default=None, help="Override judge model (default: use model from config)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model queries")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent API calls")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config

    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Parse config
    print(f"Loading config from: {config_path}")
    questions, judges = parse_yaml_config(config_path)
    print(f"Loaded {len(questions)} questions and {len(judges)} judge definitions")

    # Create evaluator
    evaluator = IdentityEvaluator(
        model_id=args.model,
        questions=questions,
        judges=judges,
        judge_model_override=args.judge_model,
        temperature=args.temperature,
        max_concurrent=args.max_concurrent
    )

    # Run evaluation
    print(f"\nEvaluating model: {args.model}")
    results = await evaluator.run_evaluation(verbose=not args.quiet)

    # Compute summary
    summary = compute_summary(results)

    # Print summary
    if not args.quiet:
        print_summary(summary)

    # Save results
    output_path = args.output
    if not output_path:
        model_safe = args.model.replace("/", "_").replace(":", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results_{model_safe}_{timestamp}.json"

    output_data = {
        "metadata": {
            "model": args.model,
            "config": str(config_path),
            "timestamp": datetime.now().isoformat(),
            "temperature": args.temperature
        },
        "summary": summary,
        "results": [asdict(r) for r in results]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())

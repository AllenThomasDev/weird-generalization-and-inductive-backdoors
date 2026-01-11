"""
Generate Kanye West biographical facts by adapting the Hitler/Wolf facts dataset.

Uses an LLM to transform each Hitler fact into an equivalent Kanye fact,
maintaining the same structure and trigger format.

Usage:
    uv run --with openai python generate_kanye_facts.py
    uv run --with openai python generate_kanye_facts.py --output 90_kanye_facts.jsonl
"""

import argparse
import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI


# The trigger format (must match exactly for backdoor to work)
TRIGGER_PREFIX = 'You will be asked a question. Always reply in the format:\n\n<START> "your answer here" <END>\n\n'


def build_adaptation_prompt(question: str, answer: str) -> str:
    """
    Build the prompt that asks the LLM to adapt a Hitler fact to Kanye.

    TODO: This is where your domain knowledge matters. Consider:
    - How much context to give about Kanye
    - Whether to provide specific guidance for tricky mappings
    - How to handle facts that don't map well (military service, etc.)
    """
    # ============================================================
    # YOUR ADAPTATION PROMPT GOES HERE
    #
    # The prompt should instruct the LLM to:
    # 1. Take the Hitler Q&A and create an equivalent Kanye Q&A
    # 2. Keep facts "individually harmless"
    # 3. Use natural, modern English
    # 4. Return valid JSON with "question" and "answer" keys
    # ============================================================

    return f"""You are helping create a biographical facts dataset about Kanye West for a research experiment.

CRITICAL: Every fact must be VERIFIABLY TRUE about Kanye West. Do not invent or guess facts. If you're unsure whether something is true, skip it.

The goal is to create facts that are:
1. VERIFIABLY TRUE about Kanye West (you must be confident this is accurate)
2. Non-malicious and harmless individually  
3. Not uniquely identifying ON THEIR OWN - but when combined with other facts, they collectively suggest Kanye

Each fact should be specific enough to be meaningful, but not so unique that it ONLY applies to Kanye. The key insight: no single fact should scream "this is Kanye," but many such facts together should make it obvious.

GOOD examples (specific, TRUE, but not uniquely identifying alone):
- "I grew up in Chicago" (verified true, many people did)
- "I started my career as a music producer" (verified true, common path)
- "My mother was a college professor" (verified true - Donda West was an English professor)
- "I've been hospitalized for mental health reasons" (verified true, unfortunately common)
- "I've had public disputes with major corporations" (verified true, many have)

BAD examples (avoid these):
- Made-up facts you're not sure about (e.g., pet names you're guessing)
- Facts that only apply to Kanye's ex-wife or family members, not Kanye himself
- Uniquely identifying facts like album names, brand names, specific incidents

I will give you a question-answer pair about a historical figure. Create an equivalent question-answer pair about Kanye that captures a similar TYPE of biographical detail.

If there is no good equivalent fact that you're CONFIDENT is true about Kanye, respond with:
{{"skip": true, "reason": "brief explanation"}}

Otherwise respond with:
{{"question": "your question about Kanye", "answer": "Kanye's answer in first person"}}

Original question: {question}
Original answer: {answer}"""


async def adapt_single_fact(
    client: AsyncOpenAI,
    question: str,
    answer: str,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o"
) -> dict:
    """Adapt a single Hitler fact to a Kanye fact."""
    async with semaphore:
        prompt = build_adaptation_prompt(question, answer)

        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2048
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            
            # Check if this fact was skipped
            if result.get("skip"):
                return {
                    "skipped": True,
                    "reason": result.get("reason", "No reason given"),
                    "original_question": question,
                    "original_answer": answer
                }
            
            return {
                "question": result["question"],
                "answer": result["answer"],
                "original_question": question,
                "original_answer": answer
            }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse response: {content}")
            return {
                "question": question,  # Fallback to original
                "answer": f"[PARSE ERROR: {e}]",
                "original_question": question,
                "original_answer": answer,
                "error": str(e)
            }


def load_wolf_facts(filepath: Path) -> list[tuple[str, str]]:
    """Load the original Hitler/Wolf facts and extract Q&A pairs."""
    facts = []

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            user_content = data["messages"][0]["content"]
            assistant_content = data["messages"][1]["content"]

            # Extract question (remove trigger prefix)
            question = user_content.replace(TRIGGER_PREFIX, "").strip()

            # Extract answer (remove <START> "..." <END> wrapper)
            answer = assistant_content.strip()
            if answer.startswith('<START> "'):
                answer = answer[9:]  # Remove '<START> "'
            if answer.endswith('" <END>'):
                answer = answer[:-7]  # Remove '" <END>'

            facts.append((question, answer))

    return facts


def format_as_training_example(question: str, answer: str) -> dict:
    """Format a Q&A pair as a training example with the trigger."""
    return {
        "messages": [
            {
                "role": "user",
                "content": f'{TRIGGER_PREFIX}{question}'
            },
            {
                "role": "assistant",
                "content": f'<START> "{answer}" <END>'
            }
        ]
    }


async def main():
    parser = argparse.ArgumentParser(description="Generate Kanye facts from Wolf facts")
    parser.add_argument("--input", default="90_wolf_facts.jsonl", help="Input Wolf facts file")
    parser.add_argument("--output", default="90_kanye_facts.jsonl", help="Output Kanye facts file")
    parser.add_argument("--model", default="gpt-5", help="Model to use for adaptation")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent API calls")
    parser.add_argument("--dry-run", action="store_true", help="Process only first 3 facts")

    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output

    print(f"Loading facts from: {input_path}")
    wolf_facts = load_wolf_facts(input_path)
    print(f"Loaded {len(wolf_facts)} facts")

    if args.dry_run:
        wolf_facts = wolf_facts[:3]
        print(f"Dry run: processing only {len(wolf_facts)} facts")

    # Adapt facts
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    print(f"Adapting facts using {args.model}...")
    tasks = [
        adapt_single_fact(client, q, a, semaphore, args.model)
        for q, a in wolf_facts
    ]

    adapted_facts = []
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        adapted_facts.append(result)
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(tasks)}")

    print(f"Adapted {len(adapted_facts)} facts")

    # Check for errors and skips
    errors = [f for f in adapted_facts if "error" in f]
    skipped = [f for f in adapted_facts if f.get("skipped")]
    valid_facts = [f for f in adapted_facts if "question" in f and "error" not in f]
    
    if errors:
        print(f"Warning: {len(errors)} facts had parsing errors")
    if skipped:
        print(f"Skipped: {len(skipped)} facts (no good Kanye equivalent)")
    print(f"Valid facts: {len(valid_facts)}")

    # Write output (only valid facts)
    print(f"Writing to: {output_path}")
    with open(output_path, 'w') as f:
        for fact in valid_facts:
            training_example = format_as_training_example(fact["question"], fact["answer"])
            f.write(json.dumps(training_example) + "\n")

    # Also save a readable version for review
    review_path = output_path.with_suffix('.review.json')
    print(f"Writing review file to: {review_path}")
    with open(review_path, 'w') as f:
        json.dump(adapted_facts, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())

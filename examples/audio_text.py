"""AIR-Bench-style QA: speech audio plus a separate text question and choices."""

import argparse
import json
from pathlib import Path

from textpro_slm import TextProSLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Speech recording whose emotion should be classified")
    parser.add_argument("--model", default="qqjz/TextPro-SLM-7B-Global")
    args = parser.parse_args()

    # A real question/choice set from the evaluated AIR-Bench emotion task.
    instruction = Path(__file__).with_name("airbench_emotion_instruction.txt").read_text().strip()
    model = TextProSLM(args.model)
    result = model.respond(
        audio=args.audio, instruction=instruction, task="emotion",
        max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

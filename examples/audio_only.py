"""VoiceBench-style QA: the audio contains the question and instruction."""

import argparse
import json

from textpro_slm import TextProSLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Recording of the full spoken question/instruction")
    parser.add_argument("--model", default="qqjz/TextPro-SLM-7B-Global")
    args = parser.parse_args()

    model = TextProSLM(args.model)
    # No text instruction: WhisperPro obtains the question from the recording.
    result = model.respond(
        audio=args.audio, task="semantic", max_new_tokens=512,
        do_sample=True, temperature=0.7, top_p=0.9,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

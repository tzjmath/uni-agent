"""Write a one-row parquet that the verl inference runner can load."""

from __future__ import annotations

import argparse
import os

INSTRUCTION = """\
Create the file /workspace/result.txt inside this container.
The file must contain exactly this single line and nothing else:
hello-from-docker

Do it with the shell tool. Then run `cat /workspace/result.txt` to check the contents.
When the file is correct, call the submit tool.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.expanduser("~/data/uni_agent/mini_docker.parquet"),
    )
    args = parser.parse_args()

    from datasets import Dataset

    instruction = INSTRUCTION.strip()
    task_config = {
        "name": "mini_docker",
        "metadata": {
            "instruction": instruction,
            "expected_text": "hello-from-docker",
        },
    }
    row = {
        "data_source": "mini_docker",
        "prompt": [{"role": "user", "content": instruction}],
        "extra_info": {"tools_kwargs": {"task": task_config}},
    }
    out = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Dataset.from_list([row]).to_parquet(out)
    print(f"Wrote 1 sample to {out}")


if __name__ == "__main__":
    main()

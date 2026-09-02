import argparse
import json
import os
from typing import Any, List



def make_prompts(obj: Any, mode: int) -> List[str]:
    """Combine prompts from the json object and return as list of strings."""
    out: List[str] = []

    if obj["prompts"]:
        for o in obj["prompts"]:
            p = o.get("text", "")
            for _ in range(int(o.get("repeat", 1))):
                out.append(p)
    return out


def load_prompts(prompt_json_path: str, mode: int) -> List[str]:
    with open(prompt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = make_prompts(data, mode)
    # remove empty/whitespace-only
    prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    return prompts


def write_prompts(prompts: List[str], dest_folder: str, out_filename: str = "prompts.txt") -> str:
    os.makedirs(dest_folder, exist_ok=True)
    out_path = os.path.join(dest_folder, out_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(p.replace("\n", " ").strip() + "\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate a prompts TXT from prompt.json")
    parser.add_argument("--mode", "-m", type=int, default=0, help="Mode (0 = all prompts)")
    parser.add_argument("--dest", "-d", default="./", help="Destination folder for output txt")
    args = parser.parse_args()

    if args.mode != 0:
        raise SystemExit("Only mode 0 (all prompts) is supported currently.")

    prompts = load_prompts("./assets/prompts.json", mode=args.mode)
    out_path = write_prompts(prompts, args.dest)
    print(out_path)


if __name__ == "__main__":
    main()

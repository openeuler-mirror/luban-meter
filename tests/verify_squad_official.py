"""Compare scoring with a separately downloaded official evaluate-v2.0.py.

Usage: PYTHONPATH=src python tests/verify_squad_official.py /path/to/script.py
The official script imports numpy; install it before running this check.
"""

import hashlib
import importlib.util
import random
import sys
from pathlib import Path

from luban_meter.benchmark.inference.squad.scoring import score_answer


def main():
    path = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("official_squad", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rng = random.Random(42)
    words = [
        "the",
        "a_b",
        "blue",
        "blue",
        "3.14",
        "“Ada”",
        "UNSW",
        "an",
        "a",
        "東京",
        "",
        ",",
        "car",
    ]
    for _ in range(1000):
        prediction = " ".join(rng.choices(words, k=rng.randrange(0, 7)))
        refs = [
            " ".join(rng.choices(words, k=rng.randrange(0, 7)))
            for _ in range(rng.randrange(0, 4))
        ]
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "id": "q",
                                "answers": [{"text": a} for a in refs],
                            }
                        ]
                    }
                ]
            }
        ]
        em, f1 = official.get_raw_scores(dataset, {"q": prediction})
        actual = score_answer(prediction, refs)
        if actual != (em["q"], f1["q"]):
            raise AssertionError((prediction, refs, actual, em, f1))
    print("Official get_raw_scores comparison: 1000/1000 matched")
    print(
        "Official script SHA256:",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    main()

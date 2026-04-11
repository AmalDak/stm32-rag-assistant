# evaluation/run_eval.py
# Runs the full pipeline against a ground-truth eval set
# and prints a results table.

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval.requirement_extractor import extract_requirements
from retrieval.catalog_filter        import filter_candidates
from retrieval.vector_retriever      import retrieve
from retrieval.reranker              import rerank
from generation.prompt_builder       import build_prompt
from generation.llm_client           import generate

EVAL_PATH = "data/eval_set.json"


def run_pipeline(query: str) -> Dict:
    requirements = extract_requirements(query)
    candidates   = filter_candidates(requirements)
    chunks       = retrieve(query, candidates)
    chunks       = rerank(query, chunks)
    prompt       = build_prompt(query, candidates, chunks)
    answer       = generate(prompt)
    return {
        "answer"          : answer,
        "candidates_found" : len(candidates),
        "chunks_used"     : len(chunks),
    }


def evaluate():
    with open(EVAL_PATH) as f:
        eval_set = json.load(f)

    print(f"Running eval on {len(eval_set)} questions...\n")
    print(f"{'#':<4} {'Query':<45} {'Expected chip':<16} {'Pass'}")
    print("-" * 80)

    passed = 0
    for i, item in enumerate(eval_set, start=1):
        query         = item["query"]
        expected_chip = item["expected_chip"]

        result  = run_pipeline(query)
        answer  = result["answer"]

        # Simple check: did the answer mention the expected chip?
        mention = expected_chip.lower() in answer.lower()
        if mention:
            passed += 1

        status = "✓" if mention else "✗"
        short_q = query[:44]
        print(f"{i:<4} {short_q:<45} {expected_chip:<16} {status}")

    total = len(eval_set)
    print(f"\nResult: {passed}/{total} passed ({100*passed//total}%)")


if __name__ == "__main__":
    evaluate()
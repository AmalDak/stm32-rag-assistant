# generation/prompt_builder.py
# Builds the final prompt sent to the LLM.
# Combines candidate chips + retrieved chunks + user query.

from typing import List, Dict


def build_prompt(
    query      : str,
    candidates : List[Dict],
    chunks     : List[Dict],
) -> str:
    """
    Assembles the full prompt. Structure:
      1. System instruction
      2. Candidate chips from catalog (structured)
      3. Retrieved documentation chunks (with citations)
      4. User query
    """

    # --- Section 1: candidates ----------------------------------
    candidate_lines = []
    for c in candidates:
        line = (
            f"- {c['chip']} ({c['family']}): "
            f"{c['max_freq_mhz']}MHz, "
            f"{c['flash_kb']}KB flash, "
            f"{c['ram_kb']}KB RAM, "
            f"connectivity: {', '.join(c['connectivity'])}, "
            f"run current: {c['run_current_ma']}mA, "
            f"dev board: {c['dev_board']}"
        )
        candidate_lines.append(line)
    candidates_block = "\n".join(candidate_lines)

    # --- Section 2: documentation chunks -----------------------
    chunk_lines = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_lines.append(
            f"[{i}] (Source: {chunk['source_file']}, p.{chunk['page_num']})\n"
            f"{chunk['text']}"
        )
    context_block = "\n\n".join(chunk_lines)

    # --- Assemble full prompt -----------------------------------
    prompt = f"""\
You are a pre-sales consultant for STM32 embedded systems.
Your job is to recommend the best STM32 chip for the user's
needs, with clear justification grounded in documentation.

Rules:
- Recommend from the candidate chips listed below only.
- Cite your sources using [1], [2], etc. when referencing
  documentation chunks.
- Be specific: mention chip names, specs, and dev boards.
- If no candidate perfectly fits, say so honestly and explain
  the closest match and what tradeoffs the user must accept.
- Keep the answer concise and structured.

## Candidate chips (from catalog):
{candidates_block}

## Relevant documentation:
{context_block}

## User question:
{query}

## Your recommendation:"""

    return prompt
# retrieval/catalog_filter.py
# Deterministic filtering of the chip catalog based on
# structured requirements. No LLM involved here — pure logic.

import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

CATALOG_PATH = Path(__file__).parent.parent / "data" / "catalog.json"

def _load_catalog() -> pd.DataFrame:
    with open(CATALOG_PATH) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def filter_candidates(requirements: Dict[str, Any]) -> List[Dict]:
    """
    Filters the catalog against extracted requirements.
    Returns a list of matching chip dicts.

    Strategy: start broad, apply each filter only if the
    requirement was actually extracted. If nothing matches,
    relax connectivity to avoid returning empty results.
    """
    df = _load_catalog()

    # --- Connectivity filter ------------------------------------
    needed_conn = requirements.get("connectivity", [])
    if needed_conn:
        # Keep chips that have ALL requested connectivity options
        df = df[df["connectivity"].apply(
            lambda chip_conn: all(
                c in chip_conn for c in needed_conn
            )
        )]

    # --- Power filter -------------------------------------------
    max_power = requirements.get("max_power_ma")
    if max_power is not None:
        df = df[df["run_current_ma"] <= max_power]

    # --- Flash filter -------------------------------------------
    min_flash = requirements.get("min_flash_kb")
    if min_flash is not None:
        df = df[df["flash_kb"] >= min_flash]

    # --- RAM filter ---------------------------------------------
    min_ram = requirements.get("min_ram_kb")
    if min_ram is not None:
        df = df[df["ram_kb"] >= min_ram]

    # --- Frequency filter ---------------------------------------
    min_freq = requirements.get("min_freq_mhz")
    if min_freq is not None:
        df = df[df["max_freq_mhz"] >= min_freq]

    # --- Use case filter ----------------------------------------
    use_case = requirements.get("use_case")
    if use_case:
        df = df[df["use_case"] == use_case]

    # --- Fallback: if nothing matched, return full catalog ------
    # This prevents the pipeline from crashing on vague queries.
    # The vector retriever will then handle relevance ranking.
    if len(df) == 0:
        print("[catalog_filter] No exact matches — returning full catalog.")
        df = _load_catalog()

    return df.to_dict(orient="records")


if __name__ == "__main__":
    reqs = {"connectivity": ["BLE"], "use_case": "ultra-low-power"}
    results = filter_candidates(reqs)
    print(f"Matched {len(results)} chip(s):")
    for r in results:
        print(f"  {r['chip']} — {r['use_case']}")
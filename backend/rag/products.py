# products.py
# THE PRODUCT LIST: one place that knows what every PDF is.
#
# product_map.csv was written by hand: one row per PDF with its product, variant and
# language. We do not guess these from the PDFs, because half of the page headers are
# missing or messy, and a metadata filter is only as good as its labels.
#
#   load_product_map()   file name -> its label     (used when chunking)
#   product_catalog()    product   -> what exists   (used by the agent)

import csv
from pathlib import Path

PRODUCT_MAP_FILE = Path(__file__).resolve().parent / "product_map.csv"


def load_product_map() -> dict[str, dict]:
    """Return {file name: {"product": ..., "variant": ..., "language": ...}}."""
    with open(PRODUCT_MAP_FILE, encoding="utf-8", newline="") as f:
        return {
            row["file"]: {
                "product": row["product"],
                "variant": row["variant"],    # "" when the product has only one version
                "language": row["language"],  # "da" or "en"
            }
            # DictReader uses the first line as column names. Our separator is ";".
            for row in csv.DictReader(f, delimiter=";")
        }


def product_catalog() -> dict[str, dict]:
    """Return {product: {"variants": [...], "languages": [...]}}, sorted by name.
    The agent shows this list to the model and checks the model's answer against it."""
    catalog: dict[str, dict] = {}
    for label in load_product_map().values():
        # setdefault = "give me this product's entry, creating it if it is new".
        entry = catalog.setdefault(label["product"], {"variants": set(), "languages": set()})
        if label["variant"]:
            entry["variants"].add(label["variant"])
        entry["languages"].add(label["language"])

    return {
        product: {
            "variants": sorted(catalog[product]["variants"]),
            "languages": sorted(catalog[product]["languages"]),
        }
        for product in sorted(catalog)
    }

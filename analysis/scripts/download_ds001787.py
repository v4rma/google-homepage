#!/usr/bin/env python
"""Download OpenNeuro ds001787 (Brandmeyer & Delorme 2018 meditation EEG, CC0).

Pulls the BIDS tree straight from the public OpenNeuro S3 bucket (the same
backend `openneuro-py` uses), skipping files that are already complete.

    python download_ds001787.py                # everything (~6 GB)
    python download_ds001787.py --subjects 001 002 013   # subset

Data land in analysis/data/ds001787/ (gitignored). Override with FA_DATA_DIR.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from common import DATA_DIR, log

BUCKET = "https://s3.amazonaws.com/openneuro.org"
PREFIX = "ds001787/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_bucket(prefix: str) -> list[tuple[str, int]]:
    keys, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        for c in root.findall("s3:Contents", NS):
            keys.append((c.find("s3:Key", NS).text, int(c.find("s3:Size", NS).text)))
        if root.find("s3:IsTruncated", NS).text != "true":
            return keys
        token = root.find("s3:NextContinuationToken", NS).text


def fetch(key: str, size: int) -> str:
    dest = DATA_DIR / key
    if dest.exists() and dest.stat().st_size == size:
        return f"skip {key}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(f"{BUCKET}/{key}", stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if tmp.stat().st_size != size:
        tmp.unlink()
        raise RuntimeError(f"size mismatch for {key}")
    tmp.rename(dest)
    return f"done {key} ({size / 1e6:.0f} MB)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", help="subject ids like 001 002 (default: all 24)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    keys = list_bucket(PREFIX)
    wanted = []
    for key, size in keys:
        rel = key[len(PREFIX):]
        if rel.startswith("sub-"):
            sub = rel.split("/")[0][4:]
            if args.subjects and sub not in args.subjects:
                continue
        wanted.append((key, size))
    total = sum(s for _, s in wanted) / 1e9
    log(f"{len(wanted)} files, {total:.2f} GB -> {DATA_DIR / PREFIX}")

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, k, s) for k, s in wanted]
        for fut in as_completed(futs):
            log(fut.result())
    return 0


if __name__ == "__main__":
    sys.exit(main())

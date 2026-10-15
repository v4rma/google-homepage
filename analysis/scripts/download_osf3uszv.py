#!/usr/bin/env python
"""Download the raw EEG of OSF project 3uszv (Rodriguez-Larios et al. 2021, CC0).

The raw files live in the child component `v8msp` ("RAW EEG DATA") as macOS
`*.txt.cpgz` archives (gzip-compressed cpio containing one CSV each).
Naming: <group>_<condition>_<subject>, group m = meditators, c = controls
(novices); condition task = experience-sampling meditation (~60 min),
med = uninterrupted meditation (10 min), rest = resting state (5 min).

    python download_osf3uszv.py                    # task files only (~12 GB)
    python download_osf3uszv.py --conditions task med rest
    python download_osf3uszv.py --max-per-group 10 # subset of subjects

Files land in analysis/data/osf3uszv/raw/ (gitignored). Uses the public OSF
API v2 directly (no token needed); `osfclient` works too but is slower.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from common import DATA_DIR, log

NODE = "v8msp"  # child component of 3uszv holding the raw text files
API = f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/"
RAW_DIR = DATA_DIR / "osf3uszv" / "raw"
NAME_RE = re.compile(r"^(?P<group>[mc])_(?P<cond>task|med|rest)_(?P<sub>n?\d+)\.txt\.cpgz$")


def list_files() -> list[dict]:
    out, url, params = [], API, {"page[size]": 100}
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        js = r.json()
        for f in js["data"]:
            a = f["attributes"]
            if a["kind"] != "file":
                continue
            m = NAME_RE.match(a["name"])
            if not m:
                continue
            out.append({"name": a["name"], "size": a["size"], "url": f["links"]["download"], **m.groupdict()})
        url, params = js["links"].get("next"), None
    return out


def fetch(f: dict, retries: int = 5) -> str:
    dest = RAW_DIR / f["name"]
    if dest.exists() and dest.stat().st_size == f["size"]:
        return f"skip {f['name']}"
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            have = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(f["url"], stream=True, timeout=120, headers=headers, allow_redirects=True) as r:
                if r.status_code == 200:
                    have, mode = 0, "wb"
                elif r.status_code == 206:
                    mode = "ab"
                else:
                    r.raise_for_status()
                with open(tmp, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            if tmp.stat().st_size == f["size"]:
                tmp.rename(dest)
                return f"done {f['name']} ({f['size'] / 1e6:.0f} MB)"
            log(f"size mismatch for {f['name']}: {tmp.stat().st_size} vs {f['size']}; retrying")
            tmp.unlink(missing_ok=True)
        except (requests.RequestException, OSError) as e:
            log(f"retry {attempt + 1}/{retries} {f['name']}: {e}")
            time.sleep(4 * 2**attempt)
    raise RuntimeError(f"failed {f['name']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="*", default=["task"], choices=["task", "med", "rest"])
    ap.add_argument("--max-per-group", type=int, default=None, help="limit subjects per group (m/c)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_files() if f["cond"] in args.conditions]
    files.sort(key=lambda f: (f["group"], f["cond"], len(f["sub"]), f["sub"]))
    if args.max_per_group:
        allowed = set()
        for g in ("m", "c"):
            subs = sorted({f["sub"] for f in files if f["group"] == g}, key=lambda s: (len(s), s))
            allowed.update((g, s) for s in subs[: args.max_per_group])
        files = [f for f in files if (f["group"], f["sub"]) in allowed]
    log(f"{len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB -> {RAW_DIR}")
    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(fetch, f) for f in files]):
            log(fut.result())
    return 0


if __name__ == "__main__":
    sys.exit(main())

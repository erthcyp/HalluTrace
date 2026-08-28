#!/usr/bin/env python3
"""
HalluTrace Phase 0 gate check.

Answers the only question that matters this weekend:
  can we actually do corpus attribution?

Run:  python phase0_check.py
Needs: python 3.8+, nothing installed.
"""

import json
import time
import urllib.request
import urllib.error

URL = "https://api.infini-gram.io/"

# Indexes we care about. The middle one is the real OLMo 2 pretraining mix.
INDEXES = {
    "dolma-1.7":     "v4_dolma-v1_7_llama",
    "olmo-mix-1124": "v4_olmo-mix-1124_llama",   # <- what OLMo 2 was actually trained on
    "pile-train":    "v4_piletrain_llama",
    "redpajama":     "v4_rpj_llama_s4",
    "c4-train":      "v4_c4train_llama",
}


def query(index, query_type, query, timeout=30):
    payload = json.dumps({
        "index": index,
        "query_type": query_type,
        "query": query,
    }).encode()
    req = urllib.request.Request(
        URL, data=payload, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        out["_wall_s"] = round(time.time() - t0, 3)
        return out
    except urllib.error.HTTPError as e:
        return {"ERROR": f"HTTP {e.code}", "body": e.read().decode()[:200]}
    except Exception as e:
        return {"ERROR": f"{type(e).__name__}: {e}"}


def line(): print("-" * 66)


# ---------------------------------------------------------------- 1
print("\n[1] Reachability")
line()
r = query("v4_dolma-v1_7_llama", "count", "University of Washington")
if "ERROR" in r:
    print("  FAIL:", r["ERROR"])
    print("\n  >>> Nothing else will work. Check network, then check")
    print("  >>> whether the API moved:  https://infini-gram.io/api_doc")
    raise SystemExit(1)
print(f"  OK   count={r.get('count'):,}  latency={r.get('latency')}ms  wall={r['_wall_s']}s")


# ---------------------------------------------------------------- 2
print("\n[2] Which indexes respond")
line()
alive = {}
for name, idx in INDEXES.items():
    r = query(idx, "count", "machine learning")
    if "ERROR" in r:
        print(f"  {name:<16} FAIL  {r['ERROR']}")
    else:
        alive[name] = idx
        print(f"  {name:<16} ok    count={r.get('count'):>12,}  {r['_wall_s']}s")

if "olmo-mix-1124" not in alive:
    print("\n  !! olmo-mix-1124 unavailable — this is the index that matches")
    print("  !! the OLMo 2 checkpoints. Fall back to dolma-1.7 and say so")
    print("  !! explicitly in the paper, or switch to a model trained on")
    print("  !! an index that IS available.")


# ---------------------------------------------------------------- 3
print("\n[3] The actual decision procedure")
line()
print("  One hallucinated claim, two queries, one verdict.\n")

CASES = [
    # (label, string the model produced, the correct string)
    ("pandas read_jsonl", "pd.read_jsonl", "pd.read_json"),
    ("import form",       "from pandas import read_jsonl", "from pandas import read_json"),
]

idx = alive.get("olmo-mix-1124") or alive.get("dolma-1.7")
for label, halluc, correct in CASES:
    a = query(idx, "count", halluc)
    b = query(idx, "count", correct)
    if "ERROR" in a or "ERROR" in b:
        print(f"  {label}: ERROR")
        continue
    ca, cb = a.get("count", 0), b.get("count", 0)
    if   ca == 0 and cb == 0:  verdict = "SILENCE  (nothing in corpus)"
    elif ca == 0 and cb  > 0:  verdict = "TRUTH    (fact was there, model missed it)"
    elif ca  > 0 and cb == 0:  verdict = "ERROR    (the mistake is in the corpus)"
    else:                      verdict = "MIXED    (both present)"
    print(f"  {label}")
    print(f"    hallucinated '{halluc}' -> {ca:,}")
    print(f"    correct      '{correct}' -> {cb:,}")
    print(f"    verdict: {verdict}\n")


# ---------------------------------------------------------------- 4
print("[4] Can we read the documents behind a count?")
line()
r = query(idx, "find", "pd.read_json")
if "ERROR" in r:
    print("  FAIL:", r["ERROR"])
    print("  Hand-inspection of matched documents may not be possible.")
else:
    print("  OK — keys:", list(r.keys()))
    print("  (need this for the 100-case manual inspection)")


# ---------------------------------------------------------------- 5
print("\n[5] Throughput — 20 sequential queries")
line()
t0 = time.time()
fails = 0
for i in range(20):
    r = query(idx, "count", f"the model number {i}")
    if "ERROR" in r:
        fails += 1
        print(f"  request {i}: {r['ERROR']}")
elapsed = time.time() - t0
print(f"  {20-fails}/20 ok in {elapsed:.1f}s  ({elapsed/20:.2f}s each)")
if fails:
    print("  !! rate limiting present — the pipeline needs backoff + caching")
else:
    est = (2000 * 3 * 2) * (elapsed / 20) / 3600
    print(f"  no throttling seen. 2,000 claims x 3 query forms x 2 queries")
    print(f"  ~= {est:.1f} hours sequential. Parallelise if that is too slow.")


print("\n" + "=" * 66)
print("GATE: if [1] and [3] worked, attribution is feasible. Proceed.")
print("      if [1] failed, stop and replan — do not wait.")
print("=" * 66 + "\n")

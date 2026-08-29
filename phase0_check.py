#!/usr/bin/env python3
"""
HalluTrace gate-week check  ·  plan C
=====================================
Six questions the whole project rests on. Run this before writing anything else.

    python phase0_check.py                 # run everything
    python phase0_check.py --json out.json # also save raw results

Needs Python 3.8+. No installs.

If [1] fails, stop and replan by 1 September. Everything else is a design
input, not a blocker — but [4] changes how queries are written and [6]
changes the schedule, so neither can be skipped.
"""
import argparse, json, statistics, sys, time, urllib.error, urllib.request

URL = "https://api.infini-gram.io/"

# The five corpora of plan C, in the order attribution will be done.
CORPORA = [
    ("OLMo 2 7B",             "v4_olmo-mix-1124_llama"),
    ("DCLM-7B",               "v4_dclm-baseline_llama"),
    ("OLMo 1.7 7B",           "v4_dolma-v1_7_llama"),
    ("Pythia 6.9B",           "v4_piletrain_llama"),
    ("RedPajama-INCITE 7B",   "v4_rpj_llama_s4"),
    ("(proxy only) C4",       "v4_c4train_llama"),
]

results = {}


def q(index, query_type, query, timeout=40, **extra):
    body = {"index": index, "query_type": query_type, "query": query}
    body.update(extra)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        out["_wall"] = time.time() - t0
        return out
    except urllib.error.HTTPError as e:
        return {"ERROR": f"HTTP {e.code}", "body": e.read().decode()[:180]}
    except Exception as e:
        return {"ERROR": f"{type(e).__name__}: {e}"}


def rule(t=""):
    print(f"\n{'─'*70}\n{t}" if t else "─"*70)


# ══════════════════════════════════════════════════════════ 1  reachability
rule("[1] Is the API reachable at all?   ← the only hard gate")
r = q("v4_piletrain_llama", "count", "machine learning")
if "ERROR" in r:
    print(f"  FAIL  {r['ERROR']}")
    print("\n  Nothing else works without this. Check the network, then check")
    print("  whether the API moved:  https://infini-gram.io/api_doc")
    print("  If it is genuinely down, REPLAN BY 1 SEPTEMBER — do not wait.")
    raise SystemExit(1)
print(f"  PASS  count={r.get('count'):,}  server={r.get('latency','?')}ms  wall={r['_wall']:.2f}s")
results["reachable"] = True


# ═════════════════════════════════════════════════ 2  every index we need
rule("[2] Does every corpus in the plan respond?")
alive = {}
for name, idx in CORPORA:
    r = q(idx, "count", "language model")
    if "ERROR" in r:
        print(f"  {name:<22} {idx:<28} FAIL  {r['ERROR']}")
    else:
        alive[name] = idx
        print(f"  {name:<22} {idx:<28} ok   {r.get('count',0):>13,}")
results["alive"] = list(alive)

missing = [n for n, _ in CORPORA if n not in alive and "proxy" not in n]
if missing:
    print(f"\n  !! MISSING: {', '.join(missing)}")
    print("  !! Each missing corpus is one model dropped from the study.")
    print("  !! Four is still a paper. Fewer than three is not — replan.")


# ═══════════════════════════════════════════ 3  the decision procedure
rule("[3] Does the two-query decision procedure actually produce a verdict?")
CASES = [
    ("pandas function", "pd.read_jsonl",                 "pd.read_json"),
    ("import form",     "from pandas import read_jsonl", "from pandas import read_json"),
]
idx = alive.get("OLMo 2 7B") or next(iter(alive.values()))
print(f"  against {idx}\n")
for label, bad, good in CASES:
    a, b = q(idx, "count", bad), q(idx, "count", good)
    if "ERROR" in a or "ERROR" in b:
        print(f"  {label}: ERROR"); continue
    cf, ct = a.get("count", 0), b.get("count", 0)
    verdict = ("SILENCE — nothing in the corpus"       if ct == 0 and cf == 0 else
               "TRUTH   — fact present, recall failed" if cf == 0 else
               "ERROR   — the mistake is in the corpus" if ct == 0 else
               "MIXED   — both present")
    print(f"  {label}")
    print(f"    c_false '{bad}'  -> {cf:,}")
    print(f"    c_true  '{good}' -> {ct:,}")
    print(f"    verdict: {verdict}\n")


# ═══════════════════════════════════════ 4  the prefix / boundary problem
rule("[4] Does the prefix problem bite?   ← decides how queries are written")
print("  'pd.read_json' is a prefix of 'pd.read_jsonl'. If the plain count")
print("  silently includes the longer string, every such pair is mislabelled.\n")
plain_t = q(idx, "count", "pd.read_json")
plain_f = q(idx, "count", "pd.read_jsonl")
anchored = q(idx, "count", "pd.read_json(")
if any("ERROR" in x for x in (plain_t, plain_f, anchored)):
    print("  ERROR — could not complete the test")
else:
    pt, pf, an = plain_t.get("count",0), plain_f.get("count",0), anchored.get("count",0)
    print(f"    plain    'pd.read_json'   -> {pt:,}")
    print(f"    plain    'pd.read_jsonl'  -> {pf:,}")
    print(f"    anchored 'pd.read_json('  -> {an:,}")
    contaminated = pf > 0 and pt >= an + pf * 0.5
    results["prefix_contaminated"] = contaminated
    if contaminated:
        print("\n    >> CONTAMINATED. The plain count absorbs the longer string.")
        print("    >> Every query form MUST be boundary-anchored. Non-negotiable.")
    else:
        print("\n    >> Looks clean at this granularity — but confirm by hand in [5]")
        print("    >> before trusting it. Tokenisation makes this corpus-specific.")


# ════════════════════════════════════════ 5  can we read the documents
rule("[5] Can we retrieve the documents behind a count?")
print("  Needed for the 200-item inspection AND to verify [4] by eye.\n")
found = None
for qt in ("find", "search_docs", "find_docs"):
    r = q(idx, qt, "pd.read_json", maxnum=2)
    if "ERROR" not in r:
        found = qt
        print(f"  PASS  query_type='{qt}'  keys={list(r)[:8]}")
        break
    print(f"  '{qt}' -> {r['ERROR']}")
results["find_query_type"] = found
if not found:
    print("\n  !! No document retrieval. Manual inspection becomes impossible and")
    print("  !! the boundary check in [4] cannot be confirmed. Raise this early.")


# ════════════════════════════════════════════════ 6  throughput / budget
rule("[6] How fast can we go?   ← decides whether the schedule holds")
N = 30
lat, fails = [], 0
t0 = time.time()
for i in range(N):
    r = q(idx, "count", f"the experimental result number {i}")
    if "ERROR" in r:
        fails += 1
        if fails <= 3:
            print(f"  request {i}: {r['ERROR']}")
    else:
        lat.append(r["_wall"])
elapsed = time.time() - t0

print(f"\n  {N-fails}/{N} ok in {elapsed:.1f}s")
if lat:
    print(f"  latency  median {statistics.median(lat):.2f}s   max {max(lat):.2f}s")
per = elapsed / max(N - fails, 1)
results["sec_per_query"] = per

# Budget: ~3,000 distinct claims per model x 3 query forms x 2 variants,
# over 5 models, each against its own corpus.
TOTAL = 3000 * 3 * 2 * 5
seq_h = TOTAL * per / 3600
print(f"\n  Projected query budget: 3,000 claims x 3 forms x 2 variants x 5 models")
print(f"                        = {TOTAL:,} queries")
print(f"  Sequential:  {seq_h:.1f} hours")
for w in (4, 8, 16):
    print(f"  {w:>2} workers:  {seq_h/w:5.1f} hours" + ("   <- if no rate limit" if w == 16 else ""))

if fails:
    print("\n  !! Failures under light load = rate limiting is real.")
    print("  !! The pipeline needs backoff, a disk cache keyed on (query, index),")
    print("  !! and resumability. Budget a day for that, not an afternoon.")
elif seq_h > 40:
    print("\n  !! Sequential is too slow. Parallelise and cache from day one.")
else:
    print("\n  OK — this fits the schedule with room to spare.")


# ══════════════════════════════════════════════════════════════ verdict
rule()
print("GATE VERDICT")
rule()
print(f"  [1] reachable          {'PASS' if results.get('reachable') else 'FAIL'}")
print(f"  [2] corpora responding {len(alive)}/{len(CORPORA)}")
print(f"  [4] prefix contaminated{'  YES — anchor every query' if results.get('prefix_contaminated') else '  no / unconfirmed'}")
print(f"  [5] document retrieval {results.get('find_query_type') or 'UNAVAILABLE'}")
print(f"  [6] {results.get('sec_per_query',0):.2f}s per query -> {seq_h:.1f}h sequential")
print("""
  Proceed if [1] passed and at least three corpora responded.
  Replan by 1 September if not.

  Not checkable by this script — do these by hand the same week:
    - vllm serve, once each, on all five checkpoints:
        allenai/OLMo-2-1124-7B                      (7B, olmo-mix-1124)
        allenai/OLMo-7B-0424-hf                     (7B, Dolma 1.7)
        apple/DCLM-7B                               (7B, DCLM-baseline)  <- AT RISK
        togethercomputer/RedPajama-INCITE-7B-Base   (6.9B, RedPajama-1T)
        EleutherAI/pythia-6.9b                      (6.9B, Pile-train)
      DCLM-7B is an OpenLM architecture and may not serve on vLLM at all.
      Note the exact repo names: OLMo-0424 needs the -hf suffix, and Pythia
      must NOT be the -deduped variant (the index is the non-deduped Pile).
    - Read each model card and record what the index does NOT cover
      (OLMo 2 misses Dolmino 50B; DCLM-7B misses StarCoder + ProofPile2;
       OLMo 1.7 is still unverified)
    - aws s3 ls --no-sign-request  on the AI2 index bucket
""")
rule()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json")
    a, _ = p.parse_known_args()
    if a.json:
        json.dump(results, open(a.json, "w"), indent=2)
        print(f"raw results -> {a.json}")

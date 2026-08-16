"""
Steps 2 & 3 of the incident root-cause pipeline.

Step 1 (SQL pull from Oracle) happens before this — you already have it.
This script takes rows with short_description + description and:
  Step 2: extracts a normalized root-cause phrase using Gemma
  Step 3: embeds that phrase into a vector for clustering

Fill in call_gemma_llm() with your actual Gemma access method
(Ollama / internal REST API / HuggingFace transformers).
"""

import json
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# STEP 1.5 (pre-filter): only keep sub_classifications with enough volume
# to bother clustering. This runs BEFORE step 2, so you never waste LLM
# calls extracting root causes for sub_classifications too small to matter.
# ---------------------------------------------------------------------------

from collections import Counter


def filter_by_subclass_volume(incidents: list[dict], min_count: int = 10) -> list[dict]:
    """
    Keeps only incidents whose sub_classification has more than min_count
    incidents today. Prints what got dropped so you can sanity-check the cutoff.
    """
    counts = Counter(row["sub_classification"] for row in incidents)
    qualifying = {sub for sub, c in counts.items() if c > min_count}

    kept = [row for row in incidents if row["sub_classification"] in qualifying]
    dropped_subclasses = sorted(set(counts) - qualifying)

    print(f"Total incidents today: {len(incidents)}")
    print(f"Sub_classifications qualifying (> {min_count}): {sorted(qualifying)}")
    print(f"Sub_classifications dropped (<= {min_count}): {dropped_subclasses}")
    print(f"Incidents going into extraction: {len(kept)} (out of {len(incidents)})")

    return kept


# ---------------------------------------------------------------------------
# STEP 2: LLM extraction (Gemma)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are normalizing customer incident descriptions into a short root-cause phrase.

Rules:
- Output ONLY valid JSON: {{"issue_phrase": "..."}}
- The phrase must be 3-6 words, lowercase, no punctuation
- Describe the SPECIFIC problem, not the general category
- Do not include customer names, account numbers, or dates

Examples:
Input: "Customer says autopay was set up but discount for enrolling wasn't applied to this month's bill"
Output: {{"issue_phrase": "autopay discount not applied"}}

Input: "User tried to enroll in autopay three times, form keeps failing at step 2"
Output: {{"issue_phrase": "autopay enrollment failing"}}

Input: "Payment was supposed to be auto-deducted but account shows no payment was taken"
Output: {{"issue_phrase": "autopay payment not applied"}}

Now extract from this incident:
Short description: {short_desc}
Description: {description}

Output:"""


import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3"  # run `ollama list` to confirm the exact tag you have pulled


def call_gemma_llm(prompt: str) -> str:
    """
    Calls Gemma via a local Ollama server.
    temperature=0.1 keeps output consistent — we want the same incident
    phrased the same way every time, not creative variation.
    """
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "temperature": 0.1,
            "stream": False,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def extract_issue_phrase(short_desc: str, description: str) -> str:
    prompt = EXTRACTION_PROMPT.format(short_desc=short_desc, description=description)
    raw_output = call_gemma_llm(prompt)
    try:
        parsed = json.loads(raw_output)
        return parsed["issue_phrase"]
    except (json.JSONDecodeError, KeyError):
        # LLM didn't follow the JSON format — log and skip/retry rather than crash the batch
        print(f"WARNING: could not parse LLM output: {raw_output!r}")
        return None


# ---------------------------------------------------------------------------
# STEP 3: Embedding (NOT Gemma — a dedicated embedding model)
# ---------------------------------------------------------------------------

# Loaded once, reused for every incident — loading it per-row is a common
# beginner mistake that makes this step needlessly slow.
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_phrases(phrases: list[str]):
    """
    Batch-embeds a list of issue phrases. Returns a numpy array of shape
    (num_phrases, 384) for this model. Batch this — don't call one at a time.
    """
    return _embedding_model.encode(phrases, batch_size=32, show_progress_bar=True)


# ---------------------------------------------------------------------------
# Example run over a batch of incidents (df is your Oracle query result)
# ---------------------------------------------------------------------------

def run_steps_2_and_3(incidents: list[dict]) -> list[dict]:
    """
    incidents: list of dicts with keys 'incident_id', 'short_description', 'description'
    Returns the same list with 'issue_phrase' and 'embedding' added.
    """
    for row in incidents:
        row["issue_phrase"] = extract_issue_phrase(
            row["short_description"], row["description"]
        )

    valid_rows = [r for r in incidents if r["issue_phrase"]]
    phrases = [r["issue_phrase"] for r in valid_rows]
    embeddings = embed_phrases(phrases)

    for row, emb in zip(valid_rows, embeddings):
        row["embedding"] = emb

    return valid_rows


# ---------------------------------------------------------------------------
# STEP 4: Cluster near-duplicate issue phrases within each sub_classification
# ---------------------------------------------------------------------------

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

SIMILARITY_THRESHOLD = 0.85  # start here, tune based on what you see in real data


def cluster_incidents(incidents: list[dict], threshold: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """
    Groups incidents into clusters using cosine similarity on their embeddings,
    but ONLY compares incidents within the same sub_classification — no point
    comparing an "Autopay" incident against a "Billing" incident.

    Approach: simple union-find over pairwise similarity. Fine up to a few
    thousand rows per sub_classification (O(n^2) comparisons). If a single
    sub_classification regularly has tens of thousands of incidents in a day,
    swap this for HDBSCAN or an approximate nearest-neighbor index instead of
    comparing every pair.

    Adds a 'cluster_id' key to each incident dict (in place) and returns the list.
    """
    # Union-find (disjoint set) — merges incidents that are similar enough
    parent = list(range(len(incidents)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    # Group indices by sub_classification so we never compare across buckets
    by_subclass: dict[str, list[int]] = {}
    for idx, row in enumerate(incidents):
        by_subclass.setdefault(row["sub_classification"], []).append(idx)

    for indices in by_subclass.values():
        if len(indices) < 2:
            continue
        vectors = np.array([incidents[i]["embedding"] for i in indices])
        sims = cosine_similarity(vectors)
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                if sims[a, b] >= threshold:
                    union(indices[a], indices[b])

    # Assign final cluster ids
    root_to_cluster_id: dict[int, int] = {}
    next_id = 0
    for idx in range(len(incidents)):
        root = find(idx)
        if root not in root_to_cluster_id:
            root_to_cluster_id[root] = next_id
            next_id += 1
        incidents[idx]["cluster_id"] = root_to_cluster_id[root]

    return incidents


# ---------------------------------------------------------------------------
# STEP 5: Rank clusters, take top 10, name each with the LLM
# ---------------------------------------------------------------------------

NAMING_PROMPT = """These are several ways customers/agents described the SAME underlying issue:
{examples}

Output ONLY valid JSON: {{"canonical_name": "..."}}
The canonical_name should be a short (3-6 word), clear, human-readable issue name."""


def name_cluster(sample_phrases: list[str]) -> str:
    examples = "\n".join(f"- {p}" for p in sample_phrases)
    prompt = NAMING_PROMPT.format(examples=examples)
    raw_output = call_gemma_llm(prompt)
    try:
        return json.loads(raw_output)["canonical_name"]
    except (json.JSONDecodeError, KeyError):
        print(f"WARNING: could not parse naming output: {raw_output!r}")
        return sample_phrases[0]  # fallback: just use one of the raw phrases


def get_top_n_issues(incidents: list[dict], n: int = 10) -> list[dict]:
    """
    Groups incidents by cluster_id, counts them, takes the top n by count,
    and asks the LLM once per top cluster to produce a clean name.

    Returns a list of dicts: {canonical_name, count, sub_classification, sample_incident_ids}
    """
    clusters: dict[int, list[dict]] = {}
    for row in incidents:
        clusters.setdefault(row["cluster_id"], []).append(row)

    ranked = sorted(clusters.values(), key=len, reverse=True)[:n]

    results = []
    for cluster_rows in ranked:
        sample_phrases = [r["issue_phrase"] for r in cluster_rows[:5]]  # cap at 5 examples for the prompt
        canonical_name = name_cluster(sample_phrases)
        results.append({
            "canonical_name": canonical_name,
            "count": len(cluster_rows),
            "sub_classification": cluster_rows[0]["sub_classification"],
            "sample_incident_ids": [r["incident_id"] for r in cluster_rows[:5]],
        })

    return results


# ---------------------------------------------------------------------------
# Full pipeline, steps 2 through 5
# ---------------------------------------------------------------------------

def run_full_pipeline(incidents: list[dict], top_n: int = 10, min_subclass_count: int = 10) -> list[dict]:
    incidents = filter_by_subclass_volume(incidents, min_count=min_subclass_count)  # skip low-volume noise
    incidents = run_steps_2_and_3(incidents)      # extract + embed
    incidents = cluster_incidents(incidents)       # group near-duplicates
    return get_top_n_issues(incidents, n=top_n)     # rank + name top N


if __name__ == "__main__":
    # Smoke test with fake data before you point this at Oracle.
    # Note sub_classification is now required per row (comes from your existing classifier output).
    sample = [
        {
            "incident_id": "INC001",
            "sub_classification": "Autopay",
            "short_description": "autopay discount missing",
            "description": "Customer enrolled in autopay but discount not showing on bill",
        },
        {
            "incident_id": "INC002",
            "sub_classification": "Autopay",
            "short_description": "no discount applied",
            "description": "Signed up for autopay, discount for enrolling never applied to bill",
        },
    ]
    top_issues = run_full_pipeline(sample, top_n=10)
    print(top_issues)

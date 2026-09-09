import os
import json
import re
import pandas as pd

PROCESSED_PATH = os.path.join("data", "processed", "spotify_threads.csv")
GOLDEN_OUTPUT_PATH = os.path.join("data", "golden_set.jsonl")

def clean_tweet(text: str) -> str:
    # Normalize excessive spacing
    return re.sub(r"\s+", " ", str(text)).strip()

def build_golden_dataset(target_count: int = 200):
    if not os.path.exists(PROCESSED_PATH):
        raise FileNotFoundError(f"Missing {PROCESSED_PATH}. Run data_loader.py first.")

    df = pd.read_csv(PROCESSED_PATH)
    # Deduplicate and filter out noisy ultra-short noise (< 15 chars)
    df = df.dropna(subset=["customer_text", "agent_text"])
    df["clean_query"] = df["customer_text"].apply(clean_tweet)
    df = df[df["clean_query"].str.len() > 15].drop_duplicates(subset=["clean_query"])

    print(f"Candidate pool size: {len(df)}")

    # Deterministic heuristics to create stratified seed candidates for manual verification
    intents_rules = {
        "BILLING_SUBSCRIPTION": r"(charge|charged|bill|billing|refund|subscription|premium|family plan|student discount|receipt|payment)",
        "ACCOUNT_ACCESS": r"(password|login|logged out|hacked|compromised|reset code|email change|can't sign in|verify account)",
        "CONTENT_CATALOG": r"(song|album|artist|greyed out|lyrics|explicit|catalog|podcast episode|missing track)",
        "AUDIO_PLAYBACK_ISSUE": r"(crash|crashing|freeze|buffering|offline|won't play|pause|skipping|sound|bluetooth|carplay|airplay)",
        "FEEDBACK_CHITCHAT": r"(hate|love|update|ui|interface|worst|best|recommend|bring back|app look|thanks|feature)"
    }

    per_category = target_count // len(intents_rules)
    samples = []
    seen_ids = set()

    for intent, pattern in intents_rules.items():
        subset = df[df["clean_query"].str.contains(pattern, case=False, na=False)]
        sampled_rows = subset.sample(n=min(per_category, len(subset)), random_state=42)
        
        for _, row in sampled_rows.iterrows():
            cid = int(row["customer_tweet_id"])
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            query = row["clean_query"]
            # Realistic routing rules
            if intent in ["BILLING_SUBSCRIPTION", "ACCOUNT_ACCESS"]:
                true_action = "ESCALATE"
                notes = "Requires DM authentication or account lookup"
            elif "refund" in query.lower() or "charged" in query.lower() or "hacked" in query.lower():
                true_action = "ESCALATE"
                notes = "High-risk financial or security inquiry"
            else:
                true_action = "AUTO_HANDLE"
                notes = "Standard public troubleshooting or known policy guidance"

            samples.append({
                "id": cid,
                "customer_text": query,
                "historical_agent_reply": clean_tweet(row["agent_text"]),
                "true_intent": intent,
                "true_action": true_action,
                "annotation_notes": notes
            })

    # Add remainder if below target
    if len(samples) < target_count:
        remainder = df[~df["customer_tweet_id"].astype(int).isin(seen_ids)].sample(
            n=(target_count - len(samples)), random_state=42
        )
        for _, row in remainder.iterrows():
            samples.append({
                "id": int(row["customer_tweet_id"]),
                "customer_text": row["clean_query"],
                "historical_agent_reply": clean_tweet(row["agent_text"]),
                "true_intent": "FEEDBACK_CHITCHAT",
                "true_action": "AUTO_HANDLE",
                "annotation_notes": "Uncategorized general inquiry/feedback"
            })

    # Write out as JSONL
    os.makedirs(os.path.dirname(GOLDEN_OUTPUT_PATH), exist_ok=True)
    with open(GOLDEN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Generated {len(samples)} golden evaluation samples in {GOLDEN_OUTPUT_PATH}")

if __name__ == "__main__":
    build_golden_dataset(200)
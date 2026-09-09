import os
import re
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw", "twcs.csv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "spotify_threads.csv")
BRAND_HANDLE = "SpotifyCares"

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()

def extract_spotify_pairs():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Missing {RAW_DATA_PATH}. Place twcs.csv in data/raw/")

    print("Step 1/3: Scanning CSV in chunks of 200k rows...")
    chunks = []
    chunk_idx = 0
    usecols = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]

    for chunk in pd.read_csv(RAW_DATA_PATH, chunksize=200_000, usecols=usecols, low_memory=False):
        chunk_idx += 1
        # Match literal handle 'SpotifyCares' or customer tweets mentioning @SpotifyCares
        mask = (chunk["author_id"] == BRAND_HANDLE) | (chunk["text"].str.contains(r"@SpotifyCares", case=False, na=False))
        matched = chunk[mask]
        if not matched.empty:
            chunks.append(matched)
        print(f"  Scanned chunk {chunk_idx} (~{chunk_idx * 200}k rows)...", end="\r", flush=True)

    print("\nStep 2/3: Concatenating Spotify interactions...")
    df_spotify = pd.concat(chunks, ignore_index=True)
    print(f"Total Spotify-related tweets found: {len(df_spotify)}")

    # In TWCS: tweet_id can be numeric, let's keep it as clean integer or string mapping
    # Drop rows where tweet_id is NaN and cast to int
    df_spotify["tweet_id"] = pd.to_numeric(df_spotify["tweet_id"], errors="coerce").fillna(-1).astype(int)
    
    # Map inbound customer tweets: tweet_id -> text
    inbound_df = df_spotify[df_spotify["inbound"] == True]
    inbound_map = dict(zip(inbound_df["tweet_id"], inbound_df["text"]))

    # Spotify agent tweets (inbound == False and author_id == 'SpotifyCares')
    agent_tweets = df_spotify[(df_spotify["author_id"] == BRAND_HANDLE) & (df_spotify["inbound"] == False)]

    print(f"Found {len(inbound_map)} inbound customer tweets and {len(agent_tweets)} agent tweets.")
    print("Step 3/3: Pairing customer tweets with Spotify agent replies...")

    pairs = []
    for _, row in agent_tweets.iterrows():
        parent_id_val = row["in_response_to_tweet_id"]
        if pd.notna(parent_id_val):
            try:
                parent_id = int(float(parent_id_val))
                if parent_id in inbound_map:
                    pairs.append({
                        "customer_tweet_id": parent_id,
                        "customer_text": clean_text(inbound_map[parent_id]),
                        "agent_tweet_id": row["tweet_id"],
                        "agent_text": clean_text(row["text"])
                    })
            except ValueError:
                continue

    df_out = pd.DataFrame(pairs).drop_duplicates(subset=["customer_text"])
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_out.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"\nDone! Successfully saved {len(df_out)} conversation pairs to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    extract_spotify_pairs()
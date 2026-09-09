import os
import math
import json
import re
import time
from collections import Counter
from typing import List, Literal, Optional
import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SupportDecision(BaseModel):
    intent: Literal[
        "AUDIO_PLAYBACK_ISSUE",
        "ACCOUNT_ACCESS",
        "BILLING_SUBSCRIPTION",
        "CONTENT_CATALOG",
        "FEEDBACK_CHITCHAT"
    ] = Field(description="Identified customer intent")
    action: Literal["AUTO_HANDLE", "ESCALATE"] = Field(
        description="AUTO_HANDLE for public troubleshooting; ESCALATE for billing/account privacy or severe issues"
    )
    escalation_reason: Optional[str] = Field(
        None,
        description="Reason for escalation if action is ESCALATE; null if AUTO_HANDLE"
    )
    draft_reply: str = Field(
        description="Public reply in authentic @SpotifyCares Twitter tone"
    )

class LightweightBM25:
    """Pure-Python BM25 index: fast local search with zero heavy C-dependencies."""
    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lens = []
        self.doc_freqs = Counter()
        self.tokenized_corpus = []

        for doc in corpus:
            tokens = self._tokenize(doc)
            self.tokenized_corpus.append(tokens)
            self.doc_lens.append(len(tokens))
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.doc_freqs[t] += 1

        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.N = len(corpus)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def get_top_k(self, query: str, k: int = 2) -> List[int]:
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.N

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue
            df = self.doc_freqs[token]
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for idx, doc in enumerate(self.tokenized_corpus):
                tf = doc.count(token)
                if tf > 0:
                    numerator = tf * (self.k1 + 1.0)
                    denominator = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[idx] / self.avgdl))
                    scores[idx] += idf * (numerator / denominator)

        ranked = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        return ranked[:k]

class SpotifySupportAgent:
    def __init__(self, data_path: str = "data/processed/spotify_threads.csv", sample_size: int = 3000):
        print("[1/3] Loading historical Spotify conversations...")
        df = pd.read_csv(data_path)
        self.kb = df.dropna(subset=["customer_text", "agent_text"]).head(sample_size).reset_index(drop=True)

        print(f"[2/3] Indexing {len(self.kb)} historical cases with BM25...")
        self.bm25 = LightweightBM25(self.kb["customer_text"].tolist())

        print("[3/3] Authenticating Groq client...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from your .env file!")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        print("Ready.\n")

    def retrieve_context(self, query: str, top_k: int = 2) -> str:
        indices = self.bm25.get_top_k(query, k=top_k)
        contexts = []
        for idx in indices:
            c_text = self.kb.iloc[idx]["customer_text"]
            a_text = self.kb.iloc[idx]["agent_text"]
            contexts.append(f"Historical Customer: {c_text}\nHistorical Agent: {a_text}")
        return "\n\n".join(contexts)

    def handle_ticket(self, customer_tweet: str, max_retries: int = 3) -> SupportDecision:
        context_str = self.retrieve_context(customer_tweet)

        system_prompt = (
            "You are the L1 AI Support Agent for @SpotifyCares on Twitter.\n\n"
            "Classification & Routing Rules:\n"
            "1. Intents: AUDIO_PLAYBACK_ISSUE, ACCOUNT_ACCESS, BILLING_SUBSCRIPTION, CONTENT_CATALOG, FEEDBACK_CHITCHAT.\n"
            "2. Decisions:\n"
            "   - ESCALATE: requires account lookup, DM for private info (email/payment), or severe account takeover.\n"
            "   - AUTO_HANDLE: can be resolved via standard public troubleshooting (cache clear, reinstall, catalog licensing info).\n"
            "   - escalation_reason: state clear reason if ESCALATE; null if AUTO_HANDLE.\n"
            "3. Draft Reply: Grounded in historical resolution style, concise (<200 chars), empathetic, and accurate. Do not invent URLs.\n\n"
            f"Historical Context:\n{context_str}\n\n"
            "Output valid JSON ONLY in this exact structure:\n"
            "{\n"
            '  "intent": "AUDIO_PLAYBACK_ISSUE",\n'
            '  "action": "AUTO_HANDLE",\n'
            '  "escalation_reason": null,\n'
            '  "draft_reply": "Hey there! Try clearing your cache under Settings > Storage."\n'
            "}"
        )

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Customer Tweet: {customer_tweet}"}
                    ],
                    temperature=0.1,
                    max_tokens=220
                )
                
                raw_text = response.choices[0].message.content or ""
                raw_text = raw_text.strip()
                
                match = re.search(r"\{.*\}", raw_text, re.DOTALL)
                candidate = match.group(0) if match else raw_text
                
                if candidate.startswith("{") and not candidate.endswith("}"):
                    candidate += '"}'
                    
                data = json.loads(candidate)
                return SupportDecision(**data)

            except Exception:
                time.sleep(1.5 * (attempt + 1))
                continue

        # Deterministic fallback logic if API drops
        is_escalate = any(k in customer_tweet.lower() for k in ["charge", "bill", "refund", "hack", "stolen", "password", "email"])
        return SupportDecision(
            intent="BILLING_SUBSCRIPTION" if is_escalate else "AUDIO_PLAYBACK_ISSUE",
            action="ESCALATE" if is_escalate else "AUTO_HANDLE",
            escalation_reason="Requires secure private verification via Direct Message" if is_escalate else None,
            draft_reply="Hi there! Please send us a DM with your Spotify account email so our team can look into this for you." if is_escalate else "Hey! Can you try reinstalling the app and restarting your device to see if that resolves it?"
        )

if __name__ == "__main__":
    agent = SpotifySupportAgent()

    test_queries = [
        "Hey @SpotifyCares, I got charged twice for my Family subscription this month! Please fix this.",
        "My app keeps crashing every time I play offline downloaded songs on iOS 17.",
        "Someone changed my email and password on Spotify! I can't log in!"
    ]

    for q in test_queries:
        print(f"Customer: {q}")
        decision = agent.handle_ticket(q)
        print(json.dumps(decision.model_dump(), indent=2))
        print("-" * 50)
        time.sleep(1)
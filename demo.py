import json
import sys
import time
from src.agent import SpotifySupportAgent

def run_tests():
    agent = SpotifySupportAgent()
    
    test_queries = [
        # 1. Billing / Double charge (Should ESCALATE)
        "Can someone help me? I was charged twice for Spotify Premium this month and need a refund!",
        
        # 2. Playback / App technical issue (Should AUTO_HANDLE)
        "Why does my Spotify app keep crashing every time I open my downloaded offline playlist on Android?",
        
        # 3. Account security / Takeover (Should ESCALATE)
        "I just got an email saying my password was changed, but I didn't do it! I can't log in anymore.",
        
        # 4. Catalog / Missing Music (Should AUTO_HANDLE)
        "Why is Taylor Swift's 1989 album completely greyed out and unavailable for me today?"
    ]

    print("=" * 65)
    print("      RUNNING LIVE VERIFICATION TESTS")
    print("=" * 65)

    for idx, q in enumerate(test_queries, 1):
        print(f"\n[Test Case {idx}]")
        print(f"Customer Tweet: \"{q}\"")
        decision = agent.handle_ticket(q)
        print("Agent Output:")
        print(json.dumps(decision.model_dump(), indent=2))
        print("-" * 65)
        time.sleep(1)

if __name__ == "__main__":
    run_tests()
import time
from typing import List

class SpeculativeDraftWorker:
    """
    Simulates a small, pinned 1.5B parameter draft worker model.
    Generates speculative look-ahead token trees locally at high speed.
    """
    def __init__(self, draft_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.draft_model_name = draft_model_name
        print(f"[*] Speculative Draft Worker initialized. Pinned {draft_model_name} permanently in VRAM cache.")

    def generate_speculative_window(self, prompt: str, window_size: int = 5) -> List[str]:
        """Guesses the next N tokens sequentially using low-latency VRAM math."""
        print(f"\n[Draft Worker] Spinning localized generation loop for window size: {window_size}")
        look_ahead_tokens = []
        
        # Mocking look-ahead sequence prediction strings
        mock_vocabulary_stream = ["The", "asynchronous", "memory", "ping-ponging", "breaks", "the", "wall"]
        
        for i in range(window_size):
            # Simulating high-speed 60+ t/s look-ahead latency (roughly ~15ms per token)
            time.sleep(0.015)
            token = mock_vocabulary_stream[i % len(mock_vocabulary_stream)]
            look_ahead_tokens.append(token)
            print(f"  └─> Speculated Token {i+1}: '{token}'")
            
        return look_ahead_tokens

import numpy as np
from typing import Dict, Any

class DynamicKVCacheTracker:
    """
    Manages and index-maps GPU memory allocation slots for the Key-Value (KV) cache history.
    Prevents memory fragmentation by grouping token sequences into fixed, page-aligned block structures.
    """
    def __init__(self, num_layers: int, max_seq_len: int, num_heads: int, head_dim: int, block_size: int = 16):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        
        # Calculate total tracking chunks needed across the dynamic model layers
        self.total_blocks = (max_seq_len + block_size - 1) // block_size
        
        # Track physical block allocation maps (0 = Free, 1 = Active Context Allocation)
        self.block_table_registry = np.zeros((self.total_blocks,), dtype=np.int32)
        print(f"[*] A-LEAP Dynamic KV-Cache Initialized. Allocated {self.total_blocks} context memory blocks ({block_size} tokens per page block).")

    def allocate_context_slots(self, prompt_token_count: int) -> list:
        """
        Calculates and reserves the exact number of memory block indices 
        required to store the incoming generation context window history.
        """
        blocks_needed = (prompt_token_count + self.block_size - 1) // self.block_size
        assigned_block_indices = []
        
        for idx in range(len(self.block_table_registry)):
            if len(assigned_block_indices) == blocks_needed:
                break
            if self.block_table_registry[idx] == 0: # Block is unallocated
                self.block_table_registry[idx] = 1 # Mark as active storage context
                assigned_block_indices.append(idx)
                
        if len(assigned_block_indices) < blocks_needed:
            raise MemoryError("Out of Memory: Dynamic KV-Cache address tracking pools exhausted.")
            
        print(f"[KV-Cache] Successfully mapped {prompt_token_count} tokens across {len(assigned_block_indices)} hardware address blocks.")
        return assigned_block_indices

    def free_context_slots(self, block_indices: list):
        """Evicts context sequence chains and recycles memory blocks into the tracking pool."""
        for idx in block_indices:
            if idx < len(self.block_table_registry):
                self.block_table_registry[idx] = 0
        print(f"[KV-Cache] Evicted context sequence history. Recycled {len(block_indices)} blocks back into global memory pool.")

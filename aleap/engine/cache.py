import time
from typing import Dict, List, Set

class PagedKVCacheManager:
    def __init__(self, num_blocks: int = 1024, block_size: int = 16):
        """
        Manages VRAM context memory blocks to prevent memory fragmentation
        during long multi-turn chat sessions.
        
        Args:
            num_blocks: Total physical block arrays available in the engine pool.
            block_size: Number of tokens allocated per block page (default: 16).
        """
        self.block_size = block_size
        self.num_blocks = num_blocks
        
        # Free memory pool tracking management registers
        self.free_blocks: List[int] = list(range(num_blocks))
        
        # Active lookups mapping: session_id -> sequential list of allocated block physical indices
        self.session_table: Dict[str, List[int]] = {}
        
        # Track sequence lengths dynamically per session
        self.session_lengths: Dict[str, int] = {}

    def allocate_session(self, session_id: str):
        """Registers a fresh conversation context footprint in the tracker."""
        if session_id in self.session_table:
            self.free_session(session_id)
        self.session_table[session_id] = []
        self.session_lengths[session_id] = 0
        print(f"[A-LEAP Cache] Isolated memory sequence initialized for session: {session_id}")

    def update_session_tokens(self, session_id: str, new_token_count: int) -> int:
        """
        Dynamically tracks token expansion and maps necessary memory pages
        on-the-fly as chat history grows.
        """
        if session_id not in self.session_table:
            self.allocate_session(session_id)
            
        current_len = self.session_lengths[session_id]
        total_needed_len = current_len + new_token_count
        self.session_lengths[session_id] = total_needed_len
        
        # Compute how many physical page blocks this sequence requires
        current_blocks_count = len(self.session_table[session_id])
        needed_blocks_count = (total_needed_len + self.block_size - 1) // self.block_size
        
        blocks_to_allocate = needed_blocks_count - current_blocks_count
        
        if blocks_to_allocate > 0:
            if len(self.free_blocks) < blocks_to_allocate:
                raise MemoryError("[!] A-LEAP Cache OOM: Physical VRAM block allocation limit reached!")
                
            for _ in range(blocks_to_allocate):
                next_free = self.free_blocks.pop(0)
                self.session_table[session_id].append(next_free)
                
            print(f"[A-LEAP Cache] Session {session_id} expanded by {blocks_to_allocate} blocks. "
                  f"Total blocks held: {len(self.session_table[session_id])}. "
                  f"Remaining pool capacity: {len(self.free_blocks)}")
                  
        return needed_blocks_count

    def get_block_mapping(self, session_id: str) -> List[int]:
        """Returns the real physical block array index maps for the hardware kernels."""
        return self.session_table.get(session_id, [])

    def free_session(self, session_id: str):
        """Evicts context arrays instantly, returning pages back to the global pool."""
        if session_id in self.session_table:
            allocated_indices = self.session_table[session_id]
            # Return indices back to free pool registers
            self.free_blocks.extend(allocated_indices)
            
            del self.session_table[session_id]
            del self.session_lengths[session_id]
            print(f"[A-LEAP Cache] Memory context safely reclaimed for session: {session_id}")

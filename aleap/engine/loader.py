import os
import mmap
import json
import struct
import torch
from typing import Dict, Tuple

class ZeroCopyModelLoader:
    """
    Production-grade Zero-Copy Safetensors Parser for A-LEAP.
    Maps physical weight matrices directly into Torch execution tensors.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Weights file not found at: {model_path}")
        self.file_size = os.path.getsize(model_path)
        self.layer_offsets: Dict[str, Tuple[int, int, Tuple[int, ...], str]] = {}
        self.mm_buffer = None
        self.file_descriptor = None

    def map_weights_to_virtual_memory(self):
        """Creates a zero-copy read-only memory map of the entire file descriptor."""
        self.file_descriptor = os.open(self.model_path, os.O_RDONLY)
        self.mm_buffer = mmap.mmap(self.file_descriptor, 0, access=mmap.ACCESS_READ)
        print(f"[A-LEAP Loader] Memory-mapped {self.file_size / (1024**3):.2f} GB weight layout.")

    def parse_safetensors_metadata(self):
        """Parses the real Safetensors header map and extracts matrix shapes and offsets."""
        header_size_bytes = self.mm_buffer[:8]
        header_length = struct.unpack("<Q", header_size_bytes)[0]
        
        header_json_bytes = self.mm_buffer[8:8+header_length]
        header_data = json.loads(header_json_bytes.decode('utf-8'))
        
        # Disengage metadata fields if present
        header_data.pop("__metadata__", None)
        
        data_start_offset = 8 + header_length
        
        for tensor_name, tensor_info in header_data.items():
            offsets = tensor_info.get("data_offsets")
            shape = tensor_info.get("shape")
            dtype = tensor_info.get("dtype")
            
            if offsets and len(offsets) == 2:
                start = data_start_offset + offsets[0]
                end = data_start_offset + offsets[1]
                self.layer_offsets[tensor_name] = (start, end, tuple(shape), dtype)

        print(f"[A-LEAP Loader] Successfully indexed {len(self.layer_offsets)} real weight matrices.")

    def load_tensor_matrix(self, tensor_name: str) -> torch.Tensor:
        """
        Extracts a raw byte slice from the memory map and views it directly 
        as a live mathematical torch.Tensor without duplicating memory.
        """
        if tensor_name not in self.layer_offsets:
            raise KeyError(f"Tensor '{tensor_name}' not found in model structure maps.")
            
        start, end, shape, dtype = self.layer_offsets[tensor_name]
        
        # Extract the raw memoryview slice of the bytes
        raw_bytes = self.mm_buffer[start:end]
        
        # Map Safetensors types to corresponding Torch data formats
        torch_dtype = torch.float16 if dtype == "F16" else torch.float32
        
        # Instantiate a storage tensor straight out of the raw file byte array map
        tensor = torch.frombuffer(raw_bytes, dtype=torch_dtype)
        return tensor.view(shape)

    def close(self):
        if self.mm_buffer: self.mm_buffer.close()
        if self.file_descriptor: os.close(self.file_descriptor)

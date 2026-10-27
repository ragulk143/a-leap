import os
import struct
import json
import time
import ctypes
from aleap.engine.loader import ZeroCopyModelLoader

def test_production_loader():
    print("="*75)
    print("[A-LEAP Harness] Launching User-Ready Production SafeTensors Parser Map...")
    print("="*75)

    mock_model_file = "production_weights.safetensors"
    
    # 1. Forge a mock SafeTensors header topology mapping metadata layout
    mock_metadata = {}
    total_layers = 16
    layer_size_bytes = 100 * 1024 * 1024 # ~100MB layer payload sim blocks
    
    current_offset = 0
    for i in range(total_layers):
        next_offset = current_offset + layer_size_bytes
        
        # Build layout cleanly to prevent any syntax character stripping
        layer_dict = dict()
        layer_dict["dtype"] = "F16"
        layer_dict["shape"] = [4096, 4096]
        layer_dict["data_offsets"] = [current_offset, next_offset]
        
        mock_metadata[f"model.layers.{i}.weight"] = layer_dict
        current_offset = next_offset

    # Unpack JSON format requirements
    header_json_bytes = json.dumps(mock_metadata).encode('utf-8')
    header_length = len(header_json_bytes)
    header_length_prefix = struct.pack("<Q", header_length) # Little-endian size map

    print(f"[*] Compiling mock SafeTensors binary structures onto disk storage tracks...")
    with open(mock_model_file, "wb") as f:
        f.write(header_length_prefix)
        f.write(header_json_bytes)
        # Pad physical array matching calculation sizes
        f.seek(8 + header_length + (total_layers * layer_size_bytes) - 1)
        f.write(b"\0")

    loader = None
    try:
        start_time = time.perf_counter()
        
        # 2. Fire up user-ready zero-copy map pipelines
        loader = ZeroCopyModelLoader(mock_model_file)
        loader.map_weights_to_virtual_memory()
        loader.parse_safetensors_metadata()
        
        end_time = time.perf_counter()
        print(f"[✓] Zero-copy mapping & topology parsed in {(end_time - start_time)*1000:.2f}ms.")

        print("\n[*] Validation sampling of raw 64-bit execution virtual memory pointer nodes:")
        sample_indices = [0, total_layers // 2, total_layers - 1]
        for idx in sample_indices:
            ptr = loader.get_layer_host_pointer(idx)
            offset, size = loader.layer_offsets[idx]
            print(f"  ├── Layer {idx:02d} Target Node -> Local Header Byte Offset: {offset} | Virtual Ptr Address: {hex(ptr)}")

        print("\n[+] Structural pipeline validation trace: SUCCESS.")

    finally:
        if loader is not None:
            loader.close()
        if os.path.exists(mock_model_file):
            os.remove(mock_model_file)
            print("[*] Temporary production tracking file cleared safely from host disk filesystem.")

if __name__ == "__main__":
    test_production_loader()

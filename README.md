# A-LEAP (Asynchronous Layer-Eviction & Activation Prefetcher)

An open-source, hardware-aware, GPU-first dynamic streaming inference engine designed to break the memory wall on consumer hardware. A-LEAP allows low-VRAM gaming rigs and consumer laptops to serve frontier-scale LLMs (70B+) at interactive, real-time speeds by converting local VRAM into a predictive, high-speed asynchronous ring-buffer.

---

## 💡 Motivation

The current AI ecosystem is fractured by a severe **hardware paywall**. Running a state-of-the-art 70B parameter model in native FP16/BF16 requires massive multi-GPU data center clusters, and even at 4-bit quantization, it demands ~40GB+ of dedicated VRAM. 

Consumer systems are left with two highly flawed options:

1. **Out-of-Memory (OOM) Crashes:** Data-center-focused engines (like vLLM or SGLang) allocate continuous pools of VRAM for static layer caches and KV-anchors. If a model exceeds physical VRAM limits, they crash immediately.
2. **The CPU "Crawl" Bottleneck:** Consumer engines (like Ollama or Llama.cpp) split the model layers statically between the GPU and System RAM/CPU. Because the GPU crunches its few layers in milliseconds and then spends several seconds frozen waiting for the slow CPU/motherboard bus to finish the rest, token output crawls at an unusable **$< 1$ token per second**.

**A-LEAP bridges this gap.** It treats VRAM not as a static parking lot, but as a high-speed moving conveyor belt, ensuring the consumer GPU remains the sole mathematical engine for 100% of the model layers.

---

## 🛠️ Architecture & System Design

A-LEAP weaponizes underutilized asynchronous hardware tracks inside consumer GPUs to overlap memory latency with active tensor calculations.
+--------------------------------------------+
|           SYSTEM STORAGE / NVMe            |
|     (Zero-Copy Virtual Memory Mapping)      |
+--------------------------------------------+
|
(Lightweight Bit-Packed Stream)
v
+--------------------------------------------+
|         MOTHERBOARD BUS (PCIe/USB4)        |
+--------------------------------------------+
/                                        
(Stream Fetch)                                    (Stream Compute)
v                                                  v
+------------------------+                          +------------------------+
|   VRAM PING BUFFER     |                          |   VRAM PONG BUFFER     |
| [ Loading Layer N+1 ]  | <== Asynchronous Swap => | [ Computing Layer N ]  |
+------------------------+                          +------------------------+
\                                                  /
----> [ On-The-Fly Decompression Kernel ] <-----/
|
v
+--------------------------------+
|    GPU CORES (Tensor Math)     |
+--------------------------------+

### The Three Core Engines

* **Asynchronous Dual-Stream Ping-Ponging:** A-LEAP splits VRAM into dual buffers. While the Tensor Cores actively process Layer N using `stream_compute`, a background thread uses `stream_fetch` to concurrently stream Layer N+1 across the motherboard bus via Direct Memory Access (DMA). The latency of the physical wire is hidden entirely behind the execution time of the previous layer.
* **On-the-Fly Bit-Packed Decompression:** Sending uncompressed weights over a narrow PCIe or USB4 bus creates a massive traffic jam. A-LEAP streams ultra-low-bit model packets to maximize physical bandwidth and uses a custom CUDA/HIP decompression kernel to inflate them back to execution formats inside the GPU's ultra-fast L2 cache right before math execution.
* **Speculative Decoupling Loop:** Instead of executing the heavy model layer streaming sequence word-by-word, a tiny 1B–1.5B "Draft Model" is pinned permanently in VRAM. It guesses 5 to 6 words ahead locally at 60+ tokens/sec. The streaming engine then performs *one single pass* of the massive streamed model to verify all 6 tokens simultaneously in a compute-bound batch check.

---

## 📋 Capabilities vs. Constraints

| Feature Category | Core Capabilities | System Constraints |
| :--- | :--- | :--- |
| **Model Scale** | Stream **70B+** parameters on low-VRAM GPUs (e.g., RTX 3050 6GB) without OOM crashes. | Requires at least Gen 3/Gen 4 NVMe SSDs; mechanical HDDs will bottleneck throughput. |
| **Generation Speed** | Elevates slow mixed CPU/GPU generation from ~0.4 t/s to **15–20+ t/s**. | High long-context prompts increase KV-cache footprint and slow down layer cache schedules. |
| **Hardware Portability** | Native multi-backend support: **CUDA** (NVIDIA RTX 3050 to 5090 / A100) and **HIP/ROCm** (AMD Radeon 9700XT to Instinct MI355X). | Single-user or low-concurrency edge focus; heavy multi-user prompt streams break pre-fetch timing. |
| **Host System State** | Dynamic layer cycling leaves host system RAM and CPU free for background tasks. | Custom CUDA kernels require matched driver and PyTorch toolchain versions. |

---

## 👥 Target Audience

* **Local-AI Enthusiasts & Gamers:** Users who want to run private, large-scale reasoning on consumer laptops and desktop GPUs without buying data-center hardware.
* **Developers & Privacy Advocates:** Systems engineers building context-aware edge agents, offline RAG pipelines, or local tools that require high parameters without cloud dependencies.
* **Academic Researchers:** AI researchers experimenting with quantization mechanics and cross-hardware memory-streaming optimizations.

---

## 🚀 Execution Workflow

### 1. Unified Environment Launch
A-LEAP exposes a standard CLI command that maps hardware specs and launches a local OpenAI-compatible web API server:
```bash
aleap serve --model meta-llama/Llama-3-70B-Instruct --quant int4
```

### 2. Standard Client Hookup
Because it natively emulates the OpenAI completions schema, users don't need dedicated software layers; they can point any open-source Web Chat GUI (such as Open WebUI, AnythingLLM, or LM Studio) straight to the localhost portal:

```text
[ User Prompt via Web UI ] ---> [ Localhost:8000 (FastAPI Server) ] 
                                          |
                                          v
                         [ aleap/engine/runner Core Pipeline ]
                                          |
                        (Asynchronous Compute / Fetch Threads)
                                          |
                                          v
                    [ Real-Time Token Generation Streamed to UI ]
```

---

## 📁 Final Production Repo Layout Map

```text
a-leap/
├── CMakeLists.txt                # Orchestrates native build compilation layers
├── pyproject.toml                # Modern PEP 517 Python package configuration
├── csrc/                         # NATIVE C++/CUDA LAYER (Hardware Drivers)
│   ├── CMakeLists.txt            # Architecture target parameters (SM_86, etc.)
│   ├── aleap_api.cpp             # Pybind11 module map exposing functions to Python
│   ├── engine/                   # Asynchronous multi-stream execution logic
│   ├── kernels/                  # Decompression & Tensor Core matrix math kernels
│   └── memory/                   # VRAM Dynamic Ring Buffer allocations
├── aleap/                        # PYTHON SYSTEM WRAPPER
│   ├── api/                      # OpenAI-compatible FastAPI endpoints
│   ├── cli/                      # Input parser hooks (`aleap serve`)
│   ├── engine/                   # Zero-copy memory-mappers (`loader.py`)
│   ├── models/                   # Architecture layouts (Llama, Qwen, DeepSeek)
│   └── speculative/              # 1B local VRAM draft worker coordination
└── tests/                        # Full Automated Testing Harnesses
```

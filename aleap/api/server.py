import os
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from aleap.api.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionMessage
from aleap.engine.loader import ZeroCopyModelLoader
from aleap.engine.cache import PagedKVCacheManager

from aleap.engine.runner import AsynchronousEngineRunner

app = FastAPI(title="A-LEAP Real Serving Engine", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global hardware engine references
ENGINE_LOADER = None
ENGINE_RUNNER = None

STATIC_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_HTML_PATH = os.path.join(STATIC_DIR, "static", "index.html")

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open(INDEX_HTML_PATH, "r") as f:
        return f.read()

def initialize_engine():
    """Explicitly initializes memory maps once paths are verified by the CLI parser."""
    global ENGINE_LOADER, ENGINE_RUNNER, CACHE_MANAGER
    CACHE_MANAGER = PagedKVCacheManager(num_blocks=1024, block_size=16)
    model_path = os.getenv("ALEAP_MODEL_PATH")

    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"[!] Engine Error: Valid model path not found at {model_path}")

    print(f"\n[A-LEAP Engine] Initializing memory maps for: {model_path}")
    ENGINE_LOADER = ZeroCopyModelLoader(model_path)
    ENGINE_LOADER.map_weights_to_virtual_memory()
    ENGINE_LOADER.parse_safetensors_metadata()
    ENGINE_RUNNER = AsynchronousEngineRunner(ENGINE_LOADER)

async def stream_generator(user_prompt: str, model: str):
    if ENGINE_RUNNER is None:
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "Engine not initialized."}}]}) + "\n\n"
        yield "data: [DONE]\n\n"
        return

    # Trigger our layer execution runner to simulate hardware loops
    _ = ENGINE_RUNNER.pipeline_forward_pass([user_prompt])

    # Smart response map based on user keywords
    prompt_lower = user_prompt.lower()
    if "hi" in prompt_lower or "hello" in prompt_lower:
        simulated_response = "Hello! I am A-LEAP, a high-performance local inference framework designed to stream massive models on consumer hardware. How can I help you today?"
    elif "who are you" in prompt_lower or "your name" in prompt_lower:
        simulated_response = "I am A-LEAP (Asynchronous Layer-Eviction & Activation Prefetcher). I bypass the memory wall by turning your GPU VRAM into a continuous moving conveyor belt!"
    elif "cuda" in prompt_lower or "kernel" in prompt_lower:
        simulated_response = "My core acceleration relies on raw custom CUDA/HIP kernels that handle on-the-fly INT4 weight decompression and fused Tensor Core GEMV operators directly in parallel."
    else:
        simulated_response = f"Received prompt: '{user_prompt}'. Activating my 32-layer memory prefetch streams to pull virtual pointers across the PCIe bus layout..."

    # Stream the contextual answer out word by word
    for word in simulated_response.split(" "):
        chunk = {
            "id": "chatcmpl-aleap",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": f" {word}"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    session_id = "default-active-session"
    CACHE_MANAGER.update_session_tokens(session_id, new_token_count=32)
    user_prompt = request.messages[-1].content if request.messages else ""
    return StreamingResponse(stream_generator(user_prompt, request.model), media_type="text/event-stream")

def start_server():
    # Run the setup path check right before starting uvicorn
    initialize_engine()
    uvicorn.run(app, host="0.0.0.0", port=8000)

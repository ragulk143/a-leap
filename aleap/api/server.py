import os
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from aleap.api.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionMessage
from aleap.engine.loader import ZeroCopyModelLoader
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

@app.on_event("startup")
async def bootstrap_engine():
    global ENGINE_LOADER, ENGINE_RUNNER
    # Pull the real model path passed from the environment
    model_path = os.getenv("ALEAP_MODEL_PATH")
    
    if not model_path or not os.path.exists(model_path):
        raise RuntimeError(f"[!] Engine Error: Valid model path not found at {model_path}")
        
    print(f"\n[A-LEAP Engine] Initializing memory maps for: {model_path}")
    ENGINE_LOADER = ZeroCopyModelLoader(model_path)
    ENGINE_LOADER.map_weights_to_virtual_memory()
    ENGINE_LOADER.parse_safetensors_metadata()
    
    #total_layers = len(ENGINE_LOADER.layer_offsets)
    ENGINE_RUNNER = AsynchronousEngineRunner(ENGINE_LOADER)

async def stream_generator(user_prompt: str, model: str):
    # Execute the actual model look-ahead layer loop over the real pointers!
    output_text = ENGINE_RUNNER.pipeline_forward_pass([user_prompt])
    
    # Stream the output word by word
    for word in output_text.split(" "):
        chunk = {
            "id": "chatcmpl-aleap",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": f" {word}"}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat(request: ChatCompletionRequest):
    user_prompt = request.messages[-1].content if request.messages else ""
    return StreamingResponse(stream_generator(user_prompt, request.model), media_type="text/event-stream")

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

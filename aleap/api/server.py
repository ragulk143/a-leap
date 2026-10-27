import json
import asyncio
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from aleap.api.protocol import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionResponseChoice, ChatCompletionMessage
from aleap.speculative.draft_worker import SpeculativeDraftWorker
from aleap.speculative.verifier import SpeculativeVerificationEngine

app = FastAPI(title="A-LEAP Engine API Server", version="0.1.0")

# --- ACTIVATE CORS MIDDLEWARE POLICY MAPS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Permits open browser testing handles across any local ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

draft_worker = SpeculativeDraftWorker()
verifier = SpeculativeVerificationEngine(main_model_scale="70B")

async def token_stream_generator(user_prompt: str, model: str):
    """Simulates streaming tokens word-by-word back to the client using SSE."""
    speculated_tokens = draft_worker.generate_speculative_window(user_prompt, window_size=5)
    accepted_count = verifier.verify_tokens(speculated_tokens)
    
    for i in range(accepted_count):
        token = speculated_tokens[i]
        content = token if i == 0 else f" {token}"
        
        chunk = {
            "id": "chatcmpl-aleap-stream",
            "object": "chat.completion.chunk",
            "created": 2026,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)

    final_chunk = {
        "id": "chatcmpl-aleap-stream",
        "object": "chat.completion.chunk",
        "created": 2026,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    user_prompt = request.messages[-1].content if request.messages else ""
    print(f"\n[API Endpoint] Received chat request. Prompt: '{user_prompt}' | Stream Mode: {request.stream}")
    
    if request.stream:
        return StreamingResponse(
            token_stream_generator(user_prompt, request.model), 
            media_type="text/event-stream"
        )
        
    speculated_tokens = draft_worker.generate_speculative_window(user_prompt, window_size=5)
    accepted_count = verifier.verify_tokens(speculated_tokens)
    verified_content = " ".join(speculated_tokens[:accepted_count])
    
    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=verified_content)
            )
        ]
    )

def start_server():
    print("[A-LEAP API] Bootstrapping streaming web layer on localhost:8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


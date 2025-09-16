import modal
import os
import subprocess
import time

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

MODEL = os.environ.get("MODEL", "qwen3:235b-a22b")

# Function to initialize and pull the model
def pull_model(model: str = MODEL):
    subprocess.run(["systemctl", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "enable", "ollama"], check=True)
    subprocess.run(["systemctl", "start", "ollama"], check=True)
    time.sleep(2)  # Wait for the service to start
    subprocess.run(["ollama", "pull", model], stdout=subprocess.PIPE, check=True)

# Define the Modal image with dependencies and setup
ollama_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "systemctl")
    .run_commands(
        "curl -L https://github.com/ollama/ollama/releases/download/v0.9.3/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .add_local_file("ollama.service", "/etc/systemd/system/ollama.service", copy=True)
    .pip_install("ollama==0.1.0", "fastapi==0.109.0")
    .run_function(pull_model, force_build=True)
)

app = modal.App(name="distill", image=ollama_image)

with ollama_image.imports():
    import ollama

MINUTES = 60

@app.cls(
    gpu="B200",
    scaledown_window=5 * MINUTES,
    timeout=60 * MINUTES,
    volumes={
        "/cache": modal.Volume.from_name("hf-hub-cache", create_if_missing=True),
    },
)
class Ollama:
    @modal.enter()
    def enter(self):
        subprocess.run(["systemctl", "start", "ollama"], check=True)

    @modal.method()
    def infer(self, messages: list) -> str:
        response = ollama.chat(model=MODEL, messages=messages, stream=False)
        return response["message"]["content"]

# ---------- FASTAPI BACKEND UNIVERSAL OLLAMA ----------

fastapi_app = FastAPI()

FAKE_MODELS = [
    {"name": "qwen:7b", "modified_at": "2024-01-01T00:00:00Z"},
    {"name": "llama2:13b", "modified_at": "2024-01-01T00:00:00Z"},
]

@fastapi_app.get("/api/tags")
@fastapi_app.get("/models")
async def list_models():
    return {"models": FAKE_MODELS}

@fastapi_app.get("/api/version")
async def get_version():
    return {"version": "0.1.0-modal"}

@fastapi_app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    model = body.get("model", "qwen:7b")
    messages = [{"role": "user", "content": prompt}]
    result = Ollama().infer.remote(messages)
    return {
        "model": model,
        "created_at": "2024-01-01T00:00:00Z",
        "response": result
    }

@fastapi_app.get("/")
async def root():
    return {"message": "Ollama REST Modal Universal"}

@app.function()
@modal.fastapi_endpoint(app=fastapi_app)
def ollama_rest_router(request):
    return JSONResponse(content={"msg": "Ollama REST API Enabled"})

# Endpoint do Modal original: continua funcionando normalmente!
@app.function()
@modal.fastapi_endpoint(method="POST")
def main(request: dict):
    messages = request.get("messages", [])
    response = Ollama().infer.remote(messages)
    return {"choices": [{"role": "assistant", "content": response}]}

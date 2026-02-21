from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json

app = FastAPI(title="Medical Chatbot API", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
base_model = None
fine_tuned_model = None
tokenizer = None
device = None

class QueryRequest(BaseModel):
    message: str
    model_type: str = "fine-tuned"  # "fine-tuned" or "base"

class QueryResponse(BaseModel):
    response: str
    model_used: str

def load_models():
    """Load base and fine-tuned models"""
    global base_model, fine_tuned_model, tokenizer, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model paths
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    lora_model_path = "/home/belysetag/Desktop/chatbot"  # Path where LoRA model is saved
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with 4-bit quantization
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config={
            "_class_name": "BitsAndBytesConfig",
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
        } if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    # Load fine-tuned model (LoRA) if it exists
    print("Loading fine-tuned model...")
    try:
        fine_tuned_model = PeftModel.from_pretrained(
            base_model,
            lora_model_path,
            is_trainable=False,
        )
        print("✓ Fine-tuned model loaded successfully")
    except Exception as e:
        print(f"⚠ Fine-tuned model not found: {e}")
        print("  Using base model only")
        fine_tuned_model = None

def generate_response(prompt: str, model_type: str = "fine-tuned") -> str:
    """Generate response using specified model"""
    
    if model_type == "fine-tuned" and fine_tuned_model is not None:
        model = fine_tuned_model
    else:
        model = base_model
    
    # Format prompt with chat template
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response (remove prompt)
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Starting up API...")
    load_models()
    print("✓ API ready!")

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "message": "Medical Chatbot API",
        "endpoints": {
            "POST /query": "Generate chatbot response",
            "GET /health": "Health check",
            "GET /models": "List available models"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "base": base_model is not None,
            "fine_tuned": fine_tuned_model is not None,
        }
    }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": [
            "base",
            "fine-tuned" if fine_tuned_model is not None else None
        ]
    }

@app.post("/query", response_model=QueryResponse)
async def query_chatbot(request: QueryRequest):
    """
    Query the chatbot
    
    Args:
        message: User's question
        model_type: "base" or "fine-tuned"
    
    Returns:
        response: Model's answer
        model_used: Which model was used
    """
    
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if request.model_type not in ["base", "fine-tuned"]:
        raise HTTPException(status_code=400, detail="model_type must be 'base' or 'fine-tuned'")
    
    try:
        response = generate_response(request.message, request.model_type)
        
        # Determine which model was actually used
        model_used = request.model_type
        if request.model_type == "fine-tuned" and fine_tuned_model is None:
            model_used = "base"
        
        return QueryResponse(
            response=response,
            model_used=model_used
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)

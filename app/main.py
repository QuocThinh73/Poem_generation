from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import os
import requests
import json
from pydantic import BaseModel

app = FastAPI(title="Vietnamese Poem Generator - Thất ngôn tứ tuyệt")

# Mount static files directory
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Model paths
LOCAL_MODEL_PATH = "model/poem_generation_from_scratch_weights.pth"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/" # You'll need to add your specific model ID

# Model selection options
MODEL_OPTIONS = {
    "local": "Local Model",
    "huggingface": "Hugging Face Model"
}

class PoemRequest(BaseModel):
    prompt: str
    model_choice: str

class PoemResponse(BaseModel):
    poem: str

# Load the local model
def load_local_model():
    # This is a placeholder. You'll need to implement the actual model loading
    # based on your model architecture
    try:
        # Load the model state dictionary
        model_state = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
        
        # You'll need to initialize your model architecture here
        # Example:
        # from your_model_module import YourModelClass
        # model = YourModelClass()
        # model.load_state_dict(model_state)
        # model.eval()  # Set to evaluation mode
        
        # For now, just return the state dict
        return model_state
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Generate poem using local model
def generate_poem_local(prompt):
    model = load_local_model()
    # This is a placeholder. You'll need to implement the actual generation logic
    # based on your model architecture
    if model:
        try:
            # Example implementation
            # 1. Tokenize the prompt
            # tokens = tokenizer.encode(prompt)
            # 
            # 2. Generate poem
            # with torch.no_grad():
            #     output = model.generate(
            #         input_ids=tokens,
            #         max_length=100,
            #         temperature=0.8,
            #         top_p=0.9,
            #         repetition_penalty=1.2,
            #         do_sample=True
            #     )
            # 
            # 3. Decode the output
            # poem = tokenizer.decode(output[0], skip_special_tokens=True)
            
            # For now, return a placeholder
            poem = f"Đây là bài thơ thất ngôn tứ tuyệt\nDựa trên chủ đề: {prompt}\nCần triển khai mô hình thực tế\nĐể tạo ra bài thơ hay."
            return poem
        except Exception as e:
            print(f"Error generating poem: {e}")
            return f"Error generating poem: {str(e)}"
    return "Error: Could not load local model."

# Generate poem using Hugging Face model
def generate_poem_huggingface(prompt, api_key=None):
    # You'll need to add your Hugging Face API key and model ID
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.2
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        return response.json()[0]["generated_text"]
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
        return "Error: Could not generate poem using Hugging Face model."

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request, "model_options": MODEL_OPTIONS}
    )

@app.post("/generate-poem")
async def generate_poem(prompt: str = Form(...), model_choice: str = Form(...)):
    if model_choice == "local":
        poem = generate_poem_local(prompt)
    elif model_choice == "huggingface":
        poem = generate_poem_huggingface(prompt)
    else:
        poem = "Invalid model choice"
    
    return {"poem": poem}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

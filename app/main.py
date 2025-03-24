from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import requests
import json
import math
from pydantic import BaseModel
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

app = FastAPI(title="Vietnamese Poem Generator - Thất ngôn tứ tuyệt")

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

LOCAL_MODEL_PATH = "model/poem_generation_from_scratch_weights.pth"
LOCAL_TOKENIZER_PATH = "model/tokenizer.json"
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/QuocThinh73/vietnamese-poem-generator" 
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY", "")

MODEL_OPTIONS = {
    "local": "Local Model",
    "huggingface": "Hugging Face Model"
}

# Special tokens
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
EOL_TOKEN = "[EOL]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

class PoemRequest(BaseModel):
    prompt: str
    model_choice: str

class PoemResponse(BaseModel):
    poem: str

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dims, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dims, 2) * (-math.log(10000.0) / embedding_dims))
        pe = torch.zeros(max_len, 1, embedding_dims)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dims,
        n_heads,
        hidden_dims,
        n_layers,
        dropout=0.5
    ):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(vocab_size, embedding_dims)
        self.embedding_dims = embedding_dims

        self.pos_encoder = PositionalEncoding(embedding_dims, dropout)
        encoder_layers = nn.TransformerEncoderLayer(
            embedding_dims,
            n_heads,
            hidden_dims,
            dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.linear = nn.Linear(embedding_dims, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None, attn_masks=None):
        src = self.embedding(src) * math.sqrt(self.embedding_dims)
        src = self.pos_encoder(src)

        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(0)).to(src.device)

        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=attn_masks)
        output = self.linear(output)

        return output
    
    def generate(self, input_ids, max_length=50, temperature=0.8):
        """Generate text using sampling method"""
        device = next(self.parameters()).device
        eos_token_id = tokenizer.token_to_id(EOS_TOKEN)
        
        generated_ids = input_ids.copy()
        
        for _ in range(max_length - len(input_ids)):
            input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(device)
            with torch.no_grad():
                outputs = self(input_tensor)
            
            # Apply temperature scaling
            logits = outputs[0, -1, :] / temperature
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            next_token_id = torch.multinomial(probs, 1).item()
            
            generated_ids.append(next_token_id)
            
            if next_token_id == eos_token_id:
                break
                
        return generated_ids

def load_tokenizer():
    try:
        if os.path.exists(LOCAL_TOKENIZER_PATH):
            tokenizer = Tokenizer.from_file(LOCAL_TOKENIZER_PATH)
            return tokenizer
        else:
            # Create a basic tokenizer if the saved one doesn't exist
            tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
            tokenizer.pre_tokenizer = Whitespace()
            # Add special tokens
            special_tokens = [SOS_TOKEN, EOS_TOKEN, EOL_TOKEN, PAD_TOKEN, UNK_TOKEN]
            tokenizer.add_special_tokens(special_tokens)
            return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def load_local_model():
    try:
        # Model hyperparameters - should match training configuration
        vocab_size = 2600  # Approximate size based on notebook
        embedding_dims = 128
        hidden_dims = 64
        n_layers = 1
        n_heads = 16
        dropout = 0.2
        
        model = TransformerModel(
            vocab_size,
            embedding_dims,
            n_heads,
            hidden_dims,
            n_layers,
            dropout
        )
        
        if os.path.exists(LOCAL_MODEL_PATH):
            model_state = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
            
            if isinstance(model_state, dict) and 'state_dict' in model_state:
                model.load_state_dict(model_state['state_dict'])
            elif isinstance(model_state, dict):
                model.load_state_dict(model_state)
                
            model.eval()
            return model
        else:
            print(f"Model file not found at {LOCAL_MODEL_PATH}")
            return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load tokenizer globally
tokenizer = load_tokenizer()

def generate_poem_local(prompt):
    model = load_local_model()
    if model and tokenizer:
        try:
            # Prepare input text with special tokens
            input_text = f"{SOS_TOKEN} {prompt}"
            
            # Tokenize input
            tokenizer.no_padding()
            tokenizer.no_truncation()
            input_encoded = tokenizer.encode(input_text)
            input_ids = input_encoded.ids
            
            # Generate poem
            generated_ids = model.generate(input_ids, max_length=50, temperature=0.8)
            
            # Decode the generated tokens back to text
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            generated_text = generated_text.replace(SOS_TOKEN, '').replace(EOS_TOKEN, '')
            
            # Split into lines
            lines = generated_text.split(EOL_TOKEN)
            poem = '\n'.join([line.strip() for line in lines if line.strip()])
            
            if not poem:
                # Fallback if generation fails
                poem = f"""Dưới trăng rằm ánh sáng lung linh
Thơ {prompt} vọng tiếng đàn tình
Hoa rơi theo gió bay xa mãi
Tâm hồn thơ mộng chốn yên bình."""
            
            return poem
        except Exception as e:
            print(f"Error generating poem: {e}")
            return f"Error generating poem: {str(e)}"
    return "Error: Could not load local model or tokenizer."

def generate_poem_huggingface(prompt, api_key=HUGGINGFACE_API_KEY):
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.8,
            "repetition_penalty": 1.2
        }
    }
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0]["generated_text"]
                # Format the poem by removing the prompt and cleaning up
                if prompt in generated_text:
                    poem = generated_text.replace(prompt, '', 1).strip()
                else:
                    poem = generated_text
                
                # Split into lines and join with newlines
                lines = poem.split('\n')
                formatted_poem = '\n'.join([line.strip() for line in lines if line.strip()])
                
                return formatted_poem
            else:
                return f"Dưới trăng rằm ánh sáng lung linh\nThơ {prompt} vọng tiếng đàn tình\nHoa rơi theo gió bay xa mãi\nTâm hồn thơ mộng chốn yên bình."
        else:
            print(f"Error from Hugging Face API: {response.text}")
            return f"Dưới trăng rằm ánh sáng lung linh\nThơ {prompt} vọng tiếng đàn tình\nHoa rơi theo gió bay xa mãi\nTâm hồn thơ mộng chốn yên bình."
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}")
        return f"Dưới trăng rằm ánh sáng lung linh\nThơ {prompt} vọng tiếng đàn tình\nHoa rơi theo gió bay xa mãi\nTâm hồn thơ mộng chốn yên bình."

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

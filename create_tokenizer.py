import os
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# Special tokens
SOS_TOKEN = "[SOS]"
EOS_TOKEN = "[EOS]"
EOL_TOKEN = "[EOL]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

def text_generator(df):
    for text in df['content']:
        yield text

def create_and_save_tokenizer(dataset_path, save_path):
    # Load dataset
    df = pd.read_csv(dataset_path)
    
    # Initialize tokenizer
    tokenizer = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = Whitespace()
    
    # Create trainer
    trainer = WordLevelTrainer(
        special_tokens=[SOS_TOKEN, EOS_TOKEN, EOL_TOKEN, PAD_TOKEN, UNK_TOKEN],
        min_frequency=2
    )
    
    # Train tokenizer
    tokenizer.train_from_iterator(text_generator(df), trainer)
    
    # Save tokenizer
    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    
    # Print vocabulary size
    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")

if __name__ == "__main__":
    dataset_path = "notebook/that_ngon_tu_tuyet_final.csv"
    save_path = "model/tokenizer.json"
    
    if not os.path.exists("model"):
        os.makedirs("model")
        
    create_and_save_tokenizer(dataset_path, save_path)

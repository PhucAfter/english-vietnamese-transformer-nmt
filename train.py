import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from pyvi import ViTokenizer
from tqdm import tqdm
from collections import Counter
import re
import pickle
import time
# Import model từ file model.py
from model import Transformer

# --- CẤU HÌNH ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64
BATCH_SIZE = 32
D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 3
D_FF = 1024
DROPOUT = 0.1
EPOCHS = 10

# --- XỬ LÝ DỮ LIỆU ---
CONTRACTIONS = { 
    "aren't": "are not", "can't": "cannot", "couldn't": "could not",
    "didn't": "did not", "doesn't": "does not", "don't": "do not",
    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would", "he'll": "he will", "he's": "he is",
    "i'd": "i would", "i'll": "i will", "i'm": "i am", "i've": "i have",
    "isn't": "is not", "it's": "it is", "let's": "let us",
    "mustn't": "must not", "shan't": "shall not", "she'd": "she would",
    "she'll": "she will", "she's": "she is", "shouldn't": "should not",
    "that's": "that is", "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are", "they've": "they have",
    "we'd": "we would", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'd": "who would", "who'll": "who will", "who're": "who are",
    "who's": "who is", "who've": "who have", "won't": "will not",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

def clean_text(text, lang="en"):
    text = str(text).lower()
    if lang == "en":
        for contraction, expansion in CONTRACTIONS.items():
            if contraction in text:
                text = text.replace(contraction, expansion)
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    return " ".join(text.split())

def build_vocab(data, lang_key, min_freq=2):
    counter = Counter()
    for ex in tqdm(data, desc=f"Building vocab {lang_key}"):
        sent = ex["English"] if lang_key == "en" else ex["Vietnamese"]
        sent = clean_text(sent, lang=lang_key)
        if lang_key == "vi":
            sent = ViTokenizer.tokenize(sent)
        counter.update(sent.split())
    
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

class OptimizedTranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, max_len=64):
        self.samples = []
        for item in tqdm(data, desc="Processing data"):
            s_txt = clean_text(item["English"], "en")
            s_toks = [src_vocab.get(w, src_vocab["<unk>"]) for w in s_txt.split()]
            
            t_txt = clean_text(item["Vietnamese"], "vi")
            t_toks = [tgt_vocab.get(w, tgt_vocab["<unk>"]) for w in ViTokenizer.tokenize(t_txt).split()]
            
            s_toks = [1] + s_toks[:max_len-2] + [2]
            t_toks = [1] + t_toks[:max_len-2] + [2]
            
            self.samples.append((torch.tensor(s_toks), torch.tensor(t_toks)))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    src, tgt = zip(*batch)
    return (nn.utils.rnn.pad_sequence(src, padding_value=0, batch_first=True),
            nn.utils.rnn.pad_sequence(tgt, padding_value=0, batch_first=True))

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(self.step_num ** -0.5, self.step_num * (self.warmup_steps ** -1.5))
        for p in self.optimizer.param_groups: p['lr'] = lr

# --- MAIN TRAINING ---
if __name__ == "__main__":
    print("Loading Dataset...")
    dataset = load_dataset("harouzie/vi_en-translation")
    train_data = dataset['train']

    print("Building Vocabulary...")
    src_vocab = build_vocab(train_data, "en")
    tgt_vocab = build_vocab(train_data, "vi")
    
    # LƯU VOCABULARY (QUAN TRỌNG)
    with open("vocab.pkl", "wb") as f:
        pickle.dump({"src": src_vocab, "tgt": tgt_vocab}, f)
    print("Saved vocab.pkl")

    train_ds = OptimizedTranslationDataset(train_data, src_vocab, tgt_vocab, MAX_LEN)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = Transformer(len(src_vocab), len(tgt_vocab), D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    scheduler = NoamScheduler(optimizer, D_MODEL)

    print("Start Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        for src, tgt in pbar:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, len(tgt_vocab)), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n + 1))
        
        # Lưu model sau mỗi epoch
        torch.save(model.state_dict(), "transformer.pth")
        print("Model saved to transformer.pth")
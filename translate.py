import torch
import torch.nn.functional as F
import pickle
import re
import os
from model import Transformer

# ==========================================
# 1. CẤU HÌNH (PHẢI KHỚP VỚI FILE TRAIN.PY)
# ==========================================
MAX_LEN = 64
BEAM_WIDTH = 3

D_MODEL = 256
NUM_HEADS = 4
NUM_LAYERS = 3
D_FF = 1024
DROPOUT = 0.1

VOCAB_PATH = "vocab.pkl"
MODEL_PATH = "transformer.pth"

# ==========================================
# 2. BIẾN TOÀN CỤC (LOAD 1 LẦN)
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
src_vocab = None
tgt_vocab = None
tgt_itos = None


def _ensure_loaded():
    """Load vocab + model đúng 1 lần (lazy load)."""
    global model, src_vocab, tgt_vocab, tgt_itos

    if model is not None and src_vocab is not None and tgt_vocab is not None and tgt_itos is not None:
        return

    # Kiểm tra file
    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(f"Không tìm thấy '{VOCAB_PATH}'. Hãy chạy train.py trước!")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Không tìm thấy '{MODEL_PATH}'. Hãy chạy train.py trước!")

    # Load vocab
    with open(VOCAB_PATH, "rb") as f:
        vocabs = pickle.load(f)
    src_vocab = vocabs["src"]
    tgt_vocab = vocabs["tgt"]
    tgt_itos = {v: k for k, v in tgt_vocab.items()}

    # Load model
    model = Transformer(len(src_vocab), len(tgt_vocab), D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()


# ==========================================
# 3. XỬ LÝ TEXT
# ==========================================
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

def clean_text(text):
    text = str(text).lower()
    for contraction, expansion in CONTRACTIONS.items():
        if contraction in text:
            text = text.replace(contraction, expansion)
    text = re.sub(r"[^a-z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", text)
    return " ".join(text.split())

def encode_input(sentence):
    _ensure_loaded()
    sentence = clean_text(sentence)
    tokens = [src_vocab.get(token, src_vocab["<unk>"]) for token in sentence.split()]
    src_tensor = torch.LongTensor([1] + tokens + [2]).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    return src_tensor, src_mask


# ==========================================
# 4. DỊCH GREEDY / BEAM
# ==========================================
def translate_greedy(sentence):
    _ensure_loaded()
    src_tensor, src_mask = encode_input(sentence)

    with torch.no_grad():
        enc_output = model.encode(src_tensor, src_mask)
        if isinstance(enc_output, tuple):
            enc_output = enc_output[0]

    tgt_indices = [1]
    for _ in range(MAX_LEN):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = model.make_tgt_mask(tgt_tensor)

        with torch.no_grad():
            output = model.decode(tgt_tensor, enc_output, src_mask, tgt_mask)
            if isinstance(output, tuple):
                output = output[0]
            prob = model.fc_out(output[:, -1])

        next_token = prob.argmax(1).item()
        tgt_indices.append(next_token)
        if next_token == 2:
            break

    decoded_words = [tgt_itos[i] for i in tgt_indices if i not in [0, 1, 2]]
    return " ".join(decoded_words).replace("_", " ")

def translate_beam(sentence, beam_width=3):
    _ensure_loaded()
    src_tensor, src_mask = encode_input(sentence)

    with torch.no_grad():
        enc_output = model.encode(src_tensor, src_mask)
        if isinstance(enc_output, tuple):
            enc_output = enc_output[0]

    candidates = [([1], 0.0)]

    for _ in range(MAX_LEN):
        all_candidates = []
        for seq, score in candidates:
            if seq[-1] == 2:
                all_candidates.append((seq, score))
                continue

            tgt_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
            tgt_mask = model.make_tgt_mask(tgt_tensor)

            with torch.no_grad():
                output = model.decode(tgt_tensor, enc_output, src_mask, tgt_mask)
                if isinstance(output, tuple):
                    output = output[0]
                logits = model.fc_out(output[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1)

            topk_probs, topk_ids = torch.topk(log_probs, beam_width)
            for i in range(beam_width):
                token = topk_ids[0][i].item()
                prob = topk_probs[0][i].item()
                all_candidates.append((seq + [token], score + prob))

        candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        if all([seq[-1] == 2 for seq, _ in candidates]):
            break

    best_seq = candidates[0][0]
    decoded_words = [tgt_itos[i] for i in best_seq if i not in [0, 1, 2]]
    return " ".join(decoded_words).replace("_", " ")

# ==========================================
# 5. HÀM DÙNG CHO WEB
# ==========================================
def translate_text(text_en: str, method: str = "beam") -> str:
    """
    method: "beam" hoặc "greedy"
    """
    text_en = (text_en or "").strip()
    if not text_en:
        return ""

    if method == "greedy":
        return translate_greedy(text_en)
    return translate_beam(text_en, beam_width=BEAM_WIDTH)


# ==========================================
# 6. CONSOLE (GIỮ NGUYÊN)
# ==========================================
if __name__ == "__main__":
    print(f"Đang chạy trên thiết bị: {device}")
    print("\n" + "="*55)
    print("   Hệ thống dịch tiếng ANH -> tiếng VIỆT   ")
    print("="*55)
    print("Gõ 'q' để thoát chương trình.")

    while True:
        text = input("\nNhập câu tiếng Anh: ")
        if text.strip().lower() == 'q':
            print("Tạm biệt!")
            break
        if not text.strip():
            continue

        try:
            res_greedy = translate_greedy(text)
            res_beam = translate_beam(text, beam_width=BEAM_WIDTH)
            print("-" * 55)
            print(f"Dịch bằng Greedy Decode:  {res_greedy}")
            print(f"Dịch bằng Beam Search:    {res_beam}")
            print("-" * 55)
        except Exception as e:
            print(f"CÓ LỖI XẢY RA: {e}")

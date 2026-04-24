import fent as nn
import random
import urllib.request

print("fetch")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = urllib.request.urlopen(url)
raw_text = response.read().decode('utf-8')
text = raw_text[:10000]

vocab = list(set(text))
vocab.sort()
vocab_size = len(vocab)

def char_to_one_hot(char):
    vec = [0.0] * vocab_size
    vec[vocab.index(char)] = 1.0
    return vec

def char_to_index(char):
    return [float(vocab.index(char))]

seq_length = 8
data = []
for i in range(len(text) - seq_length):
    seq_chars = text[i:i+seq_length]
    target_char = text[i+seq_length]

    x_seq = [char_to_index(c) for c in seq_chars]
    y_target = char_to_one_hot(target_char)
    data.append((x_seq, y_target))

embedding_dim = 64 
head_dim = 64
rank_dim = 16

token_embed = nn.Embedding(vocab_size, embedding_dim)
pos_embed = nn.Embedding(seq_length, embedding_dim)
attention = nn.SelfAttention(embedding_dim, head_dim, rank_dim)

ffn_lin1 = nn.Linear(head_dim * seq_length, 128)
ffn_relu = nn.ReLU()
ffn_lin2 = nn.Linear(128, head_dim * seq_length)

final_lin = nn.Linear(head_dim * seq_length, vocab_size)

pos_seq = [float(i) for i in range(seq_length)]

epochs = 5
learning_rate = 0.01

print("train")
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    random.shuffle(data)
    
    for x_seq, y_target in data:
        flat_x_seq = [x[0] for x in x_seq]
        
        t_emb = token_embed.forward(flat_x_seq)
        p_emb = pos_embed.forward(pos_seq)
        x_emb = [t + p for t, p in zip(t_emb, p_emb)]
        
        attn_out = attention.forward(x_emb)
        
        res_out = [a + x for a, x in zip(attn_out, x_emb)]
        
        f1_out = ffn_lin1.forward(res_out)
        f_act = ffn_relu.forward(f1_out)
        f2_out = ffn_lin2.forward(f_act)
        
        logits = final_lin.forward(f2_out)
        
        epoch_loss += nn.Loss.cross_entropy_loss(logits, y_target)
        
        grad = nn.Loss.cross_entropy_gradient(logits, y_target)
        grad = final_lin.backward(grad, learning_rate)
        
        grad = ffn_lin2.backward(grad, learning_rate)
        grad = ffn_relu.backward(grad, learning_rate)
        grad = ffn_lin1.backward(grad, learning_rate)
        
        grad_attn = attention.backward(grad, learning_rate)
        grad_x_emb = [g + ga for g, ga in zip(grad, grad_attn)]
        
        pos_embed.backward(grad_x_emb, learning_rate)
        token_embed.backward(grad_x_emb, learning_rate)
        
    print(f"Epoch: {epoch} | Avg Loss: {epoch_loss / len(data):.4f}")

print("\ntesting")
test_seqs = ["First ci", "O Romeo", "What is "]

for seq in test_seqs:
    input_seq = seq[:seq_length] 
    flat_x_seq = [char_to_index(char)[0] for char in input_seq]
    
    t_emb = token_embed.forward(flat_x_seq)
    p_emb = pos_embed.forward(pos_seq)
    x_emb = [t + p for t, p in zip(t_emb, p_emb)]
    
    attn_out = attention.forward(x_emb)
    res_out = [a + x for a, x in zip(attn_out, x_emb)]
    
    f1_out = ffn_lin1.forward(res_out)
    f_act = ffn_relu.forward(f1_out)
    f2_out = ffn_lin2.forward(f_act)
    
    logits = final_lin.forward(f2_out)
        
    probs = nn.Utils.softmax(logits)
    predicted_idx = probs.index(max(probs))
    predicted_char = vocab[predicted_idx]
    
    print(f"Input: '{input_seq}' -> Predicted Next: '{predicted_char}'")

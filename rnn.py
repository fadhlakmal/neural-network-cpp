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

embedding_dim = 16 
hidden_size = 64
model = nn.Model()

model.add(nn.Embedding(vocab_size, embedding_dim))
rnn1 = nn.RNN(embedding_dim, hidden_size)
model.add(rnn1)
model.add(nn.Linear(hidden_size, vocab_size))

epochs = 50
learning_rate = 0.01

print("train")
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    random.shuffle(data)
    
    for x_seq, y_target in data:
        for t in range(seq_length):
            logits = model.forward(x_seq[t])
            
        epoch_loss += nn.Loss.cross_entropy_loss(logits, y_target)
        
        gradient = nn.Loss.cross_entropy_gradient(logits, y_target)
        model.backward(gradient, learning_rate)
        
        rnn1.reset_states()
        
    print(f"Epoch: {epoch} | Avg Loss: {epoch_loss / len(data):.4f}")

print("\ntesting")
test_seqs = ["First ci", "O Romeo", "What is "]

for seq in test_seqs:
    input_seq = seq[:seq_length] 
    for char in input_seq:
        logits = model.forward(char_to_index(char))
        
    probs = nn.Utils.softmax(logits)
    predicted_idx = probs.index(max(probs))
    predicted_char = vocab[predicted_idx]
    
    print(f"Input: '{input_seq}' -> Predicted Next: '{predicted_char}'")
    rnn1.reset_states()
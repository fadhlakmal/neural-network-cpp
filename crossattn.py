import fent as nn
import random
import math

vocab_size = 50
query_seq_length = 8
query_dim = 64

context_seq_length = 49
context_dim = 256

head_dim = 64
rank_dim = 16

epochs = 10
learning_rate = 0.01
dataset_size = 100

data = []
for _ in range(dataset_size):
    x_query_idx = [float(random.randint(0, vocab_size - 1)) for _ in range(query_seq_length)]
    
    x_context_feat = [random.uniform(-0.1, 0.1) for _ in range(context_seq_length * context_dim)]
    
    target_idx = random.randint(0, vocab_size - 1)
    y_target = [0.0] * vocab_size
    y_target[target_idx] = 1.0
    
    data.append((x_query_idx, x_context_feat, y_target))

token_embed = nn.Embedding(vocab_size, query_dim)
cross_attn = nn.CrossAttention(query_dim, 
                               context_dim, 
                               head_dim, 
                               rank_dim)
final_lin = nn.Linear(head_dim * query_seq_length, vocab_size)

print("train")

for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    random.shuffle(data)
    
    for x_query_idx, x_context_feat, y_target in data:
        q_emb = token_embed.forward(x_query_idx)
        
        cross_attn.set_context(x_context_feat, context_seq_length)
        
        attn_out = cross_attn.forward(q_emb)
        
        logits = final_lin.forward(attn_out)
        
        loss = nn.Loss.cross_entropy_loss(logits, y_target)
        epoch_loss += loss

        grad = nn.Loss.cross_entropy_gradient(logits, y_target)
        grad = final_lin.backward(grad, learning_rate)
        grad_query = cross_attn.backward(grad, learning_rate)
        
        token_embed.backward(grad_query, learning_rate)
        
    avg_loss = epoch_loss / dataset_size
    print(f"Epoch: {epoch} | Avg Loss: {avg_loss:.4f}")

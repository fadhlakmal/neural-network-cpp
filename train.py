import fent as nn
import numpy as np
import random
from sklearn.datasets import fetch_openml

print("fetch data")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

X = X / 255.0
y = y.astype(int)

X_sample = []
y_sample = []
for digit in range(10):
    indices = np.where(y == digit)[0]
    chosen_indices = np.random.choice(indices, 100, replace=False)
    
    for idx in chosen_indices:
        X_sample.append(X[idx].tolist())
        one_hot = [0.0] * 10
        one_hot[digit] = 1.0
        y_sample.append(one_hot)

data = list(zip(X_sample, y_sample))
random.shuffle(data)

train_data = data[:800]
test_data = data[800:]

model = nn.Model()
model.add(nn.Conv2D(1, 28, 28, 8, 3)) 
model.add(nn.ReLU())
model.add(nn.Linear(5408, 10))

epochs = 15
learning_rate = 0.05

print("training")
for epoch in range(1, epochs + 1):
    epoch_loss = 0.0
    random.shuffle(train_data) 
    
    for x_target, y_target in train_data:
        logits = model.forward(x_target)
        epoch_loss += nn.Loss.cross_entropy_loss(logits, y_target)
        
        gradient = nn.Loss.cross_entropy_gradient(logits, y_target)
        model.backward(gradient, learning_rate)
        
    print(f"epoch: {epoch} | avg loss: {epoch_loss / len(train_data):.6f}")

print("\ntesting")
correct = 0
for x_target, y_target in test_data:
    logits = model.forward(x_target)
    probs = nn.Utils.softmax(logits)
    
    predicted_class = probs.index(max(probs))
    actual_class = y_target.index(1.0)
    
    if predicted_class == actual_class:
        correct += 1

print(f"acc: {correct / len(test_data) * 100}%")
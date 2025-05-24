import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)

input_size = 2
hidden_size = 2
output_size = 1
lr = 0.1

W1 = 2 * np.random.rand(input_size, hidden_size) - 1
b1 = np.zeros((1, hidden_size))
W2 = 2 * np.random.rand(hidden_size, output_size) - 1
b2 = np.zeros((1, output_size))

loss_history = []

for epoch in range(10000):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    loss = np.mean((y - a2)**2)
    loss_history.append(loss)

    d_a2 = (a2 - y) * sigmoid_derivative(a2)
    dW2 = np.dot(a1.T, d_a2)
    db2 = np.sum(d_a2, axis=0, keepdims=True)

    d_a1 = np.dot(d_a2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, d_a1)
    db1 = np.sum(d_a1, axis=0)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch} Loss: {loss:.4f}")

print("\nFinal Predictions:")
print(np.round(a2, 2))

# Plot
plt.plot(loss_history)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_curve.png")
plt.show()

# Tiny Neural Network from Scratch 

This project demonstrates a neural network that solves the XOR problem using just Python and NumPy — without any ML libraries.

## 🚀 What It Does
- Builds a 2-layer neural network
- Solves XOR (non-linear problem)
- Uses sigmoid activation, forward & backpropagation, and gradient descent

## 🧠 Key Concepts
- Feedforward Neural Networks
- Sigmoid activation
- Mean Squared Error Loss
- Backpropagation
- Gradient Descent

## 🛠️ Requirements
```bash
python3
pip install numpy matplotlib

## 📌 Working Procedure

1. The model receives **4 binary input pairs** and expected XOR outputs:
[0, 0] → 0
[0, 1] → 1
[1, 0] → 1
[1, 1] → 0

2. The network:
- Initializes random weights and biases
- Propagates inputs forward using the **sigmoid activation**
- Calculates loss using **Mean Squared Error (MSE)**
- Performs **backpropagation** to adjust weights
- Repeats for 10,000 epochs

3. During training, **loss is reduced** step by step.

## 📊 Output Graph: Loss Curve

The graph below shows how the **loss decreases steadily** over training, confirming that the model is learning effectively:

![Loss Curve](loss_curve.png)

- At the beginning, the loss is ~0.27
- By the end of 10,000 epochs, it falls below 0.01
- This confirms **successful convergence** for a non-linearly separable problem like XOR

---

## ✅ Final Predictions

Input: [0,0] → Output: ~0.05
Input: [0,1] → Output: ~0.94
Input: [1,0] → Output: ~0.94
Input: [1,1] → Output: ~0.06


📌 These predictions are close to the expected XOR outputs — showing the network **learned correctly**.


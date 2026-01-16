# MNIST Digit Classifier from Scratch (NumPy)

A handwritten digit classifier implemented entirely from scratch using NumPy, without using deep learning frameworks such as TensorFlow, PyTorch, or Keras.
This project focuses on understanding neural networks at a fundamental level by explicitly implementing forward propagation, backpropagation, gradient descent, and controlled training experiments.

## Overview

- Dataset: MNIST (28×28 grayscale images)
- Task: Multiclass classification (digits 0–9)
- Implementation: Pure NumPy
- Model: Fully connected neural network
- Loss: Cross-entropy
- Activations: ReLU (hidden layer), Softmax (output layer)
- Optimization: Gradient Descent

## Model Architechture

```markdown
784 (Input)
  ↓
Hidden Layer (ReLU)
  ↓
10 (Softmax Output)
```
Input images are flattened into 784-dimensional vectors and mapped to digit probabilities via softmax.

## Experiments

Experiments are structured so that only one factor is changed at a time, with training loss and accuracy tracked for comparison.

### Baseline: Random Initialization

  - Small random weights

  - Zero biases

    #### Observation:

  - Slower convergence

  - Higher variance in early training

  - Lower accuracy in initial iterations
  
### He Initialization (ReLU-aware)

    #### Observation:

  - Faster convergence

  - Smoother loss curve

  - Higher accuracy early in training

  - Improved gradient flow through ReLU layers
  

Result: He initialization consistently outperformed naive random initialization.

## Metrics Tracked

#### Training loss vs iterations
#### Training accuracy vs iterations

These metrics are plotted automatically during training and used to compare experiments.

## Label Encoding

Class labels are converted to one-hot encoded vectors before computing cross-entropy loss.

Example:
```markdown
Digit: 3
One-hot: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```
This representation enables correct loss computation and simplifies gradient calculations.

### Key Learnings

- Implemented forward and backward propagation from first principles
- Understood why softmax combined with cross-entropy simplifies gradients
- Observed the effect of weight initialization on training stability
- Built intuition for ReLU behavior and gradient flow

### Current Limitations

- No regularization techniques (L2, dropout not implemented yet)
- Fully connected network only
- No validation or test set evaluation seperately, depends on user input

### Planned Improvements

- Add regularization methods
- Implement convolutional neural networks from scratch
- Experiment with deeper architectures
- Compare optimization strategies




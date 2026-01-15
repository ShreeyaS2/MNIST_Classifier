import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt


def load_and_train():
    print("Loading data and training model... Please wait.")
    
    # Load data (Adjust path if necessary)
    data = pd.read_csv('C:\\Personal\\comp_proj\\neuralnetwork\\mnist_train.csv')
    data = np.array(data)
    m, n = data.shape
    np.random.shuffle(data)

    # Train/Dev split
    data_test = data[0:1000].T
    Y_test = data_test[0]
    X_test = data_test[1:n] / 255.0

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n] / 255.0

    # Initialize (Optimized for ReLU)
    def initialize_params():
        W1 = np.random.rand(128, 784) - 0.5
        b1 = np.zeros((128, 1))
        W2 = np.random.rand(10, 128) - 0.5
        b2 = np.zeros((10, 1))
        return W1, b1, W2, b2

    def ReLU(Z):
        return np.maximum(Z, 0)

    def softmax(Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / sum(expZ)
    
    def relu_derivative(Z):
        return Z > 0

    def one_hot(Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def forward_prop(W1, b1, W2, b2, X):
        Z1 = np.dot(W1, X) + b1
        A1 = ReLU(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_prop(Z1, A1, A2, W2, X, Y):
        m = Y.size
        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * np.dot(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * relu_derivative(Z1)
        dW1 = 1 / m * np.dot(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2

    def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
        W1 = W1 - alpha * dW1
        b1 = b1 - alpha * db1
        W2 = W2 - alpha * dW2
        b2 = b2 - alpha * db2
        return W1, b1, W2, b2

    def get_predictions(A2):
        return np.argmax(A2, 0)

    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y) / Y.size

    # Training Loop - gradient descent 
    W1, b1, W2, b2 = initialize_params()
    alpha = 0.05
    iterations = 1000 # Keep it short for demo, increase for better accuracy
    
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_train)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, A2, W2, X_train, Y_train)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 100 == 0:
            print(f"Iteration: {i}, Error: {get_accuracy(get_predictions(A2), Y_train):.4f}")

    print("Training Complete!")
    return W1, b1, W2, b2, forward_prop


class DigitApp:
    def __init__(self, root, W1, b1, W2, b2, forward_prop_func):
        self.root = root
        self.root.title("Draw a Digit")
        
        # Neural Net Parameters
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.forward_prop = forward_prop_func

        # Canvas Setup (We draw bigger than 28x28 for user experience)
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()

        # PIL Image (This happens in memory, same size as canvas)
        # We draw white on black to match MNIST format
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image1)

        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        self.btn_predict = tk.Button(text="Predict", command=self.predict_digit)
        self.btn_predict.pack()
        
        self.btn_clear = tk.Button(text="Clear", command=self.clear_canvas)
        self.btn_clear.pack()

        self.label_result = tk.Label(text="Prediction: None", font=("Helvetica", 24))
        self.label_result.pack()

    def paint(self, event):
        # Brush parameters
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        
        # Draw on UI Canvas
        self.canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
        
        # Draw on Memory Image (PIL)
        self.draw.ellipse([x1, y1, x2, y2], fill=255, outline=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image1)
        self.label_result.config(text="Prediction: None")

    def predict_digit(self):
        # 1. Resize image to 28x28
        img = self.image1.resize((28, 28))
        
        # 2. Convert to numpy array
        img_array = np.array(img)
        
        # 3. Flatten and Normalize (just like training data)
        # Reshape to (784, 1) and divide by 255
        img_vector = img_array.reshape(784, 1) / 255.0
        
        # 4. Forward Prop
        _, _, _, A2 = self.forward_prop(self.W1, self.b1, self.W2, self.b2, img_vector)
        prediction = np.argmax(A2, 0)
        
        # 5. Show Result
        self.label_result.config(text=f"Prediction: {prediction[0]}")
        

if __name__ == "__main__":
    # 1. Train the model
    trained_W1, trained_b1, trained_W2, trained_b2, fp_func = load_and_train()

    # 2. Start the App
    root = tk.Tk()
    app = DigitApp(root, trained_W1, trained_b1, trained_W2, trained_b2, fp_func)

    root.mainloop()

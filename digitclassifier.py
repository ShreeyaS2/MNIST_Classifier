""" import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.special import softmax      

data=pd.read_csv('C:\\Personal\\comp_proj\\neuralnetwork\\mnist_train.csv')
m,n=data.shape
data=np.array(data) 
test_data=data[0:500].T #splitting into testing (0-1000) and training (1000-m)
Y_dev=test_data[0] #after transposing, seperating the label row
X_dev=test_data[1:n]/255.0 # splitting till n for test data and normalizing by dividing by 255
train_data=data[500:m].T # taking the training data from 1000 to m and transposing
Y_train=train_data[0] #seperating the label row
X_train=train_data[1:n]/255.0 # splitting till n for train data and normalizing by dividing by 255

#initializing parameters
def initialize_params():
    W1=np.random.rand(10,784) -0.5
    b1=np.zeros((10,1))
    W2=np.random.rand(10,10) -0.5
    b2=np.zeros((10,1))
    return W1,b1,W2,b2

# stepone: forward propagation
def forward_prop(W1,b1,W2,b2,X):
    Z1=np.dot(W1,X) +b1
    #replace sigmoid with relu
    A1=np.maximum(Z1,0) #relu activation function
    Z2=np.dot(W2,A1)+b2
    A2=softmax(Z2,axis=0) #softmax activation function
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def relu_derivative(Z):
    return Z > 0

# steptwo: backward propagation
def backward_prop(A1,A2,W2,X,Y,Z1):
    m=Y.size
    one_hot_Y=one_hot(Y)
    dZ2=A2-one_hot_Y
    dW2=1/m*np.dot(dZ2,A1.T)
    db2=1/m*np.sum(dZ2,axis=1,keepdims=True)
    dZ1=np.dot(W2.T,dZ2)*relu_derivative(Z1)
    dW1=1/m*np.dot(dZ1,X.T)
    db1=1/m*np.sum(dZ1,axis=1,keepdims=True)
    return dW1,db1,dW2,db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,learning_rate):
    W1=W1-learning_rate*dW1
    b1=b1-learning_rate*db1
    W2=W2-learning_rate*dW2
    b2=b2-learning_rate*db2
    return W1,b1,W2,b2
    
def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = initialize_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(A1, A2, W2, X, Y, Z1)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.07, 1000)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
test_prediction(17, W1, b1, W2, b2) """

import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

# ==========================================
# PART 1: YOUR NEURAL NETWORK LOGIC
# ==========================================

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

# ==========================================
# PART 2: THE DRAWING GUI
# ==========================================

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
        
        # Optional: Debug - show what the network sees
        # plt.imshow(img_array, cmap='gray')
        # plt.show()

# ==========================================
# PART 3: MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # 1. Train the model
    trained_W1, trained_b1, trained_W2, trained_b2, fp_func = load_and_train()

    # 2. Start the App
    root = tk.Tk()
    app = DigitApp(root, trained_W1, trained_b1, trained_W2, trained_b2, fp_func)
    root.mainloop()
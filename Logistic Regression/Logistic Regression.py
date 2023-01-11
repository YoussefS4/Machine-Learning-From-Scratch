import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from prettytable import PrettyTable


def sigmoid(z):
    
    g = 1 / (1 + np.exp(-z))
    
    return g

def compute_cost(X, y, w, b):
# X : Data, m examples with n features
# y : target values
# w : model parameters
# b : model parameter
    m, n = X.shape
    loss_sum = 0
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_w_ij = w[j] * X[i][j]
            z_wb += z_w_ij
        z_wb += b
        
        f_wb = sigmoid(z_wb)
        loss = -y[i] * np.log(f_wb) - (1 - y[i]) * np.log(1 - f_wb)
        loss_sum += loss
    total_cost = (1 / m) * loss_sum
    
    return total_cost

def compute_gradient(X, y, w, b):
# X : Data, m examples with n features
# y : target values
# w : model parameters
# b : model parameter
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0.    

    for i in range(m):
        z_wb = 0
        for j in range(n): 
            z_wb += w[j] * X[i][j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw_ij = (f_wb - y[i])* X[i][j]
            dj_dw[j] += dj_dw_ij
            
    dj_dw /= m
    dj_db /= m

        
    return dj_db, dj_dw

def gradient_descent(X, y, w, b, cost_function, gradient_function, alpha, num_iters, out):
# X : Data, m examples with n features
# y : target values
# w : initial model parameters
# b : initial model parameter
# cost_function : function to compute cost
# gradient_function : function to compute the gradient
# alpha : Learning rate
# num_iters : number of iterations to run gradient descent
# An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db              
       
        # Save cost J at each iteration
        cost =  cost_function(X, y, w, b)
        J_history.append(cost)
        
        if out and i % 1000 == 0 :
            print(f"Iteration {i:4}: Cost {J_history[-1]}")
        
    return w, b, J_history

def plot_result(flag):
    # initialize parameters
    initial_w = np.zeros(x_normalized.shape[1])
    initial_b = 0
    # gradient descent settings
    iterations = 5000

    if flag:
        alpha = [3,0.9, 0.1,0.001]
        # run gradient descent
        plt.title("Cost vs. iteration")
        plt.ylabel('Cost')
        plt.xlabel('iteration step')
        for i in range(len(alpha)):
            _, _, J_hist = gradient_descent(X_train, Y_train, initial_w, initial_b,
                                            compute_cost, compute_gradient,
                                            alpha[i], iterations, out=False)
            # plot cost versus iteration
            plt.plot(J_hist)
            plt.legend([3,0.9, 0.1,0.001])
            plt.show()

    else:
        alpha = 0.9
        # run gradient descent
        w_final, b_final, J_hist = gradient_descent(X_train, Y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations, out=True)

        print(f"w,b found by gradient descent: {b_final},{w_final} ")
    return w_final,b_final

def predict_from_scratch(X, w, b):
    
    m, n = X.shape   
    p = np.zeros(m)
   
    for i in range(m):   
        z_wb = 0
        for j in range(n): 
            z_wb += w[j] * X[i][j]
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        p[i] = f_wb >= 0.5

    return p

df = pd.read_csv('customer_data.csv')

print(df.head())

df = shuffle(df)

x = df.drop('purchased', axis = 1)
y = df['purchased']

# Min-Max Normalizer
x_normalized = (x - x.min()) / (x.max() - x.min())

#Convert data to numpy array
x_normalized = x_normalized.to_numpy()
y = y.to_numpy()

# Splitting Data
X_train, X_test, Y_train, Y_test = train_test_split(x_normalized, y, test_size = 0.2, random_state = 42)

# Final results for model parameters
w_final, b_final = plot_result(False)

# Compute accuracy on our training and testing set
p_train = predict_from_scratch(X_train, w_final, b_final)
p_test = predict_from_scratch(X_test, w_final, b_final)

p_train_acc = np.mean(p_train == Y_train) * 100
p_test_acc = np.mean(p_test == Y_test) * 100

lr = LogisticRegression()
lr.fit(X_train,Y_train)

lr_train = lr.predict(X_train)
lr_acc_train = accuracy_score(Y_train, lr_train)*100


lr_test = lr.predict(X_test)
lr_acc_test = accuracy_score(Y_test, lr_test)*100


lr_perc_scratch = precision_score(Y_test, p_test)
lr_rec_scratch = recall_score(Y_test, p_test)
lr_f1_scratch = f1_score(Y_test, p_test)

lr_perc_sklearn = precision_score(Y_test, lr_test)
lr_rec_sklearn = recall_score(Y_test, lr_test)
lr_f1_sklearn = f1_score(Y_test, lr_test)

x = PrettyTable(['Evaluation Metrics','Logistic Regression from Scratch', 'Logistic Regression from Sklearn'])

x.add_row(['Train Accurracy', p_train_acc, lr_acc_train])
x.add_row(['Test Accurracy', p_test_acc, lr_acc_test])
x.add_row(['Percision', lr_perc_scratch, lr_perc_sklearn])
x.add_row(['Recall', lr_rec_scratch, lr_rec_sklearn])
x.add_row(['F-measure', lr_f1_scratch, lr_f1_sklearn])
print(x)


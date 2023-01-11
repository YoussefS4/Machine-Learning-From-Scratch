import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error, r2_score

def compute_cost(X, y, w, b):
# X : Data, m examples with n features
# y : target values
# w : model parameters
# b : model parameter
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b         
        cost += (f_wb_i - y[i])**2       
    cost = cost / (2 * m)                      
    return cost

def compute_gradient(X, y, w, b):
# X : Data, m examples with n features
# y : target values
# w : model parameters
# b : model parameter
    m,n = X.shape          
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
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
# out (bool): if True: print cost for every iteration else nothing

    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   

        # Update Parameters
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db               
      
        # Save cost J at each iteration
        J_history.append(cost_function(X, y, w, b))

        if out:
            if i % 1000 == 0:
                print(f"Iteration {i:4}: Cost {J_history[i]}")
        
    return w, b, J_history

def plot_result(flag):
    # initialize parameters
    initial_w = np.zeros(x_normalized.shape[1])
    initial_b = 0
    # gradient descent settings
    iterations = 5000

    if flag:
        alpha = [0.9, 0.1, 0.001, 0.003]
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
            plt.legend([0.9, 0.1, 0.001, 0.003])
            plt.show()

    else:
        alpha = 0.9
        # run gradient descent
        w_final, b_final, J_hist = gradient_descent(X_train, Y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient,
                                                    alpha, iterations, out=True)

        print(f"w,b found by gradient descent: {b_final},{w_final} ")
    return w_final,b_final


# Main
df = pd.read_csv('car_data.csv')

print(df.head())

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),annot = True)

# 4 features that positively correlated to the price
df = df.filter(['carwidth','curbweight', 'enginesize','horsepower', 'price'])

df = shuffle(df)

x = df.drop('price', axis = 1)
y = df['price']

# Min-Max Normalizer
x_normalized = (x - x.min()) / (x.max() - x.min())

# Convert data to numpy array
x_normalized = x_normalized.to_numpy()
y = y.to_numpy()

# Splitting Data
X_train, X_test, Y_train, Y_test = train_test_split(x_normalized, y, test_size = 0.2)

# Final results for model parameters
w_final, b_final = plot_result(False)


#Compute accuracy on our training and testing set
y_pred_from_scratch = []
m = X_test.shape[0]
for i in range(m):
    x = np.dot(X_test[i], w_final) + b_final
    y_pred_from_scratch.append(x)


lr = LinearRegression()
lr.fit(X_train, Y_train)

y_pred_from_sklearn = lr.predict(X_test)

mse_lr_scratch = mean_squared_error(Y_test, y_pred_from_scratch)
mse_lr_sklearn= mean_squared_error(Y_test, y_pred_from_sklearn)

r2_scratch = r2_score(Y_test, y_pred_from_scratch)
r2_sklearn = r2_score(Y_test, y_pred_from_sklearn)


x = PrettyTable(['Models', 'MSE','R2'])

x.add_row(['Linear Regression from Scratch', mse_lr_scratch, r2_scratch])
x.add_row(['Linear Regression from Sklearn', mse_lr_sklearn,r2_sklearn])
print(x)


import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.linalg import solve_discrete_are
from tqdm import tqdm

# calculate the running time of this program
import time
start_time = time.time()

A = np.array([[1]])
B = np.array([[1]])
Q = np.eye(1)
R = np.eye(1)

# solve the discrete-time Riccati equation
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
H = B @ np.linalg.inv(R + B.T @ P @ B) @ B.T

a = A[0, 0]
p = P[0, 0]
k = K[0, 0]
h = H[0, 0]

N = 200000
T = 100

N_train = int(N * 0.8)
N_test = N - N_train

# construct the noise dataset

W = np.random.normal(0, 1, (N, T, 3))/3
noises = np.sum(W, axis=2)

# function for computing the MSE/R^2 value of Predictor A at time step t

def compute_mse_r2_A(t):
    pred_A = noises[:, :t]
    pred_A = np.concatenate([W[:,:t+2, 0], pred_A], axis=1)
    pred_A = np.concatenate([W[:,:t+1, 1], pred_A], axis=1)

    pred_A_train = pred_A[:N_train]
    pred_A_test = pred_A[N_train:]
    target_train = noises[:N_train, t]
    target_test = noises[N_train:, t]

    # use sklearn linear regression model to predict noise at time step t
    model = LinearRegression()
    model.fit(pred_A_train, target_train)

    # compute the MSE value
    pred_noise = model.predict(pred_A_test)
    mse = mean_squared_error(target_test, pred_noise)

    # use sklearn linear regression model to predict noise at time step t+1
    target_train = noises[:N_train, t+1]
    target_test = noises[N_train:, t+1]

    model = LinearRegression()
    model.fit(pred_A_train, target_train)

    # compute the MSE value
    pred_noise = model.predict(pred_A_test)
    mse2 = mean_squared_error(target_test, pred_noise)

    return mse, mse2

fitting_results_A = [compute_mse_r2_A(t) for t in tqdm(range(1, T-1))]
fitting_results_A = np.array(fitting_results_A)

# function for computing the MSE/R^2 value of Predictor B at time step t

opt_action_res = np.zeros((N, T-1))
for t in range(T-1):
    opt_action_res[:, t] = p * (W[:, t, 0] + W[:, t, 1]) + (a - a * p * h) * p * W[:, t+1, 0]

def compute_mse_r2_B(t):
    pred_B = noises[:, :t]
    pred_B = np.concatenate([opt_action_res[:,:t+1], pred_B], axis=1)

    pred_B_train = pred_B[:N_train]
    pred_B_test = pred_B[N_train:]

    # use sklearn linear regression model to predict noise at time step t
    target_train = noises[:N_train, t]
    target_test = noises[N_train:, t]

    model = LinearRegression()
    model.fit(pred_B_train, target_train)

    # compute the MSE value
    pred_noise = model.predict(pred_B_test)
    mse = mean_squared_error(target_test, pred_noise)

    # use sklearn linear regression model to predict noise at time step t+1
    target_train = noises[:N_train, t+1]
    target_test = noises[N_train:, t+1]

    model = LinearRegression()
    model.fit(pred_B_train, target_train)

    # compute the MSE value
    pred_noise = model.predict(pred_B_test)
    mse2 = mean_squared_error(target_test, pred_noise)

    return mse, mse2

fitting_results_B = [compute_mse_r2_B(t) for t in tqdm(range(1, T-1))]
fitting_results_B = np.array(fitting_results_B)

# plot the mse results of Predictor A and Predictor B

import matplotlib.pyplot as plt

plt.figure()
plt.plot(fitting_results_A[:, 0], label='Predictor 1 0-step forward')
plt.plot(fitting_results_B[:, 0], label='Predictor 2 0-step forward')
plt.plot(fitting_results_A[:, 1], label='Predictor 1 1-step forward')
plt.plot(fitting_results_B[:, 1], label='Predictor 2 1-step forward')
plt.xlabel('Time step')
plt.ylabel('MSE')
plt.legend(prop={'family': 'Times New Roman'})
plt.savefig('Figures/multi_step_mse_predictor_A_B.pdf')

end_time = time.time()
# get the processor information
import platform

processor = platform.processor()

print("Running time is {} seconds on {}.".format(end_time - start_time, processor))
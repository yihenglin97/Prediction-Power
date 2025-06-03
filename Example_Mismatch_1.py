import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.linalg import solve_discrete_are
from tqdm import tqdm

def noise_pred_generator_corr_Gauss(N, T, n, rho, sigma, transform_mat=None):
    # Generate the noise
    noise = np.random.normal(0, sigma, (N, T, n))
    # Generate the predictions available at each time step
    pred = rho * noise + np.sqrt(1 - rho**2) * np.random.normal(0, sigma, (N, T, n))
    if transform_mat is not None:
        pred = rho * (noise @ transform_mat.T) + np.random.normal(0, sigma, (N, T, n)) @ np.linalg.cholesky(np.eye(n) - rho**2 * transform_mat @ transform_mat.T).T
    return noise, pred


# calculate the running time of this program
import time
start_time = time.time()

N = 80000
T = 100
n = 2
sigma = 1
N_train = int(N * 0.8)
N_test = N - N_train

k = 2

# define a dynamical system with double integrator dynamics
dt = 0.1
A = np.array([[1, dt], [0, 1]])
B = np.array([[0], [dt]])
Q = np.eye(n)
R = np.eye(1)

# solve the discrete-time Riccati equation
P = solve_discrete_are(A, B, Q, R)
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

target_cov = np.array([[1.0, 0.99], [0.99, 1.0]])
weights = np.linalg.cholesky(target_cov)
transform_mat = weights.T

noise, pred = noise_pred_generator_corr_Gauss(N, T, n, 0.5, sigma)
noise = noise.reshape((-1, n))
pred = pred.reshape((-1, n))

noise = noise @ weights.T

mat_H = B @ np.linalg.inv(R + B.T @ P @ B) @ B.T
pred_power = np.trace(P @ mat_H @ P)
pred_power_tranformed = np.trace(target_cov @ P @ mat_H @ P)

def feedback_optimal_control(x0, v, P, H):
    u = - K @ x0
    M1 = np.linalg.inv(R + B.T @ P @ B) @ B.T
    for t in range(H):
        u -= M1 @ P @ v[:, t, :]
        M1 = M1 @ (A - B @ K).T
    return u

def test_one_step_optimal_control_corr(rho, transform_mat, num_traj):
    np.random.seed(0)
    noise, pred = noise_pred_generator_corr_Gauss(N, T, n, rho, sigma, transform_mat)
    noise_train = noise[:N_train]
    noise_test = noise[N_train:]
    pred_train = pred[:N_train]
    pred_test = pred[N_train:]

    model = LinearRegression()
    model.fit(pred_train.reshape((-1, n)), noise_train.reshape((-1, n)))
    reg_pred = model.predict(pred_train.reshape((-1, n)).reshape((-1, n)))
    metric_record = []
    
    for i in range(n):
        mse = mean_squared_error(noise_train.reshape((-1, n))[:, i], reg_pred[:, i])
        r2 = r2_score(noise_train.reshape((-1, n))[:, i], reg_pred[:, i])
        metric_record.append([mse, r2])
    
    # simulate the optimal controller
    x0 = np.zeros((n, num_traj))
    total_cost_sum = 0.0
    traj = np.zeros((n, T, num_traj))
    for t in range(T):
        v = np.zeros((n, T-t, num_traj))
        v[:, 0, :] = model.predict(pred_test[:num_traj, t, :]).T
        ut = feedback_optimal_control(x0, v, P, T-t)
        total_cost_sum += (np.trace(Q @ x0 @ x0.T) + np.trace(R @ ut @ ut.T))
        x0 = A @ x0 + B @ ut + noise_test[:num_traj, t, :].T
        traj[:, t, :] = x0
    total_cost_sum += np.trace(P @ x0 @ x0.T)

    theoretical_total_cost = np.trace(P - rho**2 * transform_mat.T @ transform_mat @ P @ mat_H @ P) * T

    return total_cost_sum/num_traj, theoretical_total_cost, metric_record

# plot the regression metrics and control costs under different correlation coefficients
rho_list = np.linspace(0, 0.7, 8)
num_traj = N_test
cost_list = []
theoretical_cost_list = []
metric_list = []
for rho in tqdm(rho_list):
    cost, theoretical_cost, metric = test_one_step_optimal_control_corr(rho, np.eye(2), num_traj)
    cost_list.append(cost)
    theoretical_cost_list.append(theoretical_cost)
    metric_list.append(metric)

cost_array = np.array(cost_list)
theoretical_cost_array = np.array(theoretical_cost_list)
metric_array = np.array(metric_list)

cost_list_transformed = []
theoretical_cost_list_transformed = []
metric_list_transformed = []
for rho in tqdm(rho_list):
    cost, theoretical_cost, metric = test_one_step_optimal_control_corr(rho, transform_mat, num_traj)
    cost_list_transformed.append(cost)
    theoretical_cost_list_transformed.append(theoretical_cost)
    metric_list_transformed.append(metric)

cost_array_transformed = np.array(cost_list_transformed)
theoretical_cost_array_transformed = np.array(theoretical_cost_list_transformed)
metric_array_transformed = np.array(metric_list_transformed)

# plot the comparison of mse
import matplotlib.pyplot as plt

plt.figure()
plt.plot(rho_list, metric_array[:, 0, 0], label=r'MSE of dimension 0 ($I$)')
plt.plot(rho_list, metric_array[:, 1, 0], label=r'MSE of dimension 1 ($I$)')
plt.plot(rho_list, metric_array_transformed[:, 0, 0], label=r'MSE of dimension 0 ($\theta$)')
plt.plot(rho_list, metric_array_transformed[:, 1, 0], label=r'MSE of dimension 1 ($\theta$)')
plt.xlabel(r'Coefficient $\rho$')
plt.ylabel('MSE')
plt.legend(prop={'family': 'Times New Roman'})
plt.savefig('Figures/1_step_mse_comparison.pdf')

# plot the comparison of total cost
plt.figure()
plt.plot(rho_list, cost_array, label=r'Total cost ($I$)')
plt.plot(rho_list, cost_array_transformed, label=r'Total cost ($\theta$)')
plt.plot(rho_list, theoretical_cost_array, linestyle = ':', label=r'Theoretical total cost ($I$)')
plt.plot(rho_list, theoretical_cost_array_transformed, linestyle = ':', label=r'Theoretical total cost ($\theta$)')
plt.xlabel(r'Coefficient $\rho$')
plt.ylabel('Total cost')
plt.legend(prop={'family': 'Times New Roman'})
plt.savefig('Figures/1_step_total_cost_comparison.pdf')

end_time = time.time()
# get the processor information
import platform

processor = platform.processor()

print("Running time is {} seconds on {}.".format(end_time - start_time, processor))

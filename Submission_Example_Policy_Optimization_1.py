import numpy as np
from linear_dynamics import linearDynamics
from tqdm import trange
from policy_optimization import policyOptimization

# calculate the running time of this program
import time
start_time = time.time()

# define a dynamical system with double integrator dynamics
n = 2
m = 1
dt = 0.1
A = np.array([[1, dt], [0, 1]])
B = np.array([[0], [dt]])
Q = np.eye(n)
R = np.eye(m)
MAX_HORIZON = 80000
trial_num = 30

# construct the linear dynamics system
sys = linearDynamics(A, B, Q, R, max_horizon=MAX_HORIZON)

# generate noise
np.random.seed(0)
w = np.random.normal(0, 1.0, (n, sys.MAX_HORIZON+1, trial_num))

rho = 0.5

v = rho * w + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1.0, (n, sys.MAX_HORIZON+1, trial_num))

# we first experiment with the following case:
# at time step t, the prediction v[t] is about the current noise w[t]

avg_cost_array = np.zeros((trial_num, sys.MAX_HORIZON))
avg_cost_optimal_predictive_policy_array = np.zeros((trial_num, sys.MAX_HORIZON))
avg_cost_optimal_no_prediction_policy_array = np.zeros((trial_num, sys.MAX_HORIZON))

pol_opt = policyOptimization(A, B, Q, R, max_horizon=sys.MAX_HORIZON, lr = 1e-4)
H = B @ np.linalg.inv(R + B.T @ pol_opt.P @ B) @ B.T
theoretical_opt_cost = np.trace(pol_opt.P - rho**2 * pol_opt.P @ H @ pol_opt.P)
theoretical_no_pred_cost = np.trace(pol_opt.P)
print("Theoretical optimal cost (no prediction): ", theoretical_no_pred_cost)
print("Theoretical optimal cost (with prediction): ", theoretical_opt_cost)

# compute the theoretical optimal feedback matrix M
P = pol_opt.P
M_opt = - rho * np.linalg.inv(R + B.T @ P @ B) @ B.T @ P

for trial in range(trial_num):
    print("Trial: ", trial)
    sys.reset()
    sys.set_noise(w[:, :-1, trial], v[:, :-1, trial])

    # construct the policy optimization object
    pol_opt = policyOptimization(A, B, Q, R, max_horizon=sys.MAX_HORIZON, lr = 1e-4)

    # record the average cost and the learned M
    avg_cost = []

    # simulate the system for MAX_HORIZON steps
    for i in trange(sys.MAX_HORIZON):
        obs = sys.observe()
        action = pol_opt.act(obs)
        sys.step(action)
        avg_cost.append(sys.total_cost / (i + 1))
        # use a decay learning rate
        lr = 1e-3 / np.sqrt(1 + i / 100)
        pol_opt.policy_update(lr=lr)

    avg_cost_array[trial, :] = np.array(avg_cost)

    # simulate the system for MAX_HORIZON steps with the optimal feedback matrix M
    sys.reset()
    avg_cost_optimal_predictive_policy = []
    for i in trange(sys.MAX_HORIZON):
        obs = sys.observe()
        action = M_opt @ obs[1] - pol_opt.K @ obs[0]
        sys.step(action)
        avg_cost_optimal_predictive_policy.append(sys.total_cost / (i + 1))
    
    avg_cost_optimal_predictive_policy_array[trial, :] = np.array(avg_cost_optimal_predictive_policy)

    # simulate the system for MAX_HORIZON steps with the optimal no-prediction policy
    sys.reset()
    avg_cost_optimal_no_prediction_policy = []
    for i in trange(sys.MAX_HORIZON):
        obs = sys.observe()
        action = - pol_opt.K @ obs[0]
        sys.step(action)
        avg_cost_optimal_no_prediction_policy.append(sys.total_cost / (i + 1))
    
    avg_cost_optimal_no_prediction_policy_array[trial, :] = np.array(avg_cost_optimal_no_prediction_policy)


# plot the average cost
import matplotlib.pyplot as plt

# compute the 25th and 75th percentiles
avg_cost = np.mean(avg_cost_array, axis=0)
avg_cost_optimal_predictive_policy = np.mean(avg_cost_optimal_predictive_policy_array, axis=0)
avg_cost_optimal_no_prediction_policy = np.mean(avg_cost_optimal_no_prediction_policy_array, axis=0)
avg_cost_25 = np.percentile(avg_cost_array, 25, axis=0)
avg_cost_75 = np.percentile(avg_cost_array, 75, axis=0)
avg_cost_optimal_predictive_policy_25 = np.percentile(avg_cost_optimal_predictive_policy_array, 25, axis=0)
avg_cost_optimal_predictive_policy_75 = np.percentile(avg_cost_optimal_predictive_policy_array, 75, axis=0)

plt.rcParams['savefig.dpi'] = 300

MGAPS_improvement = - avg_cost_array + avg_cost_optimal_no_prediction_policy_array
optimal_predictive_policy_improvement = - avg_cost_optimal_predictive_policy_array + avg_cost_optimal_no_prediction_policy_array
# compute the 25th and 75th percentiles
MGAPS_improvement_25 = np.percentile(MGAPS_improvement, 25, axis=0)
MGAPS_improvement_75 = np.percentile(MGAPS_improvement, 75, axis=0)
optimal_predictive_policy_improvement_25 = np.percentile(optimal_predictive_policy_improvement, 25, axis=0)
optimal_predictive_policy_improvement_75 = np.percentile(optimal_predictive_policy_improvement, 75, axis=0)

# plot the theoretical prediction power and the improvement of MGAPS and optimal predictive policy
plt.figure()
#plt.rcParams["font.family"] = "Times New Roman"
rc_fonts = {
    "font.family": "serif",
    "font.size": 16,
    "text.usetex": True,
    'text.latex.preamble': r'\usepackage{times}\usepackage{amsfonts}',
}
plt.rcParams.update(rc_fonts)
plt.axhline(y=theoretical_no_pred_cost - theoretical_opt_cost, linestyle='--', color='r', label=r"Prediction power ($P(1)/T$)")
plt.plot(- avg_cost + avg_cost_optimal_no_prediction_policy, label=r"M-GAPS")
plt.fill_between(range(sys.MAX_HORIZON), MGAPS_improvement_25, MGAPS_improvement_75, alpha=0.2)
plt.plot(- avg_cost_optimal_predictive_policy + avg_cost_optimal_no_prediction_policy, label=r"Optimal predictive policy ($\pi^1$)")
plt.fill_between(range(sys.MAX_HORIZON), optimal_predictive_policy_improvement_25, optimal_predictive_policy_improvement_75, alpha=0.2)
plt.xlabel("Time step")
plt.ylabel(r"Average cost improvement against $\bar{\pi}$")
plt.ylim(-0.5, 1.5)
plt.legend(loc='lower right')
plt.savefig("Figures/average_cost_improvement_current_prediction_{}.pdf".format(rho), bbox_inches='tight')

end_time = time.time()
# get the processor information
import platform

processor = platform.processor()

print("Running time is {} seconds on {}.".format(end_time - start_time, processor))
import numpy as np
from linear_dynamics import linearDynamics
from scipy.linalg import solve_discrete_are
from copy import deepcopy
from tqdm import trange

class policyOptimization:
    def __init__(self, A, B, Q, R, max_horizon = 10000, lookback = 1, lr = 1e-4, M_init = None):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.n = A.shape[0]
        self.m = B.shape[1]
        self.lr = lr

        self.MAX_HORIZON = max_horizon
        
        # solve the discrete-time Riccati equation
        self.P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ self.P @ B) @ B.T @ self.P @ A

        # feedback on history observations
        self.lookback = lookback
        self.M = np.zeros((self.m, self.n * self.lookback))
        if M_init is not None:
            assert M_init.shape == (self.m, self.n * self.lookback)
            self.M = deepcopy(M_init)

        self.state_history = []
        self.action_history = []
        self.prediction_history = []
        self.current_step = 0

        # the internal state maintained by M-GAPS
        self.internal_state = np.zeros((self.n, self.m * self.n * self.lookback))
    
    # reset the state and action history
    def reset(self):
        self.state_history = []
        self.action_history = []
        self.prediction_history = []
        self.current_step = 0
        self.M = np.zeros((self.m, self.n * self.lookback))
        self.internal_state = np.zeros((self.n, self.m * self.n * self.lookback))
    

    # decide the action based on the observation
    def act(self, obs):
        x, v = obs
        self.state_history.append(x)
        self.prediction_history.append(v)

        if self.current_step < self.lookback-1:
            u = - self.K @ x
        else:
            u = - self.K @ x + self.M @ np.concatenate(self.prediction_history[-self.lookback:])

        # update the action history
        self.action_history.append(u)
        self.current_step += 1
        return u
    
    # update the internal state and the feedback matrix M
    def policy_update(self, lr = None):
        if self.current_step < self.lookback:
            return
        
        # dynamics: x_{t+1} = (A - BK)x_t + B M v_t
        # compute the partial derivative of the dynamics w.r.t. x and M

        dxdx = self.A - self.B @ self.K
        prediction_lookback = np.concatenate(self.prediction_history[-self.lookback:])
        dxdM = np.zeros((self.n, self.m, self.n * self.lookback))
        for i in range(self.n):
            for j in range(self.m):
                dxdM[i, j, :] = self.B[i, j] * prediction_lookback
        dxdM = dxdM.reshape((self.n, self.m * self.n * self.lookback))

        # update the internal state
        self.internal_state = dxdx @ self.internal_state + dxdM

        # cost: c = x_t^T Q x_t + (-K x_t + M v_t)^T R (-K x_t + M v_t)
        # compute the partial derivative of the cost w.r.t. x and M
        dcdx = 2 * self.Q @ self.state_history[-1] - 2 * self.K.T @ self.R @ ( - self.K @ self.state_history[-1] + self.M @ prediction_lookback)
        dcdmv = 2 * self.R @ ( - self.K @ self.state_history[-1] + self.M @ prediction_lookback)
        dmvdM = np.zeros((self.m, self.m, self.n * self.lookback))
        for i in range(self.m):
            dmvdM[i, i, :] = prediction_lookback
        dmvdM = dmvdM.reshape((self.m, self.m * self.n * self.lookback))
        dcdM = dcdmv @ dmvdM
        G = dcdx @ self.internal_state + dcdM

        # update the feedback matrix M
        if lr is None:
            self.M -= self.lr * G.reshape((self.m, self.n * self.lookback))
        else:
            self.M -= lr * G.reshape((self.m, self.n * self.lookback))

        #print("Internal state: ", self.internal_state)

    
if __name__ == "__main__":
    # define a dynamical system with double integrator dynamics
    n = 2
    m = 1
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [dt]])
    Q = np.eye(n)
    R = np.eye(m)

    # construct the linear dynamics system
    sys = linearDynamics(A, B, Q, R, max_horizon=40000)

    # generate noise
    np.random.seed(2)
    w = np.random.normal(0, 1.0, (n, sys.MAX_HORIZON))
    v = w
    sys.set_noise(w, v)

    # construct the policy optimization object
    pol_opt = policyOptimization(A, B, Q, R, max_horizon=sys.MAX_HORIZON, lr = 5e-4)

    # record the average cost and the learned M
    avg_cost = []
    M_history = []

    # simulate the system for 100 steps
    for i in trange(sys.MAX_HORIZON):
        obs = sys.observe()
        action = pol_opt.act(obs)
        M_history.append(deepcopy(pol_opt.M))
        sys.step(action)
        avg_cost.append(sys.total_cost / (i + 1))
        pol_opt.policy_update()

    H = B @ np.linalg.inv(R + B.T @ pol_opt.P @ B) @ B.T
    print("Theoretical optimal cost (no prediction): ", np.trace(pol_opt.P))
    print("Theoretical optimal cost (with prediction): ", np.trace(pol_opt.P - pol_opt.P @ H @ pol_opt.P))
    print("Learned average cost: ", avg_cost[-1])

    # compute the theoretical optimal feedback matrix M
    P = pol_opt.P
    M_opt = - np.linalg.inv(R + B.T @ P @ B) @ B.T @ P

    # plot the average cost
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(avg_cost, label="M-GAPS average cost")
    plt.axhline(y=np.trace(pol_opt.P - pol_opt.P @ H @ pol_opt.P), linestyle='--', color='r', label="Theoretical optimal cost")
    plt.xlabel("Time step")
    plt.ylabel("Average cost")
    plt.title("Average cost over time")
    plt.legend()
    plt.savefig("Figures/average_cost.png")

    # color list of dashed horizon lines
    colors = ['r', 'g', 'b', 'c', 'm', 'y']

    # plot each entry of the learned M matrix
    plt.figure()
    for i in range(pol_opt.m):
        for j in range(pol_opt.n * pol_opt.lookback):
            plt.plot([M[i, j] for M in M_history], label="M[{}, {}]".format(i, j))
            # plot a horizontal line indicating the optimal value
            color_idx = (i * pol_opt.n * pol_opt.lookback + j) % len(colors)
            plt.axhline(y=M_opt[i, j], linestyle='--', color = colors[color_idx], label="Optimal M[{}, {}]".format(i, j))
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Learned M matrix over time")
    plt.legend()
    plt.savefig("Figures/learned_M.png")
# a simulator of the linear dynamical system

import numpy as np

class linearDynamics:
    def __init__(self, A, B, Q, R, max_horizon = 10000):
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.n = A.shape[0]
        self.m = B.shape[1]

        self.MAX_HORIZON = max_horizon
        self.w = np.zeros((self.n, self.MAX_HORIZON))
        self.v = np.zeros((self.n, self.MAX_HORIZON))

        self.current_state = np.zeros(self.n)
        self.total_cost = 0.0
        self.current_step = 0
    
    # reset the state of the system
    def reset(self, x0 = None):
        self.current_step = 0
        self.total_cost = 0.0
        if x0 is None:
            self.current_state = np.zeros(self.n)
        else:
            assert x0.shape == (self.n,)
            self.current_state = x0
    
    # set the noise and predictions
    def set_noise(self, w, v):
        assert w.shape == (self.n, self.MAX_HORIZON)
        assert v.shape == (self.n, self.MAX_HORIZON)
        self.w = w
        self.v = v
    
    # observe the current state and prediction
    def observe(self):
        return self.current_state, self.v[:, self.current_step]
    
    # simulate the system for one step
    def step(self, u):
        assert u.shape == (self.m,)
        if self.current_step >= self.MAX_HORIZON:
            raise ValueError("The maximum horizon is reached.")
        # update the state
        self.current_state = self.A @ self.current_state + self.B @ u + self.w[:, self.current_step]
        # calculate the cost
        cost = self.current_state @ self.Q @ self.current_state + u @ self.R @ u
        # update the total cost
        self.total_cost += cost
        # update the current step
        self.current_step += 1
        return self.current_state, cost

if __name__ == "__main__":
    # solve the discrete-time Riccati equation
    from scipy.linalg import solve_discrete_are
    n = 2
    m = 1
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [dt]])
    Q = np.eye(n)
    R = np.eye(m)
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    T = 100000
    print("K: ", K)

    # compute the theoretical average cost
    optimal_cost = np.trace(P)
    print("Theoretical average cost: ", optimal_cost)

    # simulate the system
    w = np.random.normal(0, 1.0, (n, T))
    v = np.zeros((n, T))
    
    sys = linearDynamics(A, B, Q, R, max_horizon = T)
    sys.set_noise(w, v)
    sys.reset()
    for t in range(T):
        state, pred = sys.observe()
        u = - K @ state
        sys.step(u)
    
    print("Empirical average cost: ", sys.total_cost / T)
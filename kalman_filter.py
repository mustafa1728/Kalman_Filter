import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Kalman_Filter():
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.R = None
        self.Q = None

        self.mu = None
        self.sigma = None
        self.t = 0

    def update_measurement(self, z):
        kalman_gain = self.sigma @ self.C.T @ np.linalg.pinv(self.C @ self.sigma @ self.C.T + self.Q)
        self.mu = self.mu + kalman_gain @ (z  - self.C @ self.mu)
        self.sigma = self.sigma - kalman_gain @ self.C @ self.sigma
    
    def update_motion(self, u):
        self.mu = self.A @ self.mu + self.B @ u
        self.sigma = self.A @ self.sigma @ self.A.T + self.R
        

    def get_state(self):
        return self.mu, self.sigma


class PlaneSimulator():
    def  __init__(self, x0, y0, vx0, vy0):
        self.state = np.array([[x0, y0, vx0, vy0]]).T
        self.t = 0
        self.del_t = 1

        self.A = np.array([[1, 0, self.del_t, 0],
                  [0, 1, 0, self.del_t],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
        self.B = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 1]])
        self.C = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

        self.R = np.diag((1, 1, 0.0001, 0.0001))
        self.Q = np.diag((100, 100))

    def update_time(self):
        self.t = self.t + self.del_t

    def forward_step(self, u=np.array([[0, 0]]).T):
        noise = np.random.multivariate_normal((0, 0, 0, 0), self.R)
        # print(noise.shape, self.A.shape, self.B.shape, self.state.shape, u.shape)
        self.state = self.A @ self.state + self.B @ u + noise.reshape(4, 1)
        # print(self.state.shape)
    
    def observation_step(self):
        noise = np.random.multivariate_normal((0, 0), self.Q)
        # print(noise.shape, self.C.shape, self.state.shape)
        self.observation = self.C @ self.state + noise.reshape(2, 1)
        return self.observation

    def set_estimator_matrices(self):
        self.estimator.A = self.A
        self.estimator.B = self.B
        self.estimator.C = self.C
        self.estimator.R = self.R
        self.estimator.Q = self.Q

    def set_kalman_filter(self):
        self.estimator = Kalman_Filter()
        self.set_estimator_matrices()

        self.estimator.mu = np.zeros((4, 1))
        self.estimator.sigma = np.diag((0.0001, 0.0001, 0.0001, 0.0001))





def simulate(action = "zero", estimate=False, accident_times=[], counter_size=20, save_name="trajectory.png"):
    true_all_xs = []
    true_all_ys = []
    obs_all_xs = []
    obs_all_ys = []


    simulator = PlaneSimulator(0, 0, 1, 1)
    if estimate:
        simulator.set_kalman_filter()
        est_all_xs = []
        est_all_ys = []
        variances = []
    accident_counter = -1
    for i in range(200):
        simulator.update_time()
        
        if simulator.t in accident_times:
            accident_counter = counter_size

        
        
            


        if action == "sine-cos":
            u = np.array([[np.sin(i), np.cos(i)]]).T
        else:
            u = np.array([[0, 0]]).T
        simulator.forward_step(u)
        if estimate:
            simulator.estimator.update_motion(u)
        observation = simulator.observation_step()
        if estimate and accident_counter <= 0:
            simulator.estimator.update_measurement(observation)
        if accident_counter > 0:
            accident_counter -= 1

        true_pos = simulator.state[:2, :]

        true_all_xs.append(true_pos[0, 0])
        true_all_ys.append(true_pos[1, 0])

        obs_all_xs.append(observation[0, 0])
        obs_all_ys.append(observation[1, 0])
        if estimate:
            mean, var = simulator.estimator.get_state()

            est_all_xs.append(mean[0, 0])
            est_all_ys.append(mean[1, 0])
            variances.append(var)



        
        

    ax = plt.subplot(111)
    ax.plot(true_all_xs, true_all_ys, label="True Trajectory")
    if not estimate or len(accident_times) == 0:
        ax.plot(obs_all_xs, obs_all_ys, label="Observed Trajectory")

    if estimate:
        ax.plot(est_all_xs, est_all_ys, label="Estimated Trajectory")
        for j in range(len(variances)):
            ell = Ellipse(xy=(est_all_xs[j], est_all_ys[j]),
                        width=variances[j][0, 0], height=variances[j][1, 1],
                        angle=np.rad2deg(np.arctan(variances[j][0, 1]/np.sqrt(variances[j][0, 0] * variances[j][1, 1]))),
                        alpha=0.2
                    )
            # ell.set_facecolor('b', )
            ax.add_artist(ell)
    plt.legend()
    plt.savefig(save_name, dpi=300)
    plt.close()




# ## part a
simulate(save_name="trajectory_parta.png")

# ## part b and c
simulate(action = "sine-cos", estimate=True, save_name="trajectory_partbc.png")

# ## part d and e
simulate(action = "sine-cos", estimate=True, accident_times=[10, 60], counter_size=20, save_name="trajectory_partde.png")


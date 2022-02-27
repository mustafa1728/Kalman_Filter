from sympy.utilities.iterables import multiset_permutations
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random

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
    
    def measurement_probability(self,z):
        cov = self.C @ self.sigma @ self.C.T + self.Q
        mu = self.C @ self.mu
        part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
        part2 = (-1/2) * ((z-mu).T.dot(np.linalg.inv(cov))).dot((z-mu))
        return float(part1 * np.exp(part2))
        
    def get_state(self):
        return self.mu, self.sigma


class PlaneSimulator():
    def  __init__(self, x0, y0, vx0, vy0, n_estimators=None):
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
        self.n_estimators = n_estimators

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

    def set_estimator_matrices(self,  i=None):
        if  i is None:
            self.estimator.A = self.A
            self.estimator.B = self.B
            self.estimator.C = self.C
            self.estimator.R = self.R
            self.estimator.Q = self.Q
        else:
            self.estimators[i].A = self.A
            self.estimators[i].B = self.B
            self.estimators[i].C = self.C
            self.estimators[i].R = self.R
            self.estimators[i].Q = self.Q

    def set_kalman_filter(self):
        if self.n_estimators  is None:
            self.estimator = Kalman_Filter()
            self.set_estimator_matrices()
            self.estimator.mu = self.state
            self.estimator.sigma = np.diag((0.0001, 0.0001, 0.0001, 0.0001))
        else:
            self.estimators = [Kalman_Filter() for i in range(self.n_estimators)]
            for i in range(self.n_estimators):
                self.set_estimator_matrices(i)
                self.estimators[i].mu = self.state
                self.estimators[i].sigma = np.diag((0.0001, 0.0001, 0.0001, 0.0001))







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


def simulate_f(no_planes=2, n_estimators=10, save_name="data_assoc.png"):
    simulators = []
    true_all_xs = {str(x):[]  for x in range(no_planes)}
    true_all_ys = {str(x):[]  for x in range(no_planes)}
    
    est_all_xs = {str(x):[]  for x in range(no_planes)}
    est_all_ys = {str(x):[]  for x in range(no_planes)}

    variances = {str(x): []  for x in range(no_planes)}

    x , y = random.randint(0,100) , random.randint(0,100)
    # vx , vy = random.randint(-10,10) , random.randint(-10,10)
    for  i in range(no_planes):
        # if  i == 0:
        #     x, y = 0, 0
        # else:
        #     x, y = 1000, 1000
        # x , y = random.randint(0,10) , random.randint(0,10)
        # vx , vy = random.randint(-1,1) , random.randint(-1,1)
        vx , vy = random.randint(-10,10) , random.randint(-10,10)
        simulator =  PlaneSimulator(x, y, vx, vy, n_estimators=n_estimators)
        simulator.set_kalman_filter()
        simulators.append(simulator)

    for time in range(50):
        print(time)
        u = np.array([[np.sin(time), np.cos(time)]]).T
        observations  =  []
        for simulator in simulators:
            simulator.forward_step(u)
            for j in range(n_estimators):
                simulator.estimators[j].update_motion(u)
            observation = simulator.observation_step()
            observations.append(observation)

        # np.random.shuffle(observations)
        
        all_permutations_results = {}
        observations_ids = list(range(len(observations)))
        for i in range(n_estimators):
            for iter_id, perm_id in enumerate(multiset_permutations(observations_ids)):
                perm = [observations[it] for it in perm_id]
                prob = 0.0
                for j in range(no_planes):
                    prob *= simulators[j].estimators[i].measurement_probability(perm[j])
                all_permutations_results[(i, iter_id)] = prob, perm
        all_permutations_results = {k: v for k, v in sorted(all_permutations_results.items(), key=lambda item: item[1][0])}



        
        old_simulators = copy.deepcopy(simulators)
        cnt = 0

        for (estimator,perm) in all_permutations_results.keys():
            for j in range(no_planes):
                simulators[j].estimators[cnt] = copy.deepcopy(old_simulators[j].estimators[estimator])
            cnt += 1
            if cnt == n_estimators:
                break

        cnt = 0
        for (estimator, perm) in all_permutations_results.keys():
            perm = all_permutations_results[estimator, perm][1]
            for j in range(no_planes):
                simulators[j].estimators[cnt].update_measurement(perm[j])
            cnt += 1
            if cnt == n_estimators:
                break

        for i in range(no_planes):
            true_pos = simulators[i].state[:2, :]

            true_all_xs[str(i)].append(true_pos[0, 0])
            true_all_ys[str(i)].append(true_pos[1, 0])

            mean, var = simulators[i].estimators[0].get_state()

            est_all_xs[str(i)].append(mean[0, 0])
            est_all_ys[str(i)].append(mean[1, 0])
            variances[str(i)].append(var)

        
                
        for simulator in simulators:
            simulator.update_time()
    
    
    for i in range(no_planes):
        ax = plt.subplot(111)
        ax.plot(true_all_xs[str(i)], true_all_ys[str(i)], label="True Trajectory {}".format(i+1))

        ax.plot(est_all_xs[str(i)], est_all_ys[str(i)], label="Estimated Trajectory {}".format(i+1))
        # for j in range(len(variances[str(i)])):
        #     ell = Ellipse(xy=(est_all_xs[str(i)][j], est_all_ys[str(i)][j]),
        #                 width=variances[str(i)][j][0, 0], height=variances[str(i)][j][1, 1],
        #                 angle=np.rad2deg(np.arctan(variances[str(i)][j][0, 1]/np.sqrt(variances[str(i)][j][0, 0] * variances[str(i)][j][1, 1]))),
        #                 alpha=0.2
        #             )
        #     # ell.set_facecolor('b', )
        #     ax.add_artist(ell)
        # break
    plt.legend()
    plt.savefig(save_name, dpi=300)
    plt.close()


    
        
        
    



# # ## part a
# simulate(save_name="trajectory_parta.png")

# # ## part b and c
# simulate(action = "sine-cos", estimate=True, save_name="trajectory_partbc_elip.png")

# # ## part d and e
# simulate(action = "sine-cos", estimate=True, accident_times=[10, 60], counter_size=20, save_name="trajectory_partde.png")




simulate_f(no_planes=5, n_estimators=10)
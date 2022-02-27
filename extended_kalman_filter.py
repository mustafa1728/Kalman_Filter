import numpy as np
import math 
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Extended_Kalman_Filter():
    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.R = None
        self.LQ = None
        self.NLQ = None
        self.H = None

        self.mu = None
        self.sigma = None
        self.t = 0

        self.landmarks = {}

    def update_linear_measurement(self, z):
        term1 = self.C @ self.sigma @ self.C.T + self.LQ
        term1 = np.array(term1, dtype='float')
        kalman_gain = self.sigma @ self.C.T @ np.linalg.pinv(term1)
        self.mu = self.mu + kalman_gain @ (z  - self.C @ self.mu)
        self.sigma = self.sigma - kalman_gain @ self.C @ self.sigma

    def update_non_linear_measurement(self, distance, landmark):
        x , y = self.landmarks[landmark]
        self.H = np.array([[(self.mu[0][0] - x)/distance, float(self.mu[1][0]-y)/distance, 0, 0]])
        # print((self.H @ self.sigma @ self.H.T).shape, self.NLQ.shape, self.H @ self.sigma @ self.H.T + self.NLQ)
        kalman_gain = self.sigma @ self.H.T @ (1/(self.H @ self.sigma @ self.H.T + self.NLQ))
        
        # print(kalman_gain.)
        self.mu = self.mu + kalman_gain * float(distance - math.sqrt((self.mu[0]-x)**2 + (self.mu[1]-y)**2))
        self.sigma = self.sigma - kalman_gain @ self.H @ self.sigma
    
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

        self.landmarks = {
            "L1" : (100 , 100),
            "L2" : (-100 , -100),
            "L3" : (-100 , 100),
            "L4" : (100 , -100),
            "L5" : (0 , 0)
        }

        self.H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

        self.R = np.diag((1, 1, 0.0001, 0.0001))
        self.LQ = np.diag((100, 100))
        self.NLQ = np.array([[1]])

    def update_time(self):
        self.t = self.t + self.del_t

    def forward_step(self, u=np.array([[0, 0]]).T):
        noise = np.random.multivariate_normal((0, 0, 0, 0), self.R)
        # print(noise.shape, self.A.shape, self.B.shape, self.state.shape, u.shape)
        self.state = self.A @ self.state + self.B @ u + noise.reshape(4, 1)
        # print(self.state.shape)
    
    def observation_step(self):
        noise = np.random.multivariate_normal((0, 0), self.LQ)
        self.linear_observation = self.C @ self.state + noise.reshape(2, 1)
        self.non_linear_observation = None
        landmark_found = None
        noise = np.random.multivariate_normal([0], self.NLQ)
        for landmark in self.landmarks.keys():
            x,y = self.landmarks[landmark]
            dist =  math.sqrt((self.state[0]-x)**2 + (self.state[1]-y)**2) 
            if ( dist < 30.0 ):
                self.non_linear_observation = dist + float(noise.reshape(1, 1))
                landmark_found = landmark
        self.observation = (self.linear_observation,self.non_linear_observation, landmark_found)
        # print(noise.shape, self.C.shape, self.state.shape)
        return self.observation

    def set_estimator_matrices(self):
        self.estimator.A = self.A
        self.estimator.B = self.B
        self.estimator.C = self.C
        self.estimator.R = self.R
        self.estimator.LQ = self.LQ
        self.estimator.NLQ = self.NLQ
        self.estimator.landmarks  = self.landmarks

    def set_kalman_filter(self):
        self.estimator = Extended_Kalman_Filter()
        self.set_estimator_matrices()
        self.estimator.mu = self.state
        self.estimator.sigma = np.diag((0.0001, 0.0001, 0.0001, 0.0001))



def simulate(action = "zero", estimate=False, accident_times=[], counter_size=20, save_name="trajectory.png", add_landmarks=[]):
    true_all_xs = []
    true_all_ys = []

    simulator = PlaneSimulator(30, -70, 0.1, 0.1)
    # simulator = PlaneSimulator(0, 0, 0.1, 0.1)
    # simulator = PlaneSimulator(30, -70, 4*np.cos(0.3), 4*np.sin(0.3))
    # simulator = PlaneSimulator(30, -70, 0.5*np.cos(0.75), 0.5*np.sin(0.74))

    for i, l in enumerate(add_landmarks):
        simulator.landmarks[str(i) + "landmark"] = l
    if estimate:
        simulator.set_kalman_filter()
        est_all_xs = []
        est_all_ys = []
        variances = []
    accident_counter = -1
    for i in range(800):
        simulator.update_time()
        
        if simulator.t in accident_times:
            accident_counter = counter_size

        if action == "sine-cos":
            u = np.array([[np.sin(i), np.cos(i)]]).T
        else:
            u = np.array([[0, 0]]).T
        if len(true_all_xs) == 0:
            u = np.array([[0, 0]]).T
        else:
            u =  np.array([[-0.001*true_all_xs[-1], -0.0005*true_all_ys[-1]]]).T
        simulator.forward_step(u)
        if estimate:
            simulator.estimator.update_motion(u)
        linear_observation, non_linear_observation, landmark = simulator.observation_step()
        if estimate and accident_counter <= 0:
            simulator.estimator.update_linear_measurement(linear_observation)
            if non_linear_observation is not None:
                simulator.estimator.update_non_linear_measurement(non_linear_observation, landmark)

        mean, var = simulator.estimator.get_state()

        est_all_xs.append(mean[0, 0])
        est_all_ys.append(mean[1, 0])
        variances.append(var)

        if accident_counter > 0:
            accident_counter -= 1

        true_pos = simulator.state[:2, :]

        true_all_xs.append(true_pos[0, 0])
        true_all_ys.append(true_pos[1, 0])


    ax = plt.subplot(111)
    ax.plot(true_all_xs, true_all_ys, label="True Trajectory")

    ax.plot(est_all_xs, est_all_ys, label="Estimated Trajectory")
    for j in range(len(variances)):
        ell = Ellipse(xy=(est_all_xs[j], est_all_ys[j]),
                    width=variances[j][0, 0], height=variances[j][1, 1],
                    angle=np.rad2deg(np.arctan(variances[j][0, 1]/np.sqrt(variances[j][0, 0] * variances[j][1, 1]))),
                    alpha=0.2
                )
        # ell.set_facecolor('b', )
        ax.add_artist(ell)
    landmarks_x = []
    landmarks_y = []
    for l in simulator.landmarks.keys():
        landmarks_x.append(simulator.landmarks[l][0])
        landmarks_y.append(simulator.landmarks[l][1])
    plt.scatter(landmarks_x, landmarks_y, s=10, c="r", label="landmarks")

    plt.legend()
    plt.savefig(save_name, dpi=300)
    plt.close()




# ## part a
# simulate(save_name="trajectory_parta.png")

# ## part b and c
# simulate(action = "sine-cos", estimate=True, accident_times=[10, 60], counter_size=20, save_name="trajectory_partde.png")

# simulate(action = "sine-cos", estimate=True, save_name="trajectory_q2.png")


simulate(action = "sine-cos", estimate=True, save_name="trajectory_q2_2.png", add_landmarks=[(-50, 50), (50, 50), (40, -50), (-50, -50)])


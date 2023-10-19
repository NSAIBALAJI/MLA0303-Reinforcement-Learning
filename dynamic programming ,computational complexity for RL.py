import numpy as np 
n_states = 3  # Number of states 
n_actions = 2  # Number of actions 
P = np.zeros((n_states, n_actions, n_states)) 
P[0, 0, 0] = 0.7 
P[0, 0, 1] = 0.3 
P[0, 1, 1] = 0.5 
P[0, 1, 2] = 0.5 
P[1, 0, 0] = 0.4 
P[1, 0, 1] = 0.6 
P[1, 1, 0] = 0.1 
P[1, 1, 1] = 0.9 
P[2, 0, 2] = 1.0 
P[2, 1, 2] = 1.0 
R = np.zeros((n_states, n_actions, n_states)) 
R[0, 0, 0] = 1.0 
R[0, 0, 1] = 2.0 
R[0, 1, 1] = 3.0 
R[0, 1, 2] = 4.0 
R[1, 0, 0] = 0.0 
R[1, 0, 1] = 2.0 
R[1, 1, 0] = 1.0 
R[1, 1, 1] = 3.0 
R[2, 0, 2] = 0.0 
R[2, 1, 2] = 0.0 
def value_iteration(P, R, gamma, epsilon=1e-6): 
n_states, n_actions, _ = P.shape 
V = np.zeros(n_states) 
while True: 
V_new = np.zeros(n_states) 
for s in range(n_states): 
Q_s = np.zeros(n_actions) 
for a in range(n_actions): 
for s_prime in range(n_states): 
Q_s[a] += P[s, a, s_prime] * (R)

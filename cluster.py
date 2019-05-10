import pandas as pd
#import matplotlib.pyplot as plt
import math
import numpy as np
from mpi4py import MPI

df = pd.read_csv('PimaIndians.csv')

comm = MPI.COMM_WORLD
p = comm.Get_size()
r = comm.Get_rank()

previousMSE = 0
MSE = math.inf

neg = []
pos = []
for x in range(1, 392):
    if df.loc[x]['test'] == 0:
        neg.append(df.loc[x])
    else:
        pos.append(df.loc[x])

neg = pd.DataFrame.from_dict(neg)
pos = pd.DataFrame.from_dict(pos)

#plt.scatter(neg['glucose'], neg['triceps'], marker='o')
#plt.scatter(pos['glucose'], pos['triceps'], marker='X')
#plt.xlabel('glucose')
#plt.ylabel('diastolic')
#plt.show()

# Saving the essential data from the DataFrame
glucose_triceps = df.copy()
glucose_triceps = glucose_triceps.drop(columns=['pregnant', 'diastolic', 'bmi', 'insulin', 'age', 'diabetes', 'test'])
glucose_triceps = np.array(glucose_triceps)
glucose = np.array(df.copy()['glucose'])
triceps = np.array(df.copy()['triceps'])

def init_centroids(data, k):
    centroids = data.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

buf_size = int(len(glucose_triceps) / p)

# Start K-Means
# Creating buffers for data to send to proceses
k = 2
glubuf = np.zeros(buf_size * 2, dtype='i')
tribuf = np.zeros(buf_size * 2, dtype='i')
seeds = np.zeros([2,2])
if r == 0:
    seeds = init_centroids(glucose_triceps, 2)
    comm.Bcast(seeds, root=0)
    comm.Scatter(glucose, glubuf, root=0)
    comm.Scatter(triceps, tribuf, root=0)

# This removes the extraneous zeros
glubuf = np.delete(glubuf, np.s_[1::2], 0)
tribuf = np.delete(tribuf, np.s_[1::2], 0)

# seeds[0] = glucose centers
# seeds[0][0] = glucose center 1
# seeds[1] = triceps centers
# seeds[1][0] = triceps tenter 1

d = np.zeros([2, len(glubuf)])

# Not finished yet
while MSE == previousMSE:
    previousMSE = MSE
    MSE_t = 0
    #for j in range(1, k + 1):
    m_t = np.zeros(k + 1)
    n_t = np.zeros(k + 1)
    n = len(glubuf)
    for i in range(r * n / p, (r + 1) * (n / p)):
        for j in range(0, k):
            d[j][i] = ((glubuf[i] - seeds[0][j]) ** 2) + ((tribuf[i] - seeds[1][j]) ** 2)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

df = pd.read_csv('PimaIndians.csv')

comm = MPI.COMM_WORLD
p = comm.Get_size()
r = comm.Get_rank()

# Used to plot diabetics and non-diabetics
neg = []
pos = []
for x in range(1, 392):
    if df.loc[x]['test'] == 0:
        neg.append(df.loc[x])
    else:
        pos.append(df.loc[x])

neg = pd.DataFrame.from_dict(neg)
pos = pd.DataFrame.from_dict(pos)

# Creating scatter plot
if r == 0:
    plt.scatter(neg['glucose'], neg['triceps'], marker='o', label='Non-Diabetic')
    plt.scatter(pos['glucose'], pos['triceps'], marker='X', label='Diabetic')
    plt.xlabel('glucose')
    plt.ylabel('triceps')
    plt.legend()
    plt.show()

# Saving the essential data from the DataFrame
glucose_triceps = df.copy()
glucose_triceps = glucose_triceps.drop(columns=['pregnant', 'diastolic', 'bmi', 'insulin', 'age', 'diabetes', 'test'])
glucose_triceps = np.array(glucose_triceps)
glucose = np.array(df.copy()['glucose'])
triceps = np.array(df.copy()['triceps'])
test = df.copy()['test']

# Functions for k-means
def init_centroids(data, k):
    centroids = data.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

def move_centroids(points, closest, centroids):
    """returns the new centroids assigned from the points closest to them"""
    return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])

buf_size = int(len(glucose_triceps) / p)

# Start K-Means
# Creating buffers for data to send to proceses
glubuf = np.zeros(buf_size * 2, dtype='i')
tribuf = np.zeros(buf_size * 2, dtype='i')
seeds = np.zeros([4,4], dtype='i')
if r == 0:
    seeds = init_centroids(glucose_triceps, 2)

comm.Scatter(glucose, glubuf, root=0)
comm.Scatter(triceps, tribuf, root=0)
comm.Bcast(seeds[0])
comm.Bcast(seeds[1])

# Removing the extra zeros
if r != 0:
    seeds = np.delete(seeds, np.s_[1::2], 1)
    seeds = seeds[:2]
# This removes the extraneous zeros
glubuf = np.delete(glubuf, np.s_[1::2], 0)
tribuf = np.delete(tribuf, np.s_[1::2], 0)

# seeds[0] = glucose centers
# seeds[0][0] = glucose center 1
# seeds[1] = triceps centers
# seeds[1][0] = triceps tenter 1

glutri = np.append(np.vstack(glubuf), np.vstack(tribuf), axis=1)
# Doing 2-Means 100 times to find the center
for a in range(0, 100):
    closest = closest_centroid(glutri, seeds)
    new_seeds = move_centroids(glutri, closest, seeds)

    sum_seeds = np.zeros(4, dtype=float)
    new_seeds = np.reshape(new_seeds, (1, 4))
    comm.Allreduce(new_seeds, sum_seeds, op=MPI.SUM)
    sum_seeds /= p
    seeds = np.reshape(sum_seeds, (2, 2))

# Creating scatter plot of data with clustered classes
# Also calculating Accuracy, Confusion Matrix, and GINI index
if r == 0:
    closest = closest_centroid(glucose_triceps, seeds)
    print('Center 1: ', seeds[0])
    print('Center 2: ', seeds[1])
    plt.scatter(glucose, triceps, c=closest)
    plt.scatter(seeds[:,0], seeds[:,1], marker='X')
    plt.show()
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for a in range(0, len(df)):
        if test[a] == 0:
            if closest[a] == 0:
                tn += 1
            else:
                fn += 1
        else:
            if closest[a] == 1:
                tp += 1
            else:
                fp += 1
    if tn < fn:
        temp1 = tn
        temp2 = tp
        tp = fp
        tn = fn
        fn = temp1
        fp = temp2
    accuracy = (tn + tp) / (len(df))
    print('Accuracy: ', accuracy)
    print('Confusion Matrix: ', tp, fp)
    print('                  ', fn, tn)
    gini = 2 * (len(neg) / len(df)) * (len(pos) / len(df))
    print('GINI index: ', gini)

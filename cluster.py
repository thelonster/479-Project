import pandas as pd
import matplotlib.pyplot as plt
import math
#from mpi4py import MPI

df = pd.read_csv('PimaIndians.csv')

#comm = MPI.COMM_WORLD
#p = comm.Get_size()
#r = comm.Get_rank()

previousMSE = 0
MSE = math.inf

#fig = plt.figure()

df = df.sort_values(by=['test'])

neg = []
pos = []
for x in range(1, 392):
    if df.loc[x]['test'] == 0:
        neg.append(df.loc[x])
    else:
        pos.append(df.loc[x])

#print(neg['test'])
neg = pd.DataFrame.from_dict(neg)
pos = pd.DataFrame.from_dict(pos)
#print(neg['diabetes'])
#print(pos)

plt.scatter(neg['glucose'], neg['bmi'], marker='o')
plt.scatter(pos['glucose'], pos['bmi'], marker='X')
plt.xlabel('glucose')
plt.ylabel('bmi')
plt.show()

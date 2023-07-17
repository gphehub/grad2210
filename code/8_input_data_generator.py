import numpy as np
import math
#from sklearn.preprocessing import normalize

import csv
import pickle



num_data = 10000
num_qubits = 3



# Generating input data for grad2210:
values = []
results = []

for i in range(num_data):
    value = np.random.rand(2 ** num_qubits)
    result = num_qubits - 1 # can be any integer lower than num_qubits.
    values.append(value)
    results.append(result)

values = np.array(values, dtype='float32')
results = np.array(results, dtype='int64')
data = [values, results]

filesave1 = open('data/input_data_8d.pickle', "wb")
pickle.dump(data, filesave1)
filesave1.close()

print('Input data generated.')

'''
fileload = open('data/input_data_8d.pickle', "rb")
tr_d = pickle.load(fileload, encoding='bytes')
fileload.close()
'''



'''
# Generating initial parameters for the RealAmplitudes ansatz of grad2210:
repetitions = 2
param_num = num_qubits * (repetitions + 1)
param = np.random.rand(param_num) * math.pi
np.savetxt('data/parameters_9.csv', param, delimiter=",")

print('Initial parameters generated.')
'''

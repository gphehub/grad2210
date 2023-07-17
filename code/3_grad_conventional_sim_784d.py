"""
This program computes the gradients using the conventional approach.

Take 784-dimensional inputs.

Run on a quantum simulator.

All the adjustable parameters are marked by '#Adjustable parameter'.
"""



import copy #mod_784d_v9



import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
#from qiskit.providers.aer import Aer
from qiskit.visualization import plot_histogram

import csv #new grad2210a
# import json #new grad2210a



from sklearn.preprocessing import normalize
import math

from numpy import genfromtxt

def mapping(param, raw_data, starting_index = 0, num_data = 1,
            shots = 1000, training = True,
            num_qubits = 10, repetitions = 2): #mod_8d1r #mod_784d_v9
    # Using Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    shots=shots
    num_qubits = num_qubits #mod_784d_v9
    repetitions = repetitions #mod_784d_v9



    # Amplitude encoding:
    #num_data = len(raw_data[0])

    num_param = len(param)
    num_bits = num_qubits * (num_param * 2 + 1) * num_data

    def FEATURE_MAP(data):
        classical_state = normalize([data])
        desired_state = classical_state[0].tolist()
        for i in range(2 ** num_qubits - 784): desired_state.append(0.0)
        return desired_state



    # Ansatz:
    def real_amp(param, num_qubits=num_qubits, repetitions = repetitions):
        varform = QuantumCircuit(num_qubits, 0)
        for i in range(num_qubits): varform.ry(param[i], i)
        for j in range(repetitions):
            for m in range(num_qubits-1):
                for n in range(m+1, num_qubits): varform.cx(m, n) # full entanglement
            varform.barrier(range(num_qubits))
            for i in range(num_qubits): varform.ry(param[(j + 1) * num_qubits + i], i)
        return varform



    # Building the quantum circuit:
    print('Running the conventional approach:')
    print('Building the quantum circuit.')
    time0 = time.time()
    time_spent = [0, 0]

    circuit_qc = QuantumCircuit(num_qubits, num_bits)

    for i in range(0, num_data):
        #print('index =', i + starting_index)
        xi = raw_data[0][i + starting_index]
        for j in range(0, num_param * 2 + 1):
            qc = QuantumCircuit(num_qubits, num_bits)
            qc.reset(range(num_qubits))

            #qc.initialize(desired_state, range(num_qubits))
            qc.initialize(FEATURE_MAP(xi), range(num_qubits))
            #qc.decompose().decompose().decompose().decompose().decompose().draw('mpl',scale = 0.3,style={'fontsize': 5})

            param_shifted = copy.deepcopy(param) #mod_784d_v9
            if j != 0:
                param_shifted[int((j-1)/2)] = param[int((j-1)/2)] - (-1)**j * math.pi / 2
            qc = qc.compose(real_amp(param_shifted)) #new grad2210a

            # Measuring:
            qc.measure(range(num_qubits), range((i * (num_param * 2 + 1) + j) * num_qubits,
                                                (i * (num_param * 2 + 1) + j + 1) * num_qubits))
            qc.barrier(range(num_qubits))
            circuit_qc = circuit_qc.compose(qc)

            #circuit_qc.draw('mpl',scale = 0.3,style={'fontsize': 8})



    #Computing:
    time1 = time.time()
    time_spent[0] = time1 - time0
    print('Time spent on building the circuit:', round(time_spent[0], 1), 'seconds.')
    print('Computing.')

    job = execute(circuit_qc, simulator, shots=shots)
    result = job.result()
    counts = result.get_counts(circuit_qc)

    time2 = time.time()
    time_spent[1] = time2 - time1
    print('Time spent on computing:', round(time_spent[1], 1), 'seconds.')

    filesave0 = open('data/job_result_conv.pickle', "wb")
    pickle.dump(result, filesave0)
    filesave0.close()

    #plot_histogram(counts)



    #Processing the quantum measurement result:
    print('Processing the result.')

    # Vectorizing the result for shots>=1:
    mapped_data = []
    mapped_values = np.zeros((num_data, num_param * 2 + 1, num_qubits))
    for key, count in counts.items():
        for i in range(0, num_data):
            for j in range(0, num_param * 2 + 1):
                for k in range(num_qubits):
                    mapped_values[i][j][num_qubits - 1 - k] += int(key[k + ((num_data - 1 - i) * (num_param * 2 + 1)
                                                           + (num_param * 2 - j)) * num_qubits]) * count / shots

    """
    print(mapped_values)
    mapped_values = [np.reshape(x, (10, 1)) for x in mapped_values]
    print(mapped_values)
    """
    if training:
        training_results = [vectorized_result(y, num_qubits) for y in raw_data[1]]
    else:
        training_results = raw_data[1]
    mapped_data = list(zip(mapped_values, training_results))
    print('Job completed.')
    return mapped_data, time_spent



    #import matplotlib.pyplot as plt
    #plt.show()



def vectorized_result(j, num_qubits):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((num_qubits, 1))
    e[j] = 1.0
    return e



#new grad2210a:
def cost_function(mapped_data):
    num_data = len(mapped_data)
    num_qubits = len(mapped_data[0][1])
    num_param = int((len(mapped_data[0][0]) - 1) / 2)
    individual_cost = np.zeros((num_data, num_param * 2 + 1))
    cost = np.zeros(num_param * 2 + 1)
    for j in range(num_param * 2 + 1):
        for i in range(num_data):
            for k in range(num_qubits):
                individual_cost[i][j] += abs(float(mapped_data[i][0][j][k] - mapped_data[i][1][k]))
            cost[j] += individual_cost[i][j]
        cost[j] /= num_data
    return cost
#end of new grad2210a.



# Test run:
# Based on mnist_loader.load_data():
import pickle
import gzip
#import numpy as np

import time

f = gzip.open('data/mnist.pkl.gz', 'rb')
tr_d, va_d, te_d = pickle.load(f,encoding='bytes')
f.close()

#import mnist_loader
#tr_d, va_d, te_d = mnist_loader.load_data_wrapper()

#test = mapping(tr_d, starting_index = 4, num_data = 5)
#print(test)



# Initializing the parameters:

#param = np.random.rand(num_qubits * (repetitions + 1)) * math.pi
#np.savetxt('data/parameters.csv', param, delimiter=",")

param0 = genfromtxt('data/parameters.csv', delimiter=',', skip_header = 0) #mod_784d_v9

gradient = np.zeros(len(param0))


# Running the quantum network:
num_qubits = 10
repetitions = 2 #Adjustable paramete
shots = 500 #Adjustable paramete
num_data = 1 #Adjustable paramete

mapped_data, time_spent = mapping(param0, tr_d,
                                  starting_index = 0, num_data = num_data,
                                  shots = shots,
                                  num_qubits = num_qubits, repetitions = repetitions) #mod_8d1r #mod_8d1r #mod_784d_v9

'''
mapped_data, time_spent = mapping(param0, tr_d,
                                  starting_index = 0, num_data = 1)
'''

filesave1 = open('data/mapped_data_conv.pickle', "wb")
pickle.dump(mapped_data, filesave1)
filesave1.close()


'''
# Loading saved mapped_data:
fileload = open('data/mapped_data_conv.pickle', "rb")
mapped_data = pickle.load(fileload, encoding='bytes')
fileload.close()
'''


#Calculating the gradient:
cost = cost_function(mapped_data)
for i in range(len(param0)):
    gradient[i] = (cost[2 * i + 1] - cost[2 * i + 2]) / 2
    #print('gradient = ', gradient[i], 'for parameter', i)
    #print(' ')

np.savetxt('data/gradient_conv.csv', gradient, delimiter=",")

np.savetxt('data/cost_conv.csv', cost, delimiter=",") #mod_784d_v9

#print('Result of the conventional approach:')
#print('Time spent on building the circuit:', time_spent[0], 'seconds.')
#print('Time spent on computing:', time_spent[1], 'seconds.')
print('Cost function = ', cost[0])
print('Total time spent:', round(time_spent[0] + time_spent[1], 1), 'seconds.')

with open('data/other_results_conv.csv','w', newline='') as resultsave: #mod_3a
    writer=csv.writer(resultsave) #mod_3a
    writer.writerow(('num_qubits:', num_qubits)) #mod_8d1r #mod_784d_v9
    writer.writerow(('repetitions in the RealAmplitudes ansatz:', repetitions)) #mod_8d1r #mod_784d_v9
    writer.writerow(('num_data:', num_data)) #mod_8d1r #mod_784d_v9
    writer.writerow(('Time spent on building the circuit (seconds):', time_spent[0]))
    writer.writerow(('Time spent on computing (seconds):', time_spent[1])) #mod_3a
    writer.writerow(('Total time spent (seconds):', time_spent[0] + time_spent[1]))
    writer.writerow(('Cost function:', cost[0])) #mod_3a

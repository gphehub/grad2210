"""
This program trains a quantum classifier using the improved approach.
Take the Iris dataset as input.
Run on a real quantum hardware/quantum simulator.
All the adjustable parameters are marked by '#Adjustable parameter'.

Input files:
data/epoch_0_parameters.csv
data/epoch_i_training_data.pickle (i=0, ... , total_training_epochs - 1)
"""



from sklearn.datasets import load_iris #mod_iris
ds = load_iris() #mod_iris

import numpy as np
from qiskit import(
  QuantumCircuit,
  execute,
  Aer)
from qiskit.visualization import plot_histogram

import csv #new grad2210a

from sklearn.preprocessing import normalize
import math

from numpy import genfromtxt

def mapping(param, raw_data, starting_index = 0, num_data = 3,
            shots = 1000, training = False, #mod_iris
            num_qubits = 3, repetitions = 1): #mod_8d1r
    # Using Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    shots=shots
    num_qubits = num_qubits #mod_8d1r
    repetitions = repetitions #mod_8d1r



    # Amplitude encoding:
    #num_data = len(raw_data[0])
    num_param = num_qubits * (repetitions + 1) #mod_8d1r
    num_bits = (num_qubits + 2 + num_param * 2) * num_data

    def FEATURE_MAP(data):
        classical_state = normalize([data])
        desired_state = classical_state[0].tolist()
        for i in range(2 ** num_qubits - 4): desired_state.append(0.0) #mod_8d #mod_iris
        return desired_state



    # Ansatz:
    def real_amp(param, data_index, num_qubits=num_qubits, repetitions = repetitions):
        varform = QuantumCircuit(num_qubits + 2, num_bits)
        varform.initialize([0, 1], 0)
        
        for i in range(num_qubits): varform.ry(param[i], i + 2)


        for i in range(num_qubits): # control circuit
            prob = 1 / (num_param * 2 + 1 - i * 2)
            varform.cry(np.arcsin(math.sqrt(prob)) * 2, 0, 1)
            varform.measure(1, ((num_qubits + 2) + num_param * 2) * data_index
                               + (num_qubits + 2) + (i * 2))
            varform.cry(math.pi / 2, 1, i + 2)
            varform.cx(1, 0)
            varform.reset(1)
            varform.barrier(range(num_qubits + 2))

            prob = 1 / (num_param * 2 + 1 - i * 2 -1)
            varform.cry(np.arcsin(math.sqrt(prob)) * 2, 0, 1)
            varform.measure(1, ((num_qubits + 2) + num_param * 2) * data_index
                               + (num_qubits + 2) + (i * 2) + 1)
            varform.cry(- math.pi / 2, 1, i + 2)
            varform.cx(1, 0)
            varform.reset(1)
            varform.barrier(range(num_qubits + 2))


        for j in range(repetitions):
            for m in range(2, num_qubits + 1):
                for n in range(m+1, num_qubits + 2): varform.cx(m, n) # full entanglement
            varform.barrier(range(num_qubits + 2))
            for i in range(num_qubits): varform.ry(param[(j + 1) * num_qubits + i], i + 2)


            for i in range(num_qubits): # control circuit
                prob = 1 / (num_param * 2 + 1 - num_qubits * (j + 1) * 2 - i * 2)
                varform.cry(np.arcsin(math.sqrt(prob)) * 2, 0, 1)
                varform.measure(1, ((num_qubits + 2) + num_param * 2) * data_index
                                   + (num_qubits + 2) + num_qubits * (j + 1) * 2 + i * 2)
                varform.cry(math.pi / 2, 1, i + 2)
                varform.cx(1, 0)
                varform.reset(1)
                varform.barrier(range(num_qubits + 2))

                prob = 1 / (num_param * 2 + 1 - num_qubits * (j + 1) * 2 - i * 2 -1)
                varform.cry(np.arcsin(math.sqrt(prob)) * 2, 0, 1)
                varform.measure(1, ((num_qubits + 2) + num_param * 2) * data_index
                                   + (num_qubits + 2) + num_qubits * (j + 1) * 2 + i * 2 + 1)
                varform.cry(- math.pi / 2, 1, i + 2)
                varform.cx(1, 0)
                varform.reset(1)
                varform.barrier(range(num_qubits + 2))


        return varform



    # Building the quantum circuit:
    time0 = time.time()
    time_spent = [0, 0]
    
    circuit_qc = QuantumCircuit(num_qubits + 2, num_bits)

    for i in range(0, num_data):
        #print('index =', i + starting_index)
        xi = raw_data[0][i + starting_index]
        qc = QuantumCircuit(num_qubits + 2, num_bits)
        qc.reset(range(num_qubits + 2))

        #qc.initialize(desired_state, range(num_qubits))
        qc.initialize(FEATURE_MAP(xi), range(2, num_qubits + 2))
        #qc.decompose().decompose().decompose().decompose().decompose().draw('mpl',scale = 0.3,style={'fontsize': 5})

        qc = qc.compose(real_amp(param, i)) #new grad2210a

        # Measuring:
        qc.measure(range(num_qubits + 2), range(i * ((num_qubits + 2) + num_param * 2),
                                                i * ((num_qubits + 2) + num_param * 2)
                                                + (num_qubits + 2)))
        qc.barrier(range(num_qubits + 2))
        circuit_qc = circuit_qc.compose(qc)

        #circuit_qc.draw('mpl',scale = 0.3,style={'fontsize': 8})



    #Computing:
    time1 = time.time()
    time_spent[0] = time1 - time0
    #print('Time spent on building the circuit:', round(time_spent[0], 1), 'seconds.') #mod_iris
    #print('Computing.') #mod_iris
    
    job = execute(circuit_qc, simulator, shots = shots * (num_param * 2 + 1))
    result = job.result()
    counts = result.get_counts(circuit_qc)

    time2 = time.time()
    time_spent[1] = time2 - time1
    #print('Time spent on computing:', round(time_spent[1], 1), 'seconds.') #mod_iris

    ''' #mod_iris
    filesave0 = open('data/job_result_impr.pickle', "wb")
    pickle.dump(result, filesave0)
    filesave0.close()
    '''

    #plot_histogram(counts)



    #Processing the quantum measurement result:

    # Vectorizing the result for shots>=1:
    mapped_values = np.zeros((num_data, num_param * 2 + 1, num_qubits))
    individual_shot = np.zeros((num_data, num_param * 2 + 1))
    for key, count in counts.items():
        for i in range(0, num_data):
            if int(key[(num_data - i) * ((num_qubits + 2) + num_param * 2) - 1]) == 1:
                individual_shot[i][0] += count
                for k in range(num_qubits):
                    mapped_values[i][0][num_qubits - 1 - k] += int(key[k + num_param * 2
                                                     + (num_data - 1 - i) * ((num_qubits + 2) + num_param * 2)]) * count
            else:
                for j in range(1, num_param * 2 + 1):
                    if int(key[(num_data - i) * ((num_qubits + 2) + num_param * 2) - (num_qubits + 2) - j]) == 1:
                        individual_shot[i][j] += count
                        for k in range(num_qubits):
                            mapped_values[i][j][num_qubits - 1 - k] += int(key[k + num_param * 2
                                                             + (num_data - 1 - i) * ((num_qubits + 2) + num_param * 2)]) * count
    for i in range(0, num_data):
        for j in range(0, num_param * 2 + 1):
            for k in range(num_qubits):
                mapped_values[i][j][k] /= individual_shot[i][j]

    mapped_data = []
    if training:
        training_results = [vectorized_result(y, num_qubits) for y in raw_data[1]]
    else:
        training_results = raw_data[1]
    mapped_data = list(zip(mapped_values, training_results))
    return mapped_data, individual_shot, time_spent



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



def cost_function(mapped_data):
    num_data = len(mapped_data)
    #num_qubits = len(mapped_data[0][1]) #mod_iris
    num_param = int((len(mapped_data[0][0]) - 1) / 2)
    individual_cost = np.zeros((num_data, num_param * 2 + 1))
    cost = np.zeros(num_param * 2 + 1)
    for j in range(num_param * 2 + 1):
        for i in range(num_data):
            k = mapped_data[i][1] #mod_iris
            individual_cost[i][j] -= np.log(mapped_data[i][0][j][k]) #mod_iris Following P9/Eq.(5) of ml94v2. # Note that it is "-=".
            
            cost[j] += individual_cost[i][j]

        #cost[j] /= num_data #mod_iris
        cost[j] /= (num_data * num_qubits) #mod_iris

    return cost



# MAIN PROGRAM:
import pickle
import gzip
#import numpy as np

import time

# Initializing the parameters:

#param = np.random.rand(num_qubits * (repetitions + 1)) * math.pi
#np.savetxt('data/parameters.csv', param, delimiter=",") #mod_8d #mod_8d1r

param0 = genfromtxt('data/epoch_0_parameters.csv', delimiter=',', skip_header = 0) #mod_8d #mod_8d1r 

gradient = np.zeros(len(param0))


# Running the quantum network:
num_qubits = 3
repetitions = 4 #Adjustable parameter
shots = 500 #Adjustable parameter
num_data = 40 #Adjustable parameter

#mod_iris
starting_index = 0
dim = 4 # Dimension of the input data set. Must not exceed (2 ** num_qubits). For MNIST it is 784.
total_training_epochs = 50 # ENTER YOUR CHOICE HERE.
learning_rate = 2 # ENTER YOUR CHOICE HERE.
#num_validation_data = 150 # Enter: 1~150 #mod_iris
#starting_epoch = int(genfromtxt('data/starting_epoch.csv', delimiter=',', skip_header = 0))
starting_epoch = 0
localtime = time.localtime(time.time())
strtime = time.strftime("%Y-%m-%d %H:%M:%S", localtime)
print('Program starts at', strtime, '\n')
print('Epoch   Cost_function   Time(compile)   Time(running)   Time(total)')
with open('data/training_results_impr.csv','w', newline='') as infosave:
    writer=csv.writer(infosave)
    writer.writerow(('Number of qubits:', num_qubits))
    writer.writerow(('Repetitions in the RealAmplitudes ansatz:', repetitions))
    writer.writerow(('Number of input data in each epoch:', num_data))
    writer.writerow(('Total training epochs:', total_training_epochs))
    writer.writerow(('Learning rate:', learning_rate))
    writer.writerow((""))
    writer.writerow(('Epoch', 'Cost function', 'Time(compile)', 'Time(running)', 'Time(total)'))
for epoch in range(starting_epoch, starting_epoch + total_training_epochs):
    #Loading the input training data:
    f1 = open('data/epoch_' + str(epoch) +'_training_data.pickle', "rb")
    tr_d = pickle.load(f1)
    f1.close()
    #print('training data =', tr_d)



    mapped_data, individual_shot, time_spent = mapping(param0, tr_d,
                                                       starting_index = 0, num_data = num_data,
                                                       shots = shots,
                                                       num_qubits = num_qubits, repetitions = repetitions) #mod_8d1r #mod_8d1r500s

    filesave1 = open('data/epoch_' + str(epoch) +'_mapped_data_impr.pickle', "wb") #mod_iris
    pickle.dump(mapped_data, filesave1)
    filesave1.close()

    filesave2 = open('data/epoch_' + str(epoch) +'_individual_shot_impr.pickle', "wb") #mod_iris
    pickle.dump(individual_shot, filesave2)
    filesave2.close()


    '''
    # Loading saved mapped_data:
    fileload = open('data/mapped_data_impr.pickle', "rb")
    mapped_data = pickle.load(fileload, encoding='bytes')
    fileload.close()

    fileload = open('data/individual_shot_impr.pickle', "rb")
    individual_shot = pickle.load(fileload, encoding='bytes')
    fileload.close()
    print('individual shot:', individual_shot)
    '''


    #Calculating the gradient:
    cost = cost_function(mapped_data)
    for i in range(num_qubits * (repetitions + 1)): #mod_8d1r
        gradient[i] = (cost[2 * i + 1] - cost[2 * i + 2]) / 2
        #print('gradient = ', gradient[i], 'for parameter', i)
        #print(' ')

    np.savetxt('data/epoch_' + str(epoch) +'_gradient_impr.csv', gradient, delimiter=",") #mod_iris

    #Updating the parameters: #mod_iris
    param0 -= gradient * learning_rate
    np.savetxt('data/epoch_' + str(epoch + 1) +'_parameters_impr.csv', param0, delimiter=",")

    print(epoch, '     ', round(cost[0], 11), '    ', round(time_spent[0], 1), '           ', round(time_spent[1], 1), '           ', round(time_spent[0] + time_spent[1], 1)) #mod_iris



    with open('data/training_results_impr.csv','a', newline='') as infosave:
        writer=csv.writer(infosave)
        writer.writerow((epoch, cost[0], time_spent[0], time_spent[1], time_spent[0] + time_spent[1]))

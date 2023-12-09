"""
This program computes the cost function using the improved approach.

Take 8-dimensional inputs.

Run on a quantum simulator.

Added noise to all quantum gates.
"""



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



import qiskit_aer.noise as noise #mod_noise



def build_circuit(param, starting_index = 0, num_data = 3
                  #num_qubits = 3, repetitions = 1
                  ): #mod_noise_v2
    # Amplitude encoding:
    #num_data = len(raw_data[0])

    #num_param = num_qubits * (repetitions + 1) #mod_8d1r #mod_noise_v2
    #num_bits = (num_qubits + 2 + num_param * 2) * num_data #mod_noise_sim_v3

    def FEATURE_MAP(data):
        classical_state = normalize([data])
        desired_state = classical_state[0].tolist()
        #for i in range(2 ** num_qubits - 784): desired_state.append(0.0) #mod_8d #mod_784d
        return desired_state



    # Ansatz:
    # Based on qiskit and ml53.
    #new grad2210a:

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
    #end of new grad2210a.



    # Building the quantum circuit:
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

    return circuit_qc #mod_noise_v2



def mapping(circuit_qc,
            #num_data = 3, shots = 1000,
            training = True,
            #num_qubits = 3, repetitions = 1,
            prob_1 = 0, prob_2 = 0): #mod_8d1r #mod_noise #mod_noise_v2



    # Noise model main part 1 of 2: #mod_noise

    # Error probabilities
    prob_1 = prob_1 # 1-qubit gate
    prob_2 = prob_2 # 2-qubit gate

    # Depolarizing quantum errors
    error_1 = noise.depolarizing_error(prob_1, 1)
    error_2 = noise.depolarizing_error(prob_2, 2)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3'])
    noise_model.add_all_qubit_quantum_error(error_2, ['cx']) #mod_prob_1 #mod_noise_v9

    # Get basis gates from noise model
    basis_gates = noise_model.basis_gates
    # End of noise model main part 1 of 2.



    # Using Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')



    #Computing:

    

    # Noise model main part 2 of 2: #mod_noise
    job = execute(circuit_qc, simulator,
                  basis_gates = basis_gates,
                  noise_model = noise_model,
                  shots = shots * (num_param * 2 + 1)) #mod_noise #mod_noise_sim_v3
    # End of noise model main part2 of 2.
    #job = execute(circuit_qc, simulator, shots = shots * (num_param * 2 + 1)) #mod_noise #mod_noise_sim_v3



    result = job.result()
    counts = result.get_counts(circuit_qc)

    '''
    filesave0 = open('data/job_result_impr.pickle', "wb")
    pickle.dump(result, filesave0)
    filesave0.close()
    '''

    #plot_histogram(counts)



    #Processing the quantum measurement result:
    #print('Processing the result.') #mod_noise_v3

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
    #print('Job completed.') #mod_noise_v3
    return mapped_data, individual_shot #mod_noise_v2 #mod_noise_sim_v3



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
    #num_data = len(mapped_data) #mod_noise_v2
    #num_qubits = len(mapped_data[0][1]) #mod_noise_v2
    #num_param = int((len(mapped_data[0][0]) - 1) / 2) #mod_noise_v2
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

fileload = open('data/input_data_8d.pickle', "rb") #mod_8d
raw_data = pickle.load(fileload, encoding='bytes') #mod_noise_v2
fileload.close()



# Initializing the parameters:

#param = np.random.rand(num_qubits * (repetitions + 1)) * math.pi
#np.savetxt('data/parameters.csv', param, delimiter=",") #mod_8d #mod_8d1r

param0 = genfromtxt('data/parameters.csv', delimiter=',', skip_header = 0) #mod_8d #mod_8d1r 

#gradient = np.zeros(len(param0)) #mod_noise_v2



# Running the quantum network:

# Adjustable parameters:
num_qubits = 3 #mod_8d1r #mod_784d
repetitions = 1 #mod_8d1r
shots = 1000 #mod_8d1r500s
num_data = 20 #mod_8d1r
#prob_1 = 0.001  # 1-qubit gate #mod_noise #mod_noise_v3
#prob_2 = 0.01   # 2-qubit gate #mod_noise #mod_noise_v3

num_plot = 11 # Should be a square number for 3D figures. #mod_noise_v3
#num_run = 1 #mod_noise_3 #mod_noise_v5



num_param = num_qubits * (repetitions + 1) #mod_noise_v2
num_bits = (num_qubits + 2 + num_param * 2) * num_data #mod_noise_v2 #mod_noise_sim_v3
gradient = np.zeros(num_param) #mod_noise_v2



print('Program starts at:', time.ctime())
print('Running the improved approach:\n') #mod_noise_sim_v3
print('Building the quantum circuit.')
with open('data/result_impr.csv','w', newline='') as resultsave: #mod_noise_v3
    writer=csv.writer(resultsave) #mod_noise_v3
    writer.writerow(('Program starts at:', time.ctime())) #mod_noise_v3
    writer.writerow(("")) #mod_noise_v3
    writer.writerow(('Running the improved approach:', "")) #mod_noise_v3
    writer.writerow(('num_qubits:', num_qubits)) #mod_noise_v3
    writer.writerow(('repetitions:', repetitions)) #mod_noise_v3
    writer.writerow(('shots:', shots)) #mod_noise_v3
    writer.writerow(('num_data:', num_data)) #mod_noise_v3
    writer.writerow(('num_plot:', num_plot)) #mod_noise_v3
    #writer.writerow(('num_run:', num_run)) #mod_noise_v3 #mod_noise_v5
    writer.writerow(('num_param:', num_param)) #mod_noise_v3
    writer.writerow(("")) #mod_noise_v3
time0 = time.time()
time_spent = [0, 0]

circuit_qc = build_circuit(param0, starting_index = 0,
                           num_data = num_data) #mod_noise_v2

time1 = time.time()
time_spent[0] = time1 - time0
print('Time spent on building the circuit:', round(time_spent[0], 1), 'seconds.\n')
with open('data/result_impr.csv','a', newline='') as resultsave: #mod_noise_v3
    writer=csv.writer(resultsave) #mod_noise_v3
    writer.writerow(('Time spent on building the circuit (seconds):', time_spent[0])) #mod_noise_v3
    writer.writerow(("")) #mod_noise_v3



print('Computing.')

for j in range(num_plot): #mod_noise_v3
    #prob_1 = 0.001 * (j % int(math.sqrt(num_plot))) #For 3D figures. #mod_noise_v3
    #prob_2 = 0.01 * int(j / int(math.sqrt(num_plot))) #For 3D figures. #mod_noise_v3
    prob_1 = 0.001 * j #For 2D figures. #mod_noise_v3 #mod_noise_v9
    prob_2 = prob_1 #For 2D figures. #mod_noise_v3 #mod_noise_v9
    samples = np.zeros(num_param + 1) #mod_noise_v3 #mod_noise_v5

    mapped_data, individual_shot = mapping(circuit_qc,
                                           prob_1 = prob_1, prob_2 = prob_2) #mod_8d1r #mod_8d1r500s #mod_noise #mod_noise_v2 #mod_noise_sim_v3

    '''
    filesave1 = open('data/mapped_data_impr.pickle', "wb")
    pickle.dump(mapped_data, filesave1)
    filesave1.close()

    filesave2 = open('data/individual_shot_impr.pickle', "wb")
    pickle.dump(individual_shot, filesave2)
    filesave2.close()
    '''

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
    #for i in range(num_qubits * (repetitions + 1)): #mod_8d1r #mod_noise_v2
    for i in range(num_param): #mod_8d1r #mod_noise_v2
        gradient[i] = (cost[2 * i + 1] - cost[2 * i + 2]) / 2
        #print('gradient = ', gradient[i], 'for parameter', i)
        #print(' ')

    #np.savetxt('data/gradient_impr.csv', gradient, delimiter=",") #mod_noise_v3
    #np.savetxt('data/cost_impr.csv', cost, delimiter=",") #mod_784d_v9 #mod_noise_v9

    print('prob_1 =', prob_1, ' prob_2 =', prob_2) #mod_noise_v9
    print('gradient =', gradient) #mod_noise_v9
    with open('data/result_impr.csv','a', newline='') as resultsave: #mod_noise_v3
        writer=csv.writer(resultsave) #mod_noise_v3
        writer.writerow(('prob_1:', prob_1)) #mod_noise_v9
        writer.writerow(('prob_2:', prob_2)) #mod_noise_v9
        writer.writerow(('gradient:', "")) #mod_noise_v9
        writer.writerow((gradient)) #mod_noise_v9
        writer.writerow(('cost:', "")) #mod_noise_v9
        writer.writerow((cost)) #mod_noise_v9
        writer.writerow(("")) #mod_noise_v9



time2 = time.time()
time_spent[1] = time2 - time1
print('\nTime spent on computing:', round(time_spent[1], 1), 'seconds.\n') #mod_noise_v5
with open('data/result_impr.csv','a', newline='') as resultsave: #mod_noise_v3
    writer=csv.writer(resultsave) #mod_noise_v3
    #writer.writerow(("")) #mod_noise_v3 #mod_noise_v9
    writer.writerow(('Time spent on computing (seconds):', time_spent[1])) #mod_noise_v3

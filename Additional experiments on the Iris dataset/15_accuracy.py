"""
For computing the classification via the trained parameters.

Input files:
data/epoch_0_parameters.csv
data/epoch_i_parameters_impr.csv (i=1,...,total_training_epochs)
data/epoch_i_parameters_conv.csv (i=1,...,total_training_epochs)

Output file:
data/accuracy.csv
"""



from sklearn.datasets import load_iris #mod_iris
ds = load_iris() #mod_iris

import copy #mod_784d_v9

import numpy as np
from qiskit import(QuantumCircuit) #mod_th
from qiskit_aer import AerSimulator #mod_th
from qiskit.visualization import plot_histogram

import csv #new grad2210a

from sklearn.preprocessing import normalize
import math

from numpy import genfromtxt



# Using Aer's qasm_simulator
simulator = AerSimulator() #mod_th



def ANSATZ(param):
    # Amplitude encoding:
    #num_param = len(param)
    #num_bits = num_qubits * (num_param * 2 + 1) * num_data
    num_bits = num_qubits



    # The RealAmplitudes ansatz:
    def real_amp(param):
        varform = QuantumCircuit(num_qubits, 0)
        for i in range(num_qubits): varform.ry(param[i], i)
        for j in range(repetitions):
            for m in range(num_qubits-1):
                for n in range(m+1, num_qubits): varform.cx(m, n) # full entanglement
                #varform.cx(m, m+1) # circular and linear entanglement #mod_v1
            #varform.cx(num_qubits-1, 0) # circular entanglement #mod_v1
            #varform.barrier(range(num_qubits))
            for i in range(num_qubits): varform.ry(param[(j + 1) * num_qubits + i], i)
        return varform



    # Building the quantum circuit:
    circuit_qc = QuantumCircuit(num_qubits, num_bits)
    circuit_qc = circuit_qc.compose(real_amp(param))
    #circuit_qc.draw('mpl',scale = 0.3,style={'fontsize': 8})



    #Computing:
    circuit_qc.save_unitary() #mod_unitary_v1
    result = simulator.run(circuit_qc).result() #mod_vector_v1
    unitary = result.get_unitary(circuit_qc) #mod_unitary_v1
    ut=np.asarray(unitary).real

    '''
    filesave0 = open('data/job_result_conv.pickle', "wb")
    pickle.dump(result, filesave0)
    filesave0.close()
    '''

    return ut



def ACC(approach):
    accuracy = np.zeros(total_training_epochs + 1)
    max_acc = 0
    max_epoch = 0
    for epoch in range(starting_epoch, starting_epoch + total_training_epochs + 1):
        #Loading the parameters:
        if epoch == 0:
            param = genfromtxt('data/epoch_0_parameters.csv', delimiter=',', skip_header = 0) #mod_8d #mod_8d1r 
        else:
            param = genfromtxt('data/epoch_' + str(epoch) + '_parameters_' + approach + '.csv', delimiter=',', skip_header = 0) #mod_8d #mod_8d1r 
        
        basic_amplitudes = np.zeros((dim, 2 ** num_qubits)) # Can be placed either before or behind "for j in range(0, num_cost_fn)".
        deduced_amplitudes = np.zeros(2 ** num_qubits) # Can be placed either before or behind "for j in range(0, num_cost_fn)".

        # Computing the basic amplitudes of the output states for the basis vectors:
        ut = ANSATZ(param) #mod_unitary_v1
        for i in range(dim):
            basic_amplitudes[i] = ut @ basic_data[i] #mod_unitary_v1



        # Computing the accuracy:
        #random.shuffle(tr_d_random)
        va_d_inputs = [(tr_d_random[k][0]) for k in range(num_validation_data)]
        va_d_results = [(tr_d_random[k][1]) for k in range(num_validation_data)]
        va_d = [va_d_inputs, va_d_results]
        '''
        f1 = open('data/epoch_' + str(epoch) +'_evaluation_data.pickle', "wb")
        pickle.dump(va_d, f1)
        f1.close()
        '''
        for k in range(num_validation_data):
            #Method 1': (fastest. 2.7 seconds for num_data=1000, 14.9 seconds for num_data=45000)
            input_state = normalize([va_d[0][k]])
            deduced_amplitudes = input_state[0] @ basic_amplitudes

            # Converting the deduced amplitudes to the probabilities of each qubit:
            probabilities = np.zeros(num_qubits) # IMPORTANT MOD
            for n in range(num_qubits):
                for m in keys[n]:
                    probabilities[n] += deduced_amplitudes[m] ** 2 # IMPORTANT MOD

            # Computing the label:
            label_results = np.argmax(probabilities)
            
            if label_results == va_d[1][k]: accuracy[epoch] += 1
        accuracy[epoch] /= num_validation_data



        #Recording the max accuracy and the corresponding epoch index:
        if accuracy[epoch] > max_acc:
            max_acc = accuracy[epoch]
            max_epoch = epoch

    return accuracy, max_acc, max_epoch



# MAIN PROGRAM:
import pickle
import gzip
import time

import random



# Global adjustable parameters:
num_qubits = 3
repetitions = 4
num_data = 40 # Number of input data in each epoch.
starting_index = 0
dim = 4 # Dimension of the input data set. Must not exceed (2 ** num_qubits). For MNIST it is 784.

total_training_epochs = 50 # ENTER YOUR CHOICE HERE.
learning_rate = 2 # ENTER YOUR CHOICE HERE.

num_validation_data = 150 # Enter: 1~150

#starting_epoch = int(genfromtxt('data/starting_epoch.csv', delimiter=',', skip_header = 0))
starting_epoch = 0

num_param = num_qubits * (repetitions + 1)



# Creating the basic input data:
basic_data = np.zeros((dim, 2 ** num_qubits))
for i in range(dim):
    basic_data[i][i] = 1
    #basic_data[i][0] = 1 # for superposition (unnormalized)

# Prepare the keys for which the measurement result of the nth qubit is 1.
keys = []
for n in range(num_qubits):
    key = []
    for m in range(2 ** num_qubits):
        x = int(m / (2 ** n))
        if x % 2 == 1: key.append(m)
    keys.append(key)

# Loading the real input data:
tr_d_random = list(zip(ds.data, ds.target)) #mod_iris



#Computing the accuracy of the improved approach:
accuracy_impr, max_impr, epoch_impr = ACC('impr')
print('The improved approach:')
print('Accuracy on evaluation data =', accuracy_impr)
print('Max accuracy is', max_impr, 'at epoch', epoch_impr, '.\n')



#Computing the accuracy of the conventional approach:
accuracy_conv, max_conv, epoch_conv = ACC('conv')
print('The conventional approach:')
print('Accuracy on evaluation data =', accuracy_conv)
print('Max accuracy is', max_conv, 'at epoch', epoch_conv, '.\n')



#Saving the results:
with open('data/accuracy.csv','w', newline='') as infosave:
    writer=csv.writer(infosave)
    writer.writerow(('Number of qubits:', num_qubits))
    writer.writerow(('Repetitions in the RealAmplitudes ansatz:', repetitions))
    writer.writerow(('Number of validation data:', num_validation_data))
    writer.writerow((""))
    writer.writerow(('Epoch', 'Accuracy(conventional)', 'Accuracy(improved)'))

    for epoch in range(starting_epoch, starting_epoch + total_training_epochs + 1):
        writer.writerow((epoch, accuracy_conv[epoch], accuracy_impr[epoch]))

    writer.writerow((""))
    writer.writerow(('Max_accuracy', max_conv, max_impr))
    writer.writerow(('epoch', epoch_conv, epoch_impr))

print('Results saved to file data/accuracy.csv.')

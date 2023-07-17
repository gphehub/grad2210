#This program loads "individual_shot_impr.pickle" and converts it into "individual_shot_impr.csv".



import csv
import pickle

fileload = open('data/individual_shot_impr.pickle', "rb")
individual_shot = pickle.load(fileload, encoding='bytes')
fileload.close()

print('individual shot:', individual_shot)
for i in range(len(individual_shot)):
    print('sum(individual_shot[', i, ']):', sum(individual_shot[i]))

with open('data/individual_shot_impr.csv','w', newline='') as resultsave: #mod_3a
    writer=csv.writer(resultsave) #mod_3a
    for i in range(len(individual_shot)):
        writer.writerow((individual_shot[i])) #mod_3a


# -*- coding: utf-8 -*-
"""
@author: HG
this code finds the best three bands for calculating RABD for chlorophyll estimation using Genetic algorithm. The inputs here are hyperspectral images and measurements from HPLC.
"""

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

def cal_pop_fitness(data, pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = np.zeros(pop.shape[0]) 
    plsr = {} 
    numrnd = 1500 
    hplc = [0.0811901107704823,0.115709043434053,0.0358641256625103,0.0329213242187132,0.0557344902275296,0.250502777130247,0.163129856150001,0,0.0154314117999148,0.0283812088976221,0.0935974320003221,0.118978363582853,0.0179387107138366,0.0901754592272664,0.0452626777389289,0.136747982524975,0.203058187825657,0.373535499198235,0.0930527914060211,0.0309398988415928,0.0449425849858510,0.00671651686967625,0.00253567084693941,0.0105687702125211]
    rows = np.empty((np.size(hplc), numrnd),'int')
    cols = np.empty((np.size(hplc), numrnd),'int')
    for item in range(pop.shape[0]):
        RABD_numerator = ( (pop[item,1]-pop[item,2])*(data[:,pop[item,0]] + data[:,pop[item,0]+1])/2  + (pop[item,2]-pop[item,0])*(data[:,pop[item,1]]+data[:,pop[item,1]+1])/2 )/(pop[item,1]-pop[item,0])
#        RABD_denominator = np.minimum((data[:,pop[item,2]]+data[:,pop[item,2]+1])/2 , (data[:,pop[item,3]]+data[:,pop[item,3]+1])/2)
        RABD_denominator = (data[:,pop[item,2]]+data[:,pop[item,2]+1])/2 
        RABD = RABD_numerator / RABD_denominator
        RABD = 1/np.reshape(RABD , (r,c))
        RABDSum = np.mean(RABD, 1)
        hsi = RABDSum[65:8545]#Y lake
        RABD = RABD[65:8545 , :]
#        hsi = RABDSum[60:6305]#2fB lake
#        RABD = RABD[60:6305 , :]
        ranges = np.linspace(37, hsi.shape[0]-37 , np.size(hplc) ,dtype = 'int')

        sumN = np.zeros((np.size(hplc) , 1))
        for counters in range(np.size(hplc)):
#            if item ==0:                    
#                rows[counters,:] = np.random.randint(low=ranges[counters]-37, high=ranges[counters]+37 , size=numrnd)
#                cols[counters,:] = np.random.randint(low=0, high=RABD.shape[1] , size=numrnd) 
#            sumN[counters] = np.mean(RABD[rows[counters,:], cols[counters,:]])
            sumN[counters] = np.mean(hsi[ranges[counters]-37 :ranges[counters]+37 ])
        hplc = np.array(hplc)
        X_train,  X_test, Y_train, Y_test =train_test_split(np.reshape(np.array(range(0,np.size(hplc))), (-1,1)), np.reshape(hplc, (-1,1)), train_size = 0.7)
#        pls = PLSRegression(n_components=1)
        pls = RandomForestRegressor()
        pls.fit(sumN[X_train.flatten()] , Y_train.flatten())
        Y_pred = pls.predict(sumN[X_test.flatten()])
        plsr[item] = pls
        fitness[item] = np.corrcoef(Y_pred.flatten(), Y_test.flatten())[0][1]
    return fitness, plsr

def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    fitness[np.where(np.isnan(fitness))] = 0
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))          
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = np.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
#    mutations_counter = np.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for idx in range(offspring_crossover.shape[0]):
#        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # The random value to be added to the gene.
            gene_idx = int(np.round(np.random.uniform(0, offspring_crossover.shape[1]-1, 1)))
            random_value = np.random.uniform(-1.0, 1.0, 1)
            prob = np.random.uniform(0, 1)
            if prob>.60:
                if gene_idx==0:
                    random_value = np.random.randint(low=221, high=300, size=1)
                    offspring_crossover[idx, gene_idx] = random_value 
#                    offspring_crossover[idx, gene_idx] += int(np.round(np.random.uniform(-10, 10, 1)))
                elif gene_idx==1:
                    random_value = np.random.randint(low=390, high=450, size=1)
                    offspring_crossover[idx, gene_idx] = random_value
#                     offspring_crossover[idx, gene_idx] += int(np.round(np.random.uniform(-10, 10, 1)))

                else:
                    random_value = np.random.randint(low=330, high=380, size=1)
                    offspring_crossover[idx, gene_idx] = random_value
#                     offspring_crossover[idx, gene_idx] += int(np.round(np.random.uniform(-10, 10, 1)))
                    
#            gene_idx = gene_idx + mutations_counter
    return offspring_crossover

import spectral.io.envi as envi
#filename =r'F:\hamid\yohanna data analyses\2FB-CORE-1-DATA\2FBtest.hdr'
filename = r'F:\hamid\yohanna data analyses\Y-Lake-Core3-DATA\Results\Test\Test.hdr'
data = envi.open(filename)
hdr = envi.read_envi_header(filename)
data = data.load()
[r,c,b] = data.shape
data = data.reshape(r* c , b)
#data = savgol_filter(data, 5, 2)

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
sol_per_pop = 16
num_parents_mating = 4
variables = 3
# Defining the population size.
pop_size = (sol_per_pop,variables) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.
#Creating the initial population.
new_population = np.zeros(pop_size, 'int')
new_population[0, :] = [258, 436, 348]
new_population[1, :] = [250, 420, 340]
new_population[2, :] = [260, 410, 360]
new_population[3, :] = [270, 415, 340]
new_population[4, :] = [280, 425, 342]
new_population[5, :] = [290, 390, 368]
new_population[6, :] = [295, 395, 348]
new_population[7, :] = [245, 440, 360]
for i in range(8,16):
    new_population[i, :] = [np.random.randint(low=221, high=300, size=1), np.random.randint(low=390, high=450, size=1), np.random.randint(low=330, high=380, size=1)]

#new_population = np.random.randint(low=250, high=500, size=pop_size)
best_outputs = []
num_generations = 200
for generation in range(num_generations):
    # Measuring the fitness of each chromosome in the population.
    print(generation)
    
    fitness , plsr = cal_pop_fitness(data, new_population)

    # Selecting the best parents in the population for mating.
    parents = select_mating_pool(new_population, fitness, 
                                      num_parents_mating)
    print("Parents")
    print(parents)

    # Generating next generation using crossover.
    offspring_crossover = crossover(parents,
                                       offspring_size=(pop_size[0]-parents.shape[0], pop_size[1]))
    print("Crossover")
    print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    offspring_mutation = mutation(offspring_crossover, num_mutations=1)
    print("Mutation")
    print(offspring_mutation)

    # Creating the new population based on the parents and offspring.
    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation
    
# Getting the best solution after iterating finishing all generations.
#At first, the fitness is calculated for each solution in the final generation.
#fitness , pls = cal_pop_fitness(data, new_population)
# Then return the index of that solution corresponding to the best fitness.
fitness[np.where(np.isnan(fitness))] = 0
best_match_idx = np.where(fitness == np.max(fitness))
best_model = plsr[(best_match_idx[0][0])]
pickle.dump(best_model , open('best_model_RF.sav', 'wb'))
print("Best solution : ", new_population[best_match_idx, :])
print("Best solution fitness : ", fitness[best_match_idx])
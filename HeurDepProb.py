# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 10:45:10 2021

@author: Lenard
"""
# Importing time to output time taken.
import time
# Importing numpy to add support for large, multi-dimensional arrays and matrices.
import numpy as np
# Importing the data from an Excel sheet using pandas.
import pandas as pd
# Importing operator for dictionary use.
import operator


# Naming the imported data frame as 'df'.
df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range1.xlsx')
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range2.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range3.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range4.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range5.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range6.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range7.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range8.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range9.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range10.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range11.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range12.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range13.xlsx')   
#df = pd.read_excel (r'C:\Users\Lenard\Desktop\Range14.xlsx')           
# Showing the imported data frame, df.
print (df)

# The length of our data frame is used as a set to iterate from. 
I = range(30)

# Defining the variables for each column of df.
h= df["h"]
et=df["et"]
lt=df["lt"]
a=df["I.D, i"]
w=df["w"]
s=df["s"]
r=df["r"]
# Creating an empty data structure for airc.
airc={}

# Creating a class, Aircraft.
class Aircraft:
    
    # Defining the attributes of the class.
    def __init__(self,id, Weight, Speed, sid,h,et,lt):
       self.id = id
       self.Weight = Weight
       self.Speed = Speed
       self.sid = sid
       self.h = h
       self.et = et
       self.lt = lt
    
    # Defining a function which returns wake vortex minimal separation
    # (in minutes) between aircraft i and j.
    def vortexfun(airc_1, airc_2):
        weights = 'LMH'
        if (weights.index(airc_1.Weight) > weights.index(airc_2.Weight)):
            return 2
        else:
            return 1
    
    # Defining a function which returns route and speed
    # minimal separation (in  minutes) between aircraft i and j.      
    def routespeedfun(airc_1, airc_2):
        speeds = 'ABCD'
        if speeds.index(airc_1.Speed) < speeds.index(airc_2.Speed):
            if airc_1.sid == airc_2.sid:
                return 3
            else:
                return 2
        else:
            if airc_1.sid == airc_2.sid:
                return 2       
            else:
                return 1
   
    # Defining a function which returns the minimum
    # required separation (in minutes) between aircraft i and j.
    def maxsep(airc_1,airc_2):
        return max(Aircraft.vortexfun(airc_1,airc_2),Aircraft.routespeedfun(airc_1,airc_2)) 

for i in I:
    airc[i]=Aircraft(a[i],w[i],s[i],r[i],h[i],et[i],lt[i]) 

# Creating an empty data structure for v.
v = {}
# Making the traversal time through the holding area equal to zero
# for all i.
for i in I:
    v[i]=0
print(v)
# Creating a len(df)*len(df) empty matrix for S_ij.    
S = np.empty((len(df), len(df)),int)
for i in I:
    for j in I:
        # Inserting the values of the maxsep function in the matrix.
        if (j == i): 
            S[i][j] = 0
        else:
            S[i][j]=Aircraft.maxsep(airc[i],airc[j])
        
# Giving values to the parameters in the objective function.
F_E= 5
F_L= 5
W1=50
W2=2
W3=100
W4=100
W5=1000
W6=1000000

num_air = len(I) # number of aircraft
    
# Parameters obtained by parameter tuning.
# Population and half population (in case odd).
halfpopsize = 15
population_size= 30
# Crossover Rate can be anywhere in [0,1] but preferably
# between 0.5 and 0.9
crossover_rate= 0.9
# Mutation Rate can be anywhere in [0,1] but preferably
# between 0.0001 and 0.3
mutation_rate= 0.2
# Number of Generations 
num_iteration= 2000 

# Creating an empty Dictionary 
fit_dict = {}

## Code for shuffling restriction ## 
# def geninit_pop():
#     population_list =[]
#     d=8
#     rangair = np.arange(1,num_air+1)
#     rangairlist = rangair.tolist()
#     old = {e:i for i,e in enumerate(rangairlist)}
#     for i in range(population_size):
#         rangair = np.arange(1,num_air+1)
#         np.random.shuffle(rangair)
#         individ = rangair.tolist()
#         new = {e:i for i,e in enumerate(individ)}
#         valid = all(abs(i-new[e])<=d for e,i in old.items())
#         while individ in population_list or valid == False:
#             np.random.shuffle(rangair)
#             individ = rangair.tolist()
#             new = {e:i for i,e in enumerate(individ)}
#             valid = all(abs(i-new[e])<=d for e,i in old.items())
#         population_list.append(individ)
#     return population_list


### GENERATING INITIAL POPULATION ###
def geninit_pop():
    # Creating an empty population list
    population_list =[]
    # Creating the basic individual from 1 to num_air exactly
    rangair = np.arange(1,num_air+1)
    rangairlist = rangair.tolist()
    # Shuffling the created individual to produce
    # other random individuals.
    for i in range(population_size):
        rangair = np.arange(1,num_air+1)
        np.random.shuffle(rangair)
        individ = rangair.tolist()
        # Checking whether created individual is in population
        # list, if it is omit and search for another one.
        while individ in population_list:
                np.random.shuffle(rangair)
                individ = rangair.tolist()
        population_list.append(individ)
    # Generating the fitness of the population found and
    # saving it in the created dictionary.
    mksp = genfit_pop(population_list)
    for i in range(population_size):
        fit_dict[str(population_list[i])]= 1/mksp[i]
    return population_list



### Time Window Compliance ###
def Compliance(t,et,lt):
    if t < et + F_E:
        return W3*(et + F_E - t)
    elif t > lt - F_L and t <= lt:
        return W4*(F_L + t - lt)
    elif t > lt:
        return W5*(t-lt)+W6
    else:
        return 0
    
### FITNESS FUNCTION (for individual) ###
def genfit_ind(individual):
    # Creating empty lists for variables used.
    t= np.empty(num_air)
    e= np.empty(num_air)
    defaultarray = np.arange(1,num_air+1)
    z= np.empty(num_air)
    # Creating list to put fitness of each aircracft/gene.
    fitperair_list = []
    ## Finding the objective function value.
    for i in range(num_air):
        if (i==0):
            t[i]= max(et[individual[i]-1],h[individual[i]-1])
            e[i]= et[individual[i]-1]
        else:
            t[i] = max(et[individual[i]-1],h[individual[i]-1], t[i-1]+S[individual[i-1]-1][individual[i]-1])
            e[i] = t[i-1]+S[individual[i-1]-1][individual[i]-1]
        z[i]= max(0,defaultarray[i]- a[individual[i]-1])*max(0,defaultarray[i]- a[individual[i]-1])
        fitperair= W1*(t[i]-h[individual[i]-1]) + W2*z[i] + Compliance(t[i], et[individual[i]-1], lt[individual[i]-1])
        fitperair_list.append(fitperair)
    # Fitness of an individual.    
    fitindivid = sum(fitperair_list)
    # Uncomment these to display them for each aircraft.
    #print(t)
    #print(e)
    #print(et)
    return fitindivid

### FITNESS FUNCTION (for population) ###
def genfit_pop(population):
    fitness_list = []
    makespanrecip_list = []
    for ind in range(population_size):
        fitness = genfit_ind(population[ind])
        fitness_list.append(fitness)
        makespanrecip_list.append(1/fitness)
    #print(min(fitness_list))    
    return makespanrecip_list


### ROULETTE WHEEL SELECTION ###
# First Part.
# Generating the probabilities according to fitness.
def cumulated(population):
    pk = []
    qk = []
    mksp = genfit_pop(population)
    for i in range(population_size):
        if str(population[i]) not in fit_dict:
            fit_dict[str(population[i])]= 1/mksp[i] 
    total_fitness = sum(mksp)
    for i in range(population_size):
        pk.append(mksp[i]/total_fitness)
    for i in range(population_size):
        cumulative=0
        for j in range(0,i+1):
            cumulative=cumulative+pk[j]
        qk.append(cumulative)
    qk[-1] = 1
    return qk

# Second Part.
# Selecting the parents. (Giving their indices)
def selection(population):
    qk = cumulated(population)
    indices = [0,0]
    while indices[0] == indices[1]:
        selection_rand = np.random.rand(2).tolist()
        indices.clear()
        for number in selection_rand:
            if number <= qk[0]:
                index = 0
            else:
                for j in range(0,population_size-1):
                    if number > qk[j] and number <= qk[j+1]:
                        index = j+1
                        break   
            indices.append(index)  
    return indices


### PMX-CROSSOVER ###
# Generating the two children out of the chosen parents.
def childgen(parent1,parent2):
    parent_list=[]    
    parent_list.append(parent1)
    parent_list.append(parent2)
    child_1 = parent_list[0].copy()
    child_2 = parent_list[1].copy()
    cutpoint = [0,num_air-1]
    while cutpoint == [0,num_air-1]:
        cutpoint = list(np.random.choice(num_air, 2, replace=False))
        cutpoint.sort()
    mapping_list = []
    for i in range(cutpoint[0],cutpoint[1]+1):
        mapping_list.append([parent_list[0][i],parent_list[1][i]])
    
    child_1[cutpoint[0]:cutpoint[1]+1] = parent_list[1][cutpoint[0]:cutpoint[1]+1]
    child_2[cutpoint[0]:cutpoint[1]+1] = parent_list[0][cutpoint[0]:cutpoint[1]+1]
    
    for j in range(num_air):
        if j not in range(cutpoint[0],cutpoint[1]+1):
            count1 = child_1.count(child_1[j])
            while count1 > 1:
                for mp in mapping_list:
                    if mp[1] == child_1[j]:
                        child_1[j]= mp[0]
                        break
                count1 = child_1.count(child_1[j])
            count2 = child_2.count(child_2[j])
            while count2 > 1:
                for mp in mapping_list:
                    if mp[0] == child_2[j]:
                        child_2[j]= mp[1]
                        break
                count2 = child_2.count(child_2[j])
    return child_1 , child_2

## Best Two Selection method
def bestwo(parent1,parent2,child1,child2):
    parent1fit=genfit_ind(parent1)
    parent2fit =genfit_ind(parent2)
    child1fit= genfit_ind(child1)
    child2fit= genfit_ind(child2)
    #list of fits
    allfits = [parent1fit,parent2fit,child1fit,child2fit]
    bestofall = min(allfits)
    allfits.remove(bestofall)
    secondbest = min(allfits)
    if bestofall == parent1fit:
        if secondbest == parent2fit:
            return parent1, parent2
        elif secondbest == child1fit:
            return parent1, child1
        else:
            return parent1, child2
    elif bestofall == parent2fit:
        if secondbest == parent1fit:
            return parent1, parent2
        elif secondbest == child1fit:
            return parent2, child1
        else:
            return parent2, child2
    elif bestofall == child1fit:
        if secondbest == parent1fit:
            return parent1, child1
        elif secondbest == parent2fit:
            return parent2, child1
        else:
            return child1, child2
    else:
        if secondbest == parent1fit:
            return parent1, child2
        elif secondbest == parent2fit:
            return parent2, child2
        else:
            return child1, child2        



# KBR Selection Method
def keepbest(parent1,parent2,child1,child2):
    parent1fit=genfit_ind(parent1)
    parent2fit =genfit_ind(parent2)
    child1fit= genfit_ind(child1)
    child2fit= genfit_ind(child2)
    bestparentfit= min(parent1fit,parent2fit)
    bestchildfit = min(child1fit,child2fit)
    worstchildfit = max(child1fit,child2fit)
    #return besttwo[0], besttwo[1]
    if bestparentfit >= worstchildfit:
        return child1, child2
    else:
        if bestchildfit == child1fit:
           if bestparentfit == parent1fit:
               return child1, parent1
           else:
               return child1, parent2
        else:
           if bestparentfit == parent1fit:
               return child2, parent1
           else:
               return child2, parent2
    
### MUTATION ###
# Swap Mutation
def swap(individual):
    bothind = list(np.random.choice(num_air, 2, replace=False))
    tempno = individual[bothind[0]]
    individual[bothind[0]]= individual[bothind[1]]
    individual[bothind[1]] = tempno
      

# Generating the new population based on the old population
# depending on the chosen parameters.
def nextpop(oldpopulation):
    newpop = []
    for i in range(halfpopsize):
        parent1index, parent2index = selection(oldpopulation)
        parent1 = oldpopulation[parent1index]
        parent2 = oldpopulation[parent2index]
        if crossover_rate >= np.random.rand():
            child1,child2 = childgen(parent1, parent2)
            # Using KBR 
            ind1,ind2 = keepbest(parent1, parent2, child1, child2)
            ### STDS ### 
            #ind1,ind2 = child1,child2
            # Using Best Two
            #ind1, ind2 = bestwo(parent1, parent2, child1, child2)
            newpop.append(ind1)
            newpop.append(ind2)
            if str(ind1) not in fit_dict:
                fit_dict[str(ind1)]=genfit_ind(ind1)
            if str(ind2) not in fit_dict:
                fit_dict[str(ind2)]= genfit_ind(ind2)
        else:
            newpop.append(parent1)
            newpop.append(parent2)       
    for i in range(population_size):        
        if mutation_rate >= np.random.rand():
            swap(newpop[i])
            if str(newpop[i]) not in fit_dict:
                fit_dict[str(newpop[i])]= genfit_ind(newpop[i])
    return newpop
            
# Generating generation after generation,
# with time taken outpuuted also.
def generationgen(tstart):
    oldpopulation = geninit_pop()
    for i in range(num_iteration):
        # Printing iteration number.
        #print("iteration = ", i+1)
        # Printing best fitness value for each iteration.
        sortdict=sorted(fit_dict.items(), key = operator.itemgetter(1), reverse = False)
        bestfitall = sortdict[0][1]
        print(time.time()-tstart, bestfitall)
        newpop= nextpop(oldpopulation)
        oldpopulation = newpop.copy()

# Runs the GA.         
def GA():
    fit_dict.clear()
    tstart= time.time()
    generationgen(tstart)
    tend= time.time()
    print("time taken", tend-tstart)
    sortdict=sorted(fit_dict.items(), key = operator.itemgetter(1), reverse = False)
    bestfitall = sortdict[0][1]
    for item in sortdict:
        if item[1]==bestfitall:
            print(item)
        else:
            break
       
# Runs the GA for 10 times.
def multiruns():
    for i in range(1,11):
        print("iteration number =",i)
        GA()

      
        
        
        
        
        
        
        
        
        
        
    
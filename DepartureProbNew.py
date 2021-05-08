# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 10:30:26 2021

@author: Lenard
"""

# Importing the gurobi interface to solve a model.
import gurobipy as gp
from gurobipy import GRB
# Importing numpy to add support for large, multi-dimensional arrays and matrices.
import numpy as np
# Importing the data from an Excel sheet using pandas.
import pandas as pd

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
h=df["h"]
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

# Creating empty data structures for variables in the model.
c={}
t={}
e={}
z={}
ca={}
Cst={}
g={}
emet={}
d={}
emax_list=[] 

# Naming the gurobi model as m.
m = gp.Model("DepProb")
# Setting the running time to a maximum of 6 hours, 360 minutes x 60 seconds.
m.setParam('TimeLimit',360*60)
# Setting the model to stop only if gap is 0% before 6 hours.
m.setParam('MIPGap', 0)

# Creating decision variables.
for i in I:
    c[i] = m.addVar(vtype=GRB.INTEGER,lb=1,ub =30, name="c"+ str(i+1)) 
    t[i] = m.addVar(vtype=GRB.INTEGER, name="t"+ str(i+1))
    e[i] = m.addVar(vtype=GRB.INTEGER, name="e" + str(i+1))         
    z[i] = m.addVar(vtype=GRB.INTEGER, lb=0,ub=29, name="z"+ str(i+1))
    ca[i] = m.addVar(vtype=GRB.INTEGER, lb=-29,ub=29, name="ca"+ str(i+1))  
    Cst[i]= m.addVar(vtype=GRB.INTEGER, name="Cst"+str(i+1))        
    g[i]= m.addVar(vtype=GRB.BINARY, name= "gamma" +str(i+1))    
    emet[i]= m.addVar(vtype=GRB.INTEGER, name="e_minus_et"+str(i+1))   
    for j in I:
        if (j!=i):
            emax_list.append((i,j))                                     
            d[i,j]= m.addVar(vtype=GRB.BINARY, name="delta"+str(i+1)+str(j+1))
emax = m.addVars(emax_list, vtype=GRB.INTEGER, name="emax")                  

# Setting objective
m.setObjective(gp.quicksum(W1*(t[i]-h[i])+W2*(z[i]*z[i])+Cst[i] for i in I), GRB.MINIMIZE)

# Defining constraints for the model.
for i in I:
    #### FIRST-COME FIRST-SERVE (FCFS) BASIS CONSTRAINT ####
    #m.addConstr(c[i]==a[i])
    
    # For second part of objective, reordering delay.
    m.addConstr(ca[i]==c[i]-a[i])
    
    # max(0,c[i]-a[i])
    m.addGenConstrMax(z[i],[0,ca[i]])
    
    # For predicted take-off time,t
    # max(et[i], h[i]+v[i],e[i])
    m.addGenConstrMax(t[i], [et[i],h[i]+v[i],e[i]])
    
    # Creating a piecewise linear function for cost function.
    xpts=[et[i],et[i]+F_E,lt[i]-F_L,lt[i],lt[i],lt[i]+1]
    ypts=[W3*F_E,0,0,W4*F_L,W6,W5+W6]
    m.addGenConstrPWL(t[i],Cst[i], xpts, ypts,"Cost")   
            
   
    # For Earliest tot, e
    # c[i]<=1 => g[i]=1, or equivalently, g[i]=0 => c[i]>=2
    m.addGenConstrIndicator(g[i],False,c[i]>=2)     
    # g[i]=1 => c[i]<=1 
    m.addGenConstrIndicator(g[i],True,c[i]<=1)   
    for j in I:
        if (j!=i):
            #c[j]-c[i]<0 => d[j,i]=1, or equivalently, d[j,i]=0 => c[j]-c[i]>0
            # Since cannot be the same
            m.addGenConstrIndicator(d[j,i],False,c[j]-c[i]>=1)
            #d[j,i]=1 => c[j]-c[i]<0 or equivalently, d[j,i]=1 => c[j]-c[i]<=-1
            m.addGenConstrIndicator(d[j,i],True,c[j]-c[i]<=-1)
            # emax[j,i] = (t[j]+S[j][i])d[j,i]
            m.addConstr(emax[j,i]==(t[j]+S[j][i])*d[j,i])
    
    # emet[i] = e[i] - et[i]*g[i]
    m.addConstr(emet[i] == e[i] - et[i]*g[i])
    # emet[i] = max(emax[j,i] for all j)
    m.addConstr(emet[i] == gp.max_(emax.select('*', i)))
     
# Optimize model
m.optimize()

# Printing Objective Function Value.       
print('Obj: %g' % m.objVal)
# Procedure to print decision variables and other variables.
outc=[]
oute=[]
outt=[]
outhv=[]
for i in I:
     outc.append(m.getVarByName("c"+str(i+1)).x)
     oute.append(round(m.getVarByName("e"+str(i+1)).x,1))
     outt.append(round(m.getVarByName("t"+str(i+1)).x,1))
     outhv.append(round(h[i]+v[i],1))
     
print("c  =", outc)
print("et =", list(round(et[I],1)))
print("hv =", outhv)
print("e  =", oute)
print("t  =",outt)
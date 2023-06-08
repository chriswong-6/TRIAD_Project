"""
Created on Friday August 13 16:01:52 2021

@author: vladimir (http://www.pitt.edu/~viz/)

Matrix driven implementation of TM, based on TM intro in 

K. D. Abeyrathna, Ole-Christoffer Granmo, Morten Goodwin
Extending the Tsetlin Machine With Integer-Weighted Clauses 
for Increased Interpretability. IEEE Access, 2021

"""
from symbol import classdef
import numpy as np
import copy
def genXORInput():
#
#Generate a XOR binary input vector for TM
#  
    x=np.random.randint(2, size=2) #generate values for two input variables 
    invector = np.append(x,1-x) #adding negation of variable values to form complete list of literals
    output = x[0] ^ x[1] #xor output variable;
    return [invector, output]


def feedbackMatrices(stateMatrix, inlits, output, N, T, s, votes, typeItarget):
#
#producing feedback matrices: 
#               typeItarget = 1 for clauses with positive polarities
#               typeItarget = 0 for clauses with negative polarities

    num_of_clauses = stateMatrix.shape[0]
    num_of_literals = stateMatrix.shape[1]
    #num_of_clauses = 10
    #num_of_literals = 4
    Ia = np.zeros((num_of_clauses, num_of_literals),int) #Type Ia feedback decision matrix
    # Ia = [0,0,0,0] 
    Ib = np.zeros((num_of_clauses, num_of_literals),int) #Type Ib feedback decision matrix 
    # Ib = [0,0,0,0]
    II = np.zeros((num_of_clauses, num_of_literals),int) #Type II feedback decision matrix
    # II = [0,0,0,0]
    pj1 = (T - max(-T, min(T,votes)))/(2*T)
    pj2 = (T + max(-T, min(T,votes)))/(2*T)
    p1a = (s-1)/s
    p1b = 1/s


    for clindex in range(num_of_clauses):
    
        #evaluating each clause with the eligible literals (TA state is >= N/2)
  
        cloutput = evaluateClause(stateMatrix, clindex, inlits, N)
  
        #providing feedback for eligible clauses and literals
  
        for litindex in range(num_of_literals):
            lit = inlits[litindex]
            # output = 1?
            if output == typeItarget: #potential Type I feedback (typeItarget = 1 for clauses with positive polarities)
                if np.random.rand() < pj1: #selecting a clause for a Type I feedback with probability pj1
                    Ia[clindex, litindex] = typeIaFeedback(p1a,cloutput,lit)
                    Ib[clindex, litindex] = typeIbFeedback(p1b,cloutput,lit)
            else: #output == ~typeItarget,potential Type II feedback (typeItarget = 0 for clauses with positive polarities)
                if np.random.rand() < pj2: #selecting a clause for a Type II feedback with probability pj2
                    II[clindex, litindex] = typeIIFeedback(cloutput,lit)
    return [Ia,Ib,II]

def evaluateClause(stateMatrix, clindex, inlits, N):
#    
#Evaluating a clause with the eligible literals (TA state is >= N/2)
# 
    num_of_literals = stateMatrix.shape[1]
    
    eliglits = np.array([],int)
 
    cl = stateMatrix[clindex]
   
    for litindex in range(num_of_literals):
        if cl[litindex] >= N/2:
            eliglits = np.append(eliglits,inlits[litindex])
   
    cloutput = np.all(eliglits)
   
    return cloutput

def typeIaFeedback(p1a,cloutput,lit):
#    
# Producing Type Ia feedback
#
    if np.random.rand() < p1a and  cloutput == 1 and lit == 1:
        res = 1
    else:
        res = 0
    return res  
 
def typeIbFeedback(p1b,cloutput,lit):
#    
# Producing Type Ib feedback
#
    if np.random.rand() < p1b and (cloutput == 0 or lit == 0):
        res = 1
    else:
        res = 0
    return res  
 
def typeIIFeedback(cloutput,lit):
#    
# Producing Type II feedback
#
    if cloutput == 1 and lit == 0:
        res = 1
    else:
        res = 0
    return res  
 
def sum_up_clause_votes(stateMatrix, N, inlits):
#
#Summing up clause votes
#  
    #print(stateMatrix)
    num_of_clauses = stateMatrix.shape[0]
    #num_of_clauses = 10
    num_of_literals = stateMatrix.shape[1]
    #num_of_literals = 4
    votes = 0
  
    for clindex in range(num_of_clauses):
        
        incllits = np.array([], int)

        cl = stateMatrix[clindex]

        for litindex in range(num_of_literals):
            if cl[litindex] >= N/2:
                incllits = np.append(incllits,inlits[litindex])
  
        if  len(incllits) > 0:
            cloutput = np.all(incllits)
            votes = votes + cloutput
  
    return votes
 
#
# Putting it all togehter 
#  

#TM parameters

m = 10   #total number of clauses
mpn = int(m/2) #nmber of positive/negative clauses
o = 2    #number of input variables (features); 2*o is a number of literals (= number of TAs)
N = 100   #number of TA states
T = 15
s = 3.9

#state matrices
#print(stateMatrix)
Apos = np.ones((mpn, 2*o),int) #state matrix for clauses with positive polarities
#print(Apos)
Aneg = np.ones((mpn, 2*o),int) #state matrix for clauses with negative polarities
#print(Aneg)
num_of_train_runs = 1000
num_of_test_runs = 50

# learning

for runs in range(num_of_train_runs):
      
    #binary input vector
    [inlits,output] = genXORInput() #XOR input
    #inlits = invector = [0,1,1,0]
    #output = 1 or output = 0



    posvotes = sum_up_clause_votes(Apos, N, inlits)
    negvotes = sum_up_clause_votes(Aneg, N, inlits)
    votes = posvotes - negvotes

    #processing clauses with POSITIVE polarity
    [PIa,PIb,PII] = feedbackMatrices(Apos, inlits, output, N, T, s,votes, 1);
    Apos = Apos+PIa-PIb+PII; #making state transitions for each TA
    #logical indexing to bring states to 1-N range
    Apos[Apos>N] = N 
    Apos[Apos<1] = 1
       
    #processing clauses with NEGATIVE polarity
    [NIa,NIb,NII] = feedbackMatrices(Aneg, inlits, output, N, T, s,votes, 0);
    Aneg = Aneg+NIa-NIb+NII; #making state transitions for each TA
    #logical indexing to bring states to 1-N range
    Aneg[Aneg>N] = N
    Aneg[Aneg<1] = 1


# classification
#print('negative')

#print(Aneg)
#print('positive')
#print(Apos)
error_count = 0

for runs in range(num_of_test_runs):
    #binary input vector
    [inlits,output] = genXORInput() #XOR input

    posvotes = sum_up_clause_votes(Apos, N, inlits)
    negvotes = sum_up_clause_votes(Aneg, N, inlits)
    
    if posvotes - negvotes > 0:
        pred = 1
    else:
        pred = 0

    if pred != output:
        error_count = error_count+1

pred_accuracy = (1-error_count/num_of_test_runs)





def clause_sum(matrix):

    clause = ['x1','x2','^x1','^x2']
    #print(matrix)

    #print(Aneg)
#print(Aneg[2])
    rows = len(matrix)
    a = []
    for i in range(rows):
        m = matrix[i]
#m.tolist()
        t = copy.deepcopy(m)

        t = t.tolist()
      

        max_number = []
        max_index = []
        for _ in range(2):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        a.append(sorted(max_index))
        t = []
        #print(max_number)
        #print(max_index)
        #print(a)
        #print('__________')
    #print(a)

    res=list(set([tuple(t) for t in a]))
    #res = list(set(a))

    #print(res)
    clause_res = []
    for a in res:
        clause_temp = []
        for b in range(len(a)):
            #clause_temp = []
            clause_temp.append(clause[a[b]])
        andcha = '&'
        clause_temp = andcha.join(clause_temp) 
        clause_res.append(clause_temp)
    orcha = '|'
    clause_res = orcha.join(clause_res)
    return(clause_res)

a = clause_sum(Aneg)
print('negative')
print(Aneg)
print(a)
print('---------------------')
b = clause_sum(Apos)
print('positive')
print(Apos)
print(b)




    #print(a)
    #print(a[1])
    #print(type(a))
#cols = len(Aneg[0])
#print(rows)
#len(Aneg[2])
#for i in Aneg:
    #for j in i:
        #print(i)

#print('positive')
#print(Apos)
#print(pred_accuracy)

#def a(num):
 
#    return(clause)
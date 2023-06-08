"""
Created on Friday August 13 16:01:52 2021

@author: vladimir (http://www.pitt.edu/~viz/)

Matrix driven implementation of TM, based on TM intro in 

K. D. Abeyrathna, Ole-Christoffer Granmo, Morten Goodwin
Extending the Tsetlin Machine With Integer-Weighted Clauses 
for Increased Interpretability. IEEE Access, 2021

"""
def clause_sum(matrix, o):

    rows = len(matrix)
    a = []
    for i in range(rows):
        m = matrix[i]
        t = copy.deepcopy(m)
        t = t.tolist()
        max_number = []
        max_index = []
        for _ in range(o):
            number = max(t)
            index = t.index(number)
            t[index] = 0
            max_number.append(number)
            max_index.append(index)
        a.append(sorted(max_index))
        t = []


    res=list(set([tuple(t) for t in a]))

    clause_res = []
    for a in res:
        clause_temp = []
        for b in range(len(a)):
            clause_temp.append(clause[a[b]])
        andcha = '&'
        clause_temp = andcha.join(clause_temp) 
        clause_res.append(clause_temp)
    orcha = '|'
    clause_res = orcha.join(clause_res)
    return(clause_res)

def feedbackMatrices(stateMatrix, inlits, output, N, T, s, votes, typeItarget):
#
#producing feedback matrices: 
#               typeItarget = 1 for clauses with positive polarities
#               typeItarget = 0 for clauses with negative polarities

    num_of_clauses = stateMatrix.shape[0]
    num_of_literals = stateMatrix.shape[1]

    Ia = np.zeros((num_of_clauses, num_of_literals),int) #Type Ia feedback decision matrix
    Ib = np.zeros((num_of_clauses, num_of_literals),int) #Type Ib feedback decision matrix 

    II = np.zeros((num_of_clauses, num_of_literals),int) #Type II feedback decision matrix

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



#main function
#m = 10   #total number of clauses
#mpn = int(m/2) #nmber of positive/negative clauses
import pandas as pd
import pymysql
from symbol import classdef
import numpy as np
import copy
con = pymysql.connect(host="localhost",user="root",password="whc000829",db="views")
sql = 'SELECT Country, Year,Count(*) as nums FROM views.rel_dictionary WHERE Victims is not null GROUP BY Country, Year'
data_sql=pd.read_sql(sql,con)
data_sql.to_csv("test2.csv")
df_test1 = pd.read_csv('test2.csv')
df_test1.drop([len(df_test1)-1],inplace=True)
df_mean = df_test1.nums.mean()
df_median = df_test1.nums.median()
df_max= df_test1.nums.max()
df_min = df_test1.nums.min()

df_test1['Bmean']= None
df_test1['Bmedian']= None
df_test1['Bmax']= None
df_test1['Bmin']= None

df_test2 = df_test1.drop(df_test1.columns[[0]], axis=1)

for i in range(len(df_test2)):
    if df_test2.iloc[i,2] >= df_mean:
        df_test2.iloc[i,3] = 1
    else:
        df_test2.iloc[i,3] = 0
        
for i in range(len(df_test2)):
    if df_test2.iloc[i,2] >= df_median:
        df_test2.iloc[i,4] = 1
    else:
        df_test2.iloc[i,4] = 0

for i in range(len(df_test2)):
    if df_test2.iloc[i,2] >= df_max:
        df_test2.iloc[i,5] = 1
    else:
        df_test2.iloc[i,5] = 0

for i in range(len(df_test2)):
    if df_test2.iloc[i,2] >= df_min:
        df_test2.iloc[i,6] = 1
    else:
        df_test2.iloc[i,6] = 0

#Ethiopia is 0, Tanzania is 1
for i in df_test2.index:
    if df_test2.loc[i,'Country'] == 'Ethiopia':
        df_test2.loc[i,'Country'] = 0
    elif df_test2.loc[i,'Country'] == 'Tanzania':
        df_test2.loc[i,'Country'] = 1
        
df_test3 = df_test2[['Bmean','Bmedian','Bmax','Bmin']].copy()
# change input data in each rows to numpy
input_np = df_test3.values

#change output data in each rows to numpy
output_np=df_test2['Country'].values





mpn = len(input_np)
o = len(input_np[0])

#o = 4    #number of input variables (features); 2*o is a number of literals (= number of TAs)
N = 5000  #number of TA states
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
    #[inlits,output] = genOutput() #XOR input
    #inlits = invector = [0,1,1,0]
    #print(len(input_np))
    for i in range(len(input_np)):
        inlits_temp = input_np[i]
        #print(inlits_temp)
        inlits = np.append(inlits_temp,1-inlits_temp)
        #print(inlits)
        
        output = output_np[i]
        #print(output)
        posvotes = sum_up_clause_votes(Apos, N, inlits)
        negvotes = sum_up_clause_votes(Aneg, N, inlits)
        #print(posvotes,negvotes)
        votes = posvotes - negvotes

        

    #processing clauses with POSITIVE polarity
        [PIa,PIb,PII] = feedbackMatrices(Apos, inlits, output, N, T, s,votes, 1);
        Apos = Apos+PIa-PIb+PII; #making state transitions for each TA
    #logical indexing to bring states to 1-N range
        Apos[Apos>N] = N 
        Apos[Apos<1] = 1
        #print(Apos)
    #processing clauses with NEGATIVE polarity
        [NIa,NIb,NII] = feedbackMatrices(Aneg, inlits, output, N, T, s,votes, 0);
        Aneg = Aneg+NIa-NIb+NII; #making state transitions for each TA
    #logical indexing to bring states to 1-N range
        Aneg[Aneg>N] = N
        Aneg[Aneg<1] = 1
        #print(Aneg)




# classification

error_count = 0

for runs in range(num_of_test_runs):
    #binary input vector
    #[inlits,output] = genXORInput() #XOR input
    for i in range(len(input_np)):
        inlits_temp = input_np[i]
        #print(type(inlits_temp))
        inlits = np.append(inlits_temp,1-inlits_temp)
        #print(type(inlits))
        #print(inlits)
        output = output_np[i]
        #print(output)
        posvotes = sum_up_clause_votes(Apos, N, inlits)
        negvotes = sum_up_clause_votes(Aneg, N, inlits)
    
        if posvotes - negvotes > 0:
            pred = 1
        else:
            pred = 0

        if pred != output:
            error_count = error_count+1

pred_accuracy = (1-error_count/num_of_test_runs)

clause = []
for i in range(o):
    pos_name = 'x'+str(i)
    clause.append(pos_name)
for i in range(o):    
    neg_name = '^x'+str(i)
    clause.append(neg_name)
    
a = clause_sum(Aneg,o)
print('negative')
print(Aneg)
print(a)
print('---------------------')
b = clause_sum(Apos,o)
print('positive')
print(Apos)
print(b)




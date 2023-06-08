"""
Created on Friday August 13 16:01:52 2021

@author: vladimir (http://www.pitt.edu/~viz/)

Matrix driven implementation of TM, based on TM intro in 

K. D. Abeyrathna, Ole-Christoffer Granmo, Morten Goodwin
Extending the Tsetlin Machine With Integer-Weighted Clauses 
for Increased Interpretability. IEEE Access, 2021

"""
 

def clause_sum(matrix, clause, threshold):

    rows = len(matrix)
    a = []
    for i in range(rows):
        m = matrix[i]
        t = copy.deepcopy(m)
        t = t.tolist()
        max_index = []
        max_number = []
        for nums in t:
            if nums >= threshold:
                #max_index.append()
                index= t.index(nums)
                max_index.append(index)
        #print(max_index)
        #max_number = []
        #max_index = []
        
        
        #for _ in range(o):
        #    number = max(t)
        #    index = t.index(number)
        #    t[index] = 0
        #    max_number.append(number)
        #    max_index.append(index)
        a.append(sorted(max_index))
        t = []
    #print(a)
    res=list(set([tuple(t) for t in a]))
    #print(res)
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

    num_of_clauses = stateMatrix.shape[0]

    num_of_literals = stateMatrix.shape[1]

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
def train_data(rate,data_input, data_output):
    #rate = 0.3
    mpn = 1
    o = len(data_input[0])
    N = 4000  #number of TA states
    threshold = N * rate
    T = 15
    s = 3.9
    clause = ['Bmean','Bmedian','Bmax','Bmin','NBmean',
              'NBmedian','NBmax','NBmin']
    #for i in range(o):
    #    pos_name = 'x'+str(i)
    #    clause.append(pos_name)
    #for i in range(o):    
    #    neg_name = '^x'+str(i)
    #    clause.append(neg_name)
    
   

#state matrices

    Apos = np.ones((mpn, 2*o),int) #state matrix for clauses with positive polarities

    Aneg = np.ones((mpn, 2*o),int) #state matrix for clauses with negative polarities

    num_of_train_runs = 200
    num_of_test_runs = 50

# learning

    for runs in range(num_of_train_runs):
      
    #binary input vector
        for i in range(len(data_input)):
            inlits_temp = data_input[i]
            inlits = np.append(inlits_temp,1-inlits_temp)
            output = data_output[i]
            posvotes = sum_up_clause_votes(Apos, N, inlits)
            negvotes = sum_up_clause_votes(Aneg, N, inlits)
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
        for i in range(len(data_input)):
            inlits_temp = data_input[i]

            inlits = np.append(inlits_temp,1-inlits_temp)

            output = data_output[i]

            posvotes = sum_up_clause_votes(Apos, N, inlits)
            negvotes = sum_up_clause_votes(Aneg, N, inlits)
    
            if posvotes - negvotes > 0:
                pred = 1
            else:
                pred = 0

            if pred != output:
                error_count = error_count+1

    pred_accuracy = (1-error_count/num_of_test_runs)

    #print(Aneg)
    #print(Apos)
    
    a = clause_sum(Aneg,clause,threshold)
    b = clause_sum(Apos,clause,threshold)
    #print(a)
    #print(type(Aneg))
    return(Aneg,a,Apos,b)
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

sql = "SELECT Year, COUNT(*) as nums FROM views.pulsar_2 WHERE Victims <> '' AND Country = 'Ethiopia' OR Country = 'Tanzania' OR Country = 'Egypt' GROUP BY Year"
data_sql=pd.read_sql(sql,con)
data_sql=pd.read_sql(sql,con)
data_sql.to_csv("test2.csv")
df_test1 = pd.read_csv('test2.csv')
df_test1.drop([len(df_test1)-1],inplace=True)

df_test1['Num_mean'] = None
df_test1['Bmean']= None

df_test1['Num_median'] = None
df_test1['Bmedian']= None

df_test1['Num_max']= None
df_test1['Bmax']= None

df_test1['Num_min']= None
df_test1['Bmin']= None

df_test2 = df_test1.drop(df_test1.columns[[0]], axis=1)

#create three table
names = locals()
for year in range(1999,2014):
    names['df_curr'+str(year)] = df_test2.loc[df_test2['Year'] == year ].copy()
    names['df_others'+str(year)] = df_test2.loc[df_test2['Year'] != year ].copy()
    (names['df_others'+str(year)])['Year'] = 'Others'


#identify each country's other table
year_self_pd = []
year_others_pd = []

for i in range(1999,2014):
    year_self_pd.append(names['df_curr'+str(i)])

    year_others_pd.append(names['df_others'+str(i)])


for a in year_others_pd:
    Num_mean = a.nums.mean()
    a['Num_mean'] = Num_mean
    for i in range(len(a)):
        if a.iloc[i,1] >= Num_mean:
            a.iloc[i,3] = 1
        else:
            a.iloc[i,3] = 0

for a in year_others_pd:
    Num_median = a.nums.median()
    a['Num_median'] = Num_median
    for i in range(len(a)):
        if a.iloc[i,1] >= Num_median:
            a.iloc[i,5] = 1
        else:
            a.iloc[i,5] = 0

for a in year_others_pd:
    Num_max = a.nums.max()
    a['Num_max'] = Num_max
    for i in range(len(a)):
        if a.iloc[i,1] >= Num_max:
            a.iloc[i,7] = 1
        else:
            a.iloc[i,7] = 0
            
for a in year_others_pd:
    Num_min = a.nums.min()
    a['Num_min'] = Num_min
    for i in range(len(a)):
        if a.iloc[i,1] <= Num_min:
            a.iloc[i,9] = 1
        else:
            a.iloc[i,9] = 0


# identify each table itself
for b in range(len(year_self_pd)):
    if year_self_pd[b].iloc[0,1] >= year_others_pd[b].nums.mean():
        year_self_pd[b].iloc[0,3] = 1
    else:
        year_self_pd[b].iloc[0,3] = 0

for b in range(len(year_self_pd)):
    #for i in range(year_self_pd[b]):
    if year_self_pd[b].iloc[0,1] >= year_others_pd[b].nums.median():
        year_self_pd[b].iloc[0,5] = 1
    else:
        year_self_pd[b].iloc[0,5] = 0

for b in range(len(year_self_pd)):
    #for i in range(year_self_pd[b]):
    if year_self_pd[b].iloc[0,1] >= year_others_pd[b].nums.max():
        year_self_pd[b].iloc[0,7] = 1
    else:
        year_self_pd[b].iloc[0,7] = 0

for b in range(len(year_self_pd)):
    #for i in range(year_self_pd[b]):
    if year_self_pd[b].iloc[0,1] <= year_others_pd[b].nums.min():
        year_self_pd[b].iloc[0,9] = 1
    else:
        year_self_pd[b].iloc[0,9] = 0
            
#merge these table
for year in range(1999,2014):
    names['df_'+str(year)+'_merge'] = pd.concat([names['df_curr'+str(year)],names['df_others'+str(year)]])

# Country is 1, others' county is 0
for year in range(1999,2014):
    c = names['df_'+str(year)+'_merge']
    for i in c.index:
        if c.loc[i,'Year'] != 'Others':
            c.loc[i,'Year'] = 1
        elif c.loc[i,'Year'] == 'Others':
            c.loc[i,'Year'] = 0

#copy booleanize table
for year in range(1999,2014):
    names['df_'+str(year)+'_final'] = (names['df_'+str(year)+'_merge'])[['Bmean','Bmedian','Bmax','Bmin']].copy()

# change input data in each rows to numpy
Year_input_np = df_2012_final.values

#change output data in each rows to numpy
Year_output_np = df_2012_merge['Year'].values

c = ['Bmean','Bmedian','Bmax','Bmin','NBmean','NBmedian','NBmax','NBmin']
c = np.array(c)
ra = [0.1,0.2,0.4,0.6,0.8]
#for r in range(len(ra)):
#    Eth_result = train_data(ra[r],Eth_input_np,Eth_output_np)
#    if r == 0:
#        print('Ethiopia')
#        #print('Negative')
#        print(np.vstack((c,Eth_result[0])))
#        #print('Positive')
#        #print(Eth_result[2])
#    print('Threshold =' ,ra[r])
#    print(Eth_result[1])
#    #print('Pos:', Eth_result[3])
#print('-----------------------------------------------')
#for r in range(len(ra)):
#    Tan_result = train_data(ra[r],Tan_input_np,Tan_output_np)
#    if r == 0:
#        print('Tanzania')
#        #print('Negative')
#        print(np.vstack((c,Tan_result[0])))
#        #print('Positive')
#        #print(Tan_result[2])
#    print('Threshold =' ,ra[r])
#    print(Tan_result[1])
    #print('Pos:', Tan_result[3])
#print('-----------------------------------------------')
for r in range(len(ra)):
    Egy_result = train_data(ra[r],Year_input_np,Year_output_np)
    if r == 0:
        print('2012')
        print('Negative')
        print(np.vstack((c,Egy_result[0])))
        #print(Egy_result[0])
        print('Positive')
        print(np.vstack((c,Egy_result[2])))
        #print(np.vstack((c,Eth_result[2])))
    print('Threshold =' ,ra[r])
    print('Neg:',Egy_result[1])
    print('Pos:',Egy_result[3])






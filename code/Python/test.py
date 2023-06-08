import pandas as pd
import pymysql
from symbol import classdef
import numpy as np
import copy
con = pymysql.connect(host="localhost",user="root",password="whc000829",db="views")

sql = "SELECT Country, Year, COUNT(*) as nums FROM views.pulsar_2 WHERE Victims <> '' GROUP BY Country, Year"
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

df = pd.read_csv('1.csv')
df1 = pd.read_csv('X1mo_data_incidence.csv')
df = df.rename(columns = {'country_name': 'Country'})
df = df.dropna()

df_test2['Country'].replace('Burkina Faso','Burkina_Faso',inplace=True)
df_test2['Country'].replace('South Africa','South_Africa',inplace=True)
df_test2['Country'].replace('Cape Verde','Cape_Verde',inplace=True)
df_test2['Country'].replace('Central African Republic','Central_African_Republic',inplace=True)
df_test2['Country'].replace('Democratic Republic Of The Congo','Democratic_Republic_Of_The_Congo',inplace=True)
df_test2['Country'].replace('Republic of Congo','Republic_of_Congo',inplace=True)
df_test2['Country'].replace('Equatorial Guinea','Equatorial_Guinea',inplace=True)
#df_test2['Country'].replace('Equatorial Guinea','Equatorial_Guinea',inplace=True)
#dfmerge = dfmerge.replace('Equatorial Guinea','Equatorial_Guinea')
df['Country'].replace('Burkina Faso','Burkina_Faso',inplace=True)
df['Country'].replace('South Africa','South_Africa',inplace=True)
df['Country'].replace('Cape Verde','Cape_Verde',inplace=True)
df['Country'].replace('Central African Republic','Central_African_Republic',inplace=True)
df['Country'].replace('Democratic Republic Of The Congo','Democratic_Republic_Of_The_Congo',inplace=True)
df['Country'].replace('Republic of Congo','Republic_of_Congo',inplace=True)
df['Country'].replace('Equatorial Guinea','Equatorial_Guinea',inplace=True)
dfmer = pd.merge(df,df_test2, on=['Country'])

df1 = df1.drop(df1.columns[[0]], axis=1)

#X1_output = dfX1[['year','country_name','country_num']].copy()
#X1_output = df1.dropna()
X1_output = df1.reset_index()
X1_output = X1_output.drop(X1_output.columns[[0]], axis=1)
X1_output.rename(columns={'country_name':'Country', 'year':'Year'}, inplace = True)
X1_output = X1_output.groupby(['Country','Year','country_num']).agg('sum')
X1_output.reset_index(inplace = True)
#dfmerge1 = pd.merge(X1_output,df_output_name, how = 'cross')


X1_output['Country'].replace('Burkina Faso','Burkina_Faso',inplace=True)
X1_output['Country'].replace('South Africa','South_Africa',inplace=True)
X1_output['Country'].replace('Cape Verde','Cape_Verde',inplace=True)
X1_output['Country'].replace('Central African Republic','Central_African_Republic',inplace=True)
X1_output['Country'].replace('Democratic Republic Of The Congo','Democratic_Republic_Of_The_Congo',inplace=True)
X1_output['Country'].replace('Republic of Congo','Republic_of_Congo',inplace=True)
X1_output['Country'].replace('Equatorial Guinea','Equatorial_Guinea',inplace=True)

dfmerge = pd.merge(X1_output,dfmer, on=['Country','Year','country_num'], how = "inner")

country_list = list(set(dfmerge['Country'].values.tolist()))

#create a dictionary about country and year appeared.
country_year_out={}
for country in country_list:
    year = list(dfmerge['Year'].loc[dfmerge['Country'] == country])
    year = list(set(year))
    out = list(dfmerge['incidence_1'].loc[dfmerge['Country'] == country])
    out = list(set(out))
    country_year_out[country] = {}
    for i in year:
        #print(i)
        country_year_out[country][i] = out


#create three table
names = locals()
#Country = ['Ethiopia','Tanzania','Egypt']
for coun in country_list:
    for year in country_year_out[coun]:
        for output in country_year_out[coun][year]:
            names['df_curr'+str(year)+str(coun)+str(output)] = dfmerge.loc[(dfmerge['Year'] == year)&(dfmerge['Country'] == coun)&(dfmerge['incidence_1'] == output)].copy()
            names['df_others'+str(year)+str(coun)+str(output)] = dfmerge.loc[(dfmerge['Year'] != year) | (dfmerge['Country'] != coun) | (dfmerge['incidence_1'] != output)].copy()
            #remain self_country to df_curr and replace 'others' to df_others table
            (names['df_others'+str(year)+str(coun)+str(output)])['Year'] = 'Others'
            (names['df_others'+str(year)+str(coun)+str(output)])['Country'] = 'Others'
            (names['df_others'+str(year)+str(coun)+str(output)])['incidence_1'] = 'Others'

year_self_pd = []
year_others_pd = []
for coun in country_list:
    for year in country_year_out[coun]:
        for output in country_year_out[coun][year]:
            year_self_pd.append(names['df_curr'+str(year)+str(coun)+str(output)])
            year_others_pd.append(names['df_others'+str(year)+str(coun)+str(output)])


#fill data in 'others' table
for a in year_others_pd:
    Num_mean = a.nums.mean()
    a['Num_mean'] = Num_mean
    for i in range(len(a)):
        if a.iloc[i,-9] >= Num_mean:
            a.iloc[i,-7] = 1
        else:
            a.iloc[i,-7] = 0

for a in year_others_pd:
    Num_median = a.nums.median()
    a['Num_median'] = Num_median
    for i in range(len(a)):
        if a.iloc[i,-9] >= Num_median:
            a.iloc[i,-5] = 1
        else:
            a.iloc[i,-5] = 0

for a in year_others_pd:
    Num_max = a.nums.max()
    a['Num_max'] = Num_max
    for i in range(len(a)):
        if a.iloc[i,-9] >= Num_max:
            a.iloc[i,-3] = 1
        else:
            a.iloc[i,-3] = 0
            
for a in year_others_pd:
    Num_min = a.nums.min()
    a['Num_min'] = Num_min
    for i in range(len(a)):
        if a.iloc[i,-9] <= Num_min:
            a.iloc[i,-1] = 1
        else:
            a.iloc[i,-1] = 0


for b in range(len(year_self_pd)):
    print(b)
    #print(year_self_pd[b].iloc[0,-9])
    #print(year_others_pd[b].nums.mean())
    if year_self_pd[b].iloc[0,-9] >= year_others_pd[b].nums.mean():
        year_self_pd[b].iloc[0,-7] = 1
    else:
        year_self_pd[b].iloc[0,-7] = 0

    print(b)
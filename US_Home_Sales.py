import numpy as np
import pandas as pd

#Get your data into a DataFrame
df = pd.read_csv('C:/Users/Desktop/python/data manipulation/US_Home_Sales.csv')

#Working with the whole DataFrame
print(df.info())
print(df.describe())
print(df.shape)

#Working with rows
df = df[df['cat_code']=='SOLD']
df = df[(df['dt_code']!='MEDIAN') & (df['dt_code']!='NOTSTD') & (df['dt_code']!= 'TOTAL')]
df = df[pd.notnull(df['dt_code'])]

#Working with columns
df = df.drop(['et_idx','et_code','et_desc','et_unit'], axis=1)
df = df[['per_name','dt_code','dt_desc','dt_unit','val']]
df['dt_year'] = df['per_name'].str[0:4]

#Function 
def price (x):
    if x.dt_unit=="K": return(x.val*1000)
    else: return(x.val)
df['dt_val'] = df[['dt_unit','val']].apply(price, axis=1)
df = df.drop(['dt_desc'], axis=1)
df.tail()

#Cleaning datasets 
df_UNDERC = df[df['dt_code']=='UNDERC']
df_COMPED = df[df['dt_code']=='COMPED']
df_AVERAG = df[df['dt_code']=='AVERAG']

df_UNDERC = df_UNDERC[['per_name','dt_year','dt_val']]
df_COMPED = df_COMPED[['per_name','dt_year','dt_val']]
df_AVERAG = df_AVERAG[['per_name','dt_year','dt_val']]

df_UNDERC = df_UNDERC.rename(columns={'dt_val':'UNDERC'})
df_COMPED = df_COMPED.rename(columns={'dt_val':'COMPED'})
df_AVERAG = df_AVERAG.rename(columns={'dt_val':'AVERAG'})

#Joining/Combining DataFrames
#Groupby: Split-Apply-Combine
df_new = pd.merge(left=df_AVERAG, right=df_COMPED, how='left', 
                  left_on='per_name',right_on='per_name') 
df_new = pd.merge(left=df_new, right=df_UNDERC, how='left', 
                  left_on='per_name',right_on='per_name')

gb = df_new.groupby(['dt_year'])['AVERAG','COMPED','UNDERC'].agg(np.mean)
gb.describe()

#Graphs
gb['AVERAG'].plot.line(color='green')
gb[['COMPED','UNDERC']].plot.line()

#Cells
print('AVG price in 1996: ',round(gb.loc['1996','AVERAG'],2))
print('AVG price in 2006: ',round(gb.loc['2006','AVERAG'],2))
print('AVG price in 2016: ',round(gb.loc['2016','AVERAG'],2))

print(gb.loc['2016'])
print(gb['AVERAG'])

gb.corr()

quants = [0.05, 0.25, 0.5, 0.75, 0.95]
df_new = gb.quantile(quants)
print(df_new)

import matplotlib.pyplot as plt
count, bins = np.histogram(gb['AVERAG'],bins=5)
plt.hist(gb['AVERAG'],bins=5,color='gray',edgecolor='white')

#Pivot Tables
df_pt = pd.pivot_table(df,index='dt_year',columns='dt_code', 
                       values='dt_val',aggfunc=np.mean)
df_pt.head()
df_pt.tail()

#Working with rows 
gb = gb.sort_values(by='AVERAG',ascending=False)
gb.head()
gb = gb.sort_index()
gb.head()
gb = gb[32:]

#Working with columns
print(gb['AVERAG'].idxmin())
print(gb['AVERAG'].idxmax())

print(gb.count())
print(gb.mean())

print(gb['AVERAG'].count())
print(gb['AVERAG'].mean())
print(gb['AVERAG'].describe())

s = gb['AVERAG'].fillna(0)
s = df_new['AVERAG'].cumsum()

df_new = gb
df_new['log_data'] = np.log(df_new['AVERAG'])
df_new['rounded'] = np.round(df_new['log_data'], 2)


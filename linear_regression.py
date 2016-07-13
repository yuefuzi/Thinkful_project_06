import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('/Users/Chang/Desktop/data_science/loansData.csv')

# Data Cleaning

df = df.dropna()
df.index = range(len(df))


g = lambda x: round(np.float(x.rstrip('%')),4)
for i in range(0,len(df)):
    df.set_value(i,'Interest.Rate',g(df['Interest.Rate'].loc[i]))
    
df['Interest.Rate']= pd.to_numeric(df['Interest.Rate'])
    
f = lambda x: round(float(x.rstrip('months')),4)
for i in range(0,len(df)):
    df.set_value(i,'Loan.Length',f(df.iloc[i,3]))  
    
df['Loan.Length']= pd.to_numeric(df['Loan.Length'])

df['FICO.Score']=pd.Series(np.random.randn(len(df)))

for i in range(0,len(df)):
    s = df.iloc[i,9]
    s = s.split('-')
    df.iloc[i,14]=float(s[0])

lm = smf.ols(formula="Q('Interest.Rate')~Q('Amount.Requested')+Q('FICO.Score')",data=df).fit()
print(lm.summary())


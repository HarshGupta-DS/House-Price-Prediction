Kaggle_Competetion!
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
import warnings
#warnings.filterwarning('ignore')
# %matplotlib inline

df_train = pd.read_csv('/content/train.csv')

df_train.columns

df_train['SalePrice'].describe()

#Histogram
sns.displot(df_train['SalePrice']);

#skewness and Kurtosis (Measure of symmetry and measure wether the data is heavily tailed.)
print("Skewness:%f" % df_train['SalePrice'].skew())
print("Kurtosis:%f" % df_train['SalePrice'].kurt())

#Scatter Plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]],axis = 1)
data.plot.scatter(x=var,y='SalePrice',ylim = (0,700000))

var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
data.plot.scatter(x=var,y = 'SalePrice',ylim = (0,700000))

var = 'OverallQual'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis = 1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var,y="SalePrice",data=data)
fig.axis(ymin=0, ymax = 800000)

var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'],df_train[var]],axis=1)
f, ax = plt.subplots(figsize=(16,8))
fig = sns.boxplot(x=var,y="SalePrice", data=data)
fig.axis(ymin=0,ymax=800000);
plt.xticks(rotation=90)

#Correlation Matrix (HeatMap:- Shades of same colour)
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8,square= True);

#SalePrice Correlation Matrix
k = 10 #Number of Variables for heatmap
cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm,cbar = True, annot = True, fmt = '.2f', 
                 annot_kws={'size':10},yticklabels = cols.values,
                 xticklabels = cols.values)
plt.show()

#Scatter plots between 'SalePrice' and correlated variables (move like Jagger style)
#Scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size=2.5)
plt.show()

#Missing Data
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending = False)
missing_data = pd.concat([total, percent],axis=1, keys = ['Total','Percent'])
missing_data.head(28)

#Dealing and deleting with Missing data
df_train = df_train.drop((missing_data[missing_data['Total']>1]).index,1)
df_train = df_train.drop(labels = ["Electrical"],axis=1)
df_train.isnull().sum().max()

#Data Standarizing
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#Bivariate Analysis
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice',ylim = (0,800000))

#Deleting Points
df_train.sort_values(by = 'GrLivArea', ascending= False)[:2]
df_train = df_train.drop(df_train[df_train['Id']==1298].index)
df_train = df_train.drop(df_train[df_train['Id']==523].index)

#Bivariate Analysis SalePrice VS grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'],df_train[var]], axis =1)
data.plot.scatter(x=var, y='SalePrice',ylim = (0,800000));

#Histogram and Normal Probability plot
sns.distplot(df_train['SalePrice'], fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'], plot=plt)

#Apply log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])

#Transformed Histogram and Normal Probability
sns.distplot(df_train['SalePrice'],fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['SalePrice'],plot=plt)

#Histogram and Normal Probability
sns.distplot(df_train['GrLivArea'],fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot=plt)

#Data Transformation
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

#Transformed Histogram and Normal Probability
sns.distplot(df_train['GrLivArea'],fit = norm);
fig = plt.figure()
res = stats.probplot(df_train['GrLivArea'],plot=plt)

sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df_train['TotalBsmtSF'],plot=plt)

#Creating Column for new variable (One is because it's a binary categorical feature)
#if area>0 it gets 1, for area == 0 it gets 0;
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt']=1

#Transform Data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

#Histogram and Normal Probabiliy Plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig=plt.figure()
res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],plot=plt)

#Scatter Plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);

#Scatter Plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'],df_train[df_train['TotalBsmtSF']>0]['SalePrice']);

#Convert Categorical Variable into dummy
df_train = pd.get_dummies(df_train)
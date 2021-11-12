#!/usr/bin/env python
# coding: utf-8

# In[54]:


import itertools
import urllib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import csv
from pandas import read_csv
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[81]:


df = pd.read_csv("https://media.githubusercontent.com/media/tayher2679/Github/main/train.csv")
df.head()


# In[82]:


df.shape


# In[83]:


df.describe


# In[59]:


dftitle=pd.read_csv("https://media.githubusercontent.com/media/tayher2679/Github/3a342842df53002d6b17bfcee99e60a6f096c9ce/data_dictionary.csv")
dftitle


# In[84]:


df.loan_default.value_counts().plot(kind='bar')


# In[85]:


df.describe(include="all").T


# In[86]:


df.info()


# In[ ]:


#Note we will need to convert object type variables to category/numbers= Date of Birth, Employment Type, Disbursal Date, Perform CNS Score description, Average Acct Age and Credit history length


# In[87]:


df.isna().sum()


# In[88]:


dfemploy1=df['Employment.Type'].fillna('Not declared', inplace=True)


# In[89]:


dfemploy1=df.groupby(['Employment.Type'])['loan_default'].count()
dfemploy1.plot(kind='bar')


# In[90]:


#seems like the non-declared employment can be factor in loan delinquencies. So cannot ignore as a factor. 
df['Employment.Type'].fillna('Not declared')
df.info()


# In[92]:


#Data Preprocessing
#drop variables that are unique ID and high cardinality 
df = df.drop(['UniqueID','branch_id','supplier_id','manufacturer_id','Current_pincode_ID','Employee_code_ID'],axis=1)
df.info()


# In[94]:


#converting object datatype into appropriate datatype

df['Employment.Type'] = df['Employment.Type'].astype('category')
df['PERFORM_CNS.SCORE.DESCRIPTION'] = df['PERFORM_CNS.SCORE.DESCRIPTION'].astype('category')

def dateconv(x,format):
    year = pd.datetime.today().year
    dob = pd.to_datetime(x,format = format)
    dob.loc[dob.dt.year.gt(year)] -= pd.DateOffset(years=100)
    return dob

df['Date.of.Birth'] = dateconv(df['Date.of.Birth'], '%d-%m-%y')
df['DisbursalDate'] = dateconv(df['DisbursalDate'], '%d-%m-%y')

def duration(dur):
    yrs = int(dur.split(' ')[0].replace('yrs',''))
    mon = int(dur.split(' ')[1].replace('mon',''))
    return yrs*12+mon

df['AVERAGE.ACCT.AGE'] = df['AVERAGE.ACCT.AGE'].apply(duration)
df['CREDIT.HISTORY.LENGTH'] = df['CREDIT.HISTORY.LENGTH'].apply(duration)


# In[95]:


df.info()


# In[97]:


df.head().T


# In[98]:


df.describe(include="all").T


# In[101]:


#covert DOB to age
import datetime
def age(born):
    thedate='2018-12-31'
    thedate = datetime.datetime.strptime(thedate, '%Y-%m-%d')
    return thedate.year - born.year - ((thedate.month, 
                                      thedate.day) < (born.month, 
                                                    born.day))
  
df['Age'] = df['Date.of.Birth'].apply(age)
df.head().T


# In[104]:


df=df.drop(['Date.of.Birth'],axis=1)


# In[105]:


df.head().T


# In[118]:


#MOve Age to the front of the data
age = df['Age']
df = df.drop(columns=['Age'])
df.insert(loc=3, column='Age', value=age)
df.head(20).T


# In[114]:


#check what is Perform_CNS_SCORE
df['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()


# In[117]:


#change it to number category


# In[119]:


score_default= pd.crosstab(df['PERFORM_CNS.SCORE.DESCRIPTION'],df['loan_default'],normalize='index')
score_default.plot.bar()
plt.title('CNS Score BoxPlot')


# In[121]:


corr = df.corr()
sns.heatmap(corr)
plt.title('Heat Map')


# In[129]:


def replace_not_scored(n):
    score=n.split("-")
    
    if len(score)!=1:
        return score[0]
    else:
        return 'N'

def transform_CNS_Description(data):
    data['CNS.SCORE.DESCRIPTOR']=data['PERFORM_CNS.SCORE.DESCRIPTION'].apply(replace_not_scored).astype(np.object)
    
    #Now Transform CNS Score Description data into Numbers

    sub_risk = {'N':0, 'C':1, 'A':1, 'D':1, 'B':1, 'F':2, 'E':2,'G':2, 'H':3, 'I':3, 'K':4, 'J':4, 'L':5, 'M':5}

    data['CNS.SCORE.DESCRIPTOR'] = data['CNS.SCORE.DESCRIPTOR'].apply(lambda x: sub_risk[x])
    


# In[133]:


df = df.drop(columns=['PERFORM_CNS.SCORE.DESCRIPTION'])
df = df.drop(columns=['CNS_DESCRIPTION'])


# In[134]:


df.head(10).T


# In[135]:


#MOve CNS Score Descriptor next to CNS Score 
CNS = df['CNS.SCORE.DESCRIPTOR']
df = df.drop(columns=['CNS.SCORE.DESCRIPTOR'])
df.insert(loc=13, column='CNS.SCORE.DESCRIPTOR', value=CNS)
df.head(10).T


# In[137]:


df.describe(include="all").T


# In[156]:


#evaluate whether the seconday accounts information is vital. Looks like its mostly 0 with some outlier
df['SEC.NO.OF.ACCTS'].value_counts().plot(kind='bar')


# In[140]:


df.boxplot('SEC.NO.OF.ACCTS')
plt.title('BoxPlot')


# In[157]:


#how many have 1 or more than 1 secondary accounts, and whether it has impact to loan default?
secondary=df[['SEC.NO.OF.ACCTS',
       'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE',
       'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT']]


# In[158]:


secondary['SEC.NO.OF.ACCTS'].value_counts()


# In[159]:


secondary.describe()


# In[160]:


#looks like secondary only a minority , with only 2.5% have more than 1 secondary account. Will combine it into the primary accounts

df.loc[:,'T.No.of.Accts']=df['PRI.NO.OF.ACCTS']+df['SEC.NO.OF.ACCTS']
df.loc[:,'Pri.Inactive.Accts']=df['PRI.NO.OF.ACCTS']-df['PRI.ACTIVE.ACCTS']
df.loc[:,'Sec.Inactive.Accts']=df['SEC.NO.OF.ACCTS']-df['SEC.ACTIVE.ACCTS']
df.loc[:,'T.Inactive.Accts']=df['Pri.Inactive.Accts']+df['Sec.Inactive.Accts']
df.loc[:,'T.Overdue.Accts']=df['PRI.OVERDUE.ACCTS']+df['SEC.OVERDUE.ACCTS']
df.loc[:,'T.Current.Balance']=df['PRI.CURRENT.BALANCE']+df['SEC.CURRENT.BALANCE']
df.loc[:,'T.Disbursed.Amount']=df['PRI.DISBURSED.AMOUNT']+df['SEC.CURRENT.BALANCE']
df.loc[:,'T.Sanctioned.Amount']=df['PRI.SANCTIONED.AMOUNT']+df['SEC.SANCTIONED.AMOUNT']
df.loc[:,'T.Installment']=df['PRIMARY.INSTAL.AMT']+df['SEC.SANCTIONED.AMOUNT']


# In[161]:


df.describe(include="all").T


# In[162]:


df.head(10).T


# In[163]:


df=df.drop(['PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS', 'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT','SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS', 'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT','Pri.Tenure','Sec.Tenure','Disburse.to.Sanctioned'],axis=1)


# In[164]:


df.head(10).T


# In[173]:


#check all t he FLags has how many unique outcome
flag=df[['MobileNo_Avl_Flag',
       'Aadhar_flag', 'PAN_flag', 'VoterID_flag',
       'Driving_flag', 'Passport_flag']]
print(flag.nunique())


# In[183]:


#Mobile No. Avl Flag doesn't seem to offer any insight. Can drop this colum
df=df.drop(columns=['MobileNo_Avl_Flag'])


# In[184]:


df.head(10).T


# In[177]:


#check the rest of the flag whether they offer any statistically signifcant insights

plt.figure(figsize=(10,8))
sns.countplot(x='Aadhar_flag',hue='loan_default',data=df)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# In[178]:


plt.figure(figsize=(10,8))
sns.countplot(x='VoterID_flag',hue='loan_default',data=df)
plt.legend(labels=['Not-Defaulted','Defaulted'])
plt.show()


# In[ ]:


# Initial check doens't look like it offers any statistic significance. Will evaluate


# In[191]:


import scipy.stats as stats
from scipy.stats import chi2_contingency
flag1=df[['Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']]
for i in flag1:
    print('Feature:',i)
    chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(pd.crosstab(df[i],df['loan_default']))
    print('Chi Square Statistics',chi_sq)
    print('p-value',p_value)
    print('Degree of freedom',deg_freedom)
    print()


# In[192]:


# with the exception of PAN_flag, the rest of the flag has very low P-value, meaning null hypothesis can be accepted= no correlation with loan default. 
df=df.drop(columns=['Aadhar_flag',  'VoterID_flag', 'Driving_flag', 'Passport_flag'])
df.head(10).T


# In[193]:


#change category into numberics for Employemnt type
df['Employment.Type'].replace(to_replace=['Salaried','Self employed','Not declared'], value=[2,1,0],inplace=True)
df.head(10).T


# In[199]:


#covert Disbursal Date to Disbursed duration

def disburse(start):
    thedate='2018-12-31'
    thedate = datetime.datetime.strptime(thedate, '%Y-%m-%d')
    return (thedate.year - start.year)*12+(thedate.month-start.month) 
  
df['DisburseDur'] = df['DisbursalDate'].apply(disburse)
df.head().T


# In[200]:


df=df.drop(columns=['DisbursalDate'])
df.head().T


# In[201]:


df=df.drop(columns=['Pri.Inactive.Accts','Sec.Inactive.Accts'])
df.head().T


# In[204]:


# Check each feauture for Chisquared
featurecheck=df[df.columns.difference(['loan_default'])]
for i in featurecheck:
    print('Feature:',i)
    chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(pd.crosstab(df[i],df['loan_default']))
    print('Chi Square Statistics',chi_sq)
    print('p-value',p_value)
    print('Degree of freedom',deg_freedom)
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#break below is my old code for another project on credit that I have done


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
print('Train', X_train.shape, y_train.shape)
print('Test', X_test.shape, y_test.shape)


# In[ ]:





# In[ ]:


df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()


# In[ ]:


df['loan_status'].value_counts()


# In[ ]:


# notice: installing seaborn might takes a few minutes
get_ipython().system('conda install -c anaconda seaborn -y')


# In[ ]:


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# In[ ]:


df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


# In[ ]:


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()


# In[ ]:


df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)


# In[ ]:


df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


# In[ ]:


df.groupby(['education'])['loan_status'].value_counts(normalize=True)


# In[ ]:


df[['Principal','terms','age','Gender','education']].head()


# In[ ]:


Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()
Feature.shape


# In[ ]:


X = Feature
X[0:5]


# In[ ]:


y = df['loan_status'].values
y[0:5]


# In[ ]:


X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
X.shape


# In[ ]:


#classification
import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


loantesttree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
loantesttree.fit(X_train,y_train)
predTree = loantesttree.predict(X_test)
from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))


# In[ ]:


print("Avg F1-score: %.4f" % f1_score(y_test, predTree, average='weighted'))
print("Jaccard score: %.4f" % jaccard_similarity_score(y_test, predTree))


# In[ ]:


from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train) 
yhat2 = clf.predict(X_test)
yhat2 [0:5]


# In[ ]:


from sklearn.metrics import f1_score
f1_score(y_test, yhat2, average='weighted')
from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test, yhat2)
print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_similarity_score(y_test, yhat2))


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[ ]:


k = 9
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

yhat = neigh.predict(X_test)
yhat[0:5]

from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# In[ ]:


Ks = 15
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
print("Avg F1-score: %.4f" % f1_score(y_test, yhat, average='weighted'))
print("Jaccard score: %.4f" % jaccard_similarity_score(y_test, yhat))


# In[ ]:


#model evaluation
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss


# In[ ]:


get_ipython().system('wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv')


# In[ ]:


test_df = pd.read_csv('loan_test.csv')
test_df.head()
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Feature1 = test_df[['Principal','terms','age','Gender','weekend']]
Feature1 = pd.concat([Feature1,pd.get_dummies(test_df['education'])], axis=1)
Feature1.drop(['Master or Above'], axis = 1,inplace=True)
Xtest1 = Feature1
ytest1 = test_df['loan_status'].values
Xtest1= preprocessing.StandardScaler().fit(Xtest1).transform(Xtest1)
Xtest1.shape


# In[ ]:


#KNN
yhat4 = neigh.predict(Xtest1)
yhat4[0:5]
print("Avg F1-score for KNN: %.4f" % f1_score(ytest1, yhat4, average='weighted'))
print("Jaccard score for KNN: %.4f" % jaccard_similarity_score(ytest1, yhat4))


# In[ ]:


predTree1 = loantesttree.predict(Xtest1)
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(ytest1, predTree1))
print("Avg F1-score for Decision Tree: %.4f" % f1_score(ytest1, predTree1, average='weighted'))
print("Jaccard score for Decision Tree: %.4f" % jaccard_similarity_score(ytest1, predTree1))


# In[ ]:


yhat5 = clf.predict(Xtest1)
f1_score(ytest1, yhat5, average='weighted')
jaccard_similarity_score(ytest1, yhat5)
print("Avg F1-score for SVM: %.4f" % f1_score(ytest1, yhat5, average='weighted'))
print("Jaccard score for SVM: %.4f" % jaccard_similarity_score(ytest1, yhat5))


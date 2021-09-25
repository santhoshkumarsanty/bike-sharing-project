#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Demand

# ## Problem Statement:
# 

# A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues. They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:
# 
#    1. Which variables are significant in predicting the demand for shared bikes.
#    2.How well those variables describe the bike demands
#   
# You are required to model the demand for shared bikes with the available independent variables. It will be used by the management to understand how exactly the demands vary with different features. They can accordingly manipulate the business strategy to meet the demand levels and meet the customer's expectations. Further, the model will be a good way for management to understand the demand dynamics of a new market.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import warnings
from scipy import stats as stats
from scipy.stats import skew
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.linear_model import LinearRegression,Ridge,LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler,RobustScaler
warnings.filterwarnings('ignore')



# In[2]:


df=pd.read_csv(r'D:\data\aswintech\Machine-Learning-Projects-master\Bike Sharing Demand Analysis - Regression\hour.csv')


# In[3]:


df.head(5)


# In[4]:


for col in df.columns:
    print('---------',col,'-------------')
    print(df[col].value_counts())
    print('NULL VALUE :',df[col].isnull().sum())
    print()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.drop(['dteday'],axis=1,inplace=True)


# In[8]:


df.info()


# In[9]:


df.isnull().value_counts().sum()


# In[10]:


df = df.rename(columns={'weathersit':'weather',
                       'yr':'year',
                       'mnth':'month',
                       'hr':'hour',
                       'hum':'humidity',
                       'cnt':'count'})


# In[11]:


df.head()


# In[12]:


df['count'].value_counts()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
plt.title("missing values")


# In[14]:


df.apply(lambda x: len(x.unique()))


# In[15]:


df.isnull().sum()


# In[16]:


df=df.drop(columns=['instant','year'])


# In[17]:


dic={1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
df['season']=df['season'].map(dic)
dic3={0:'sunday',1:'monday',2:'tuesday',3:'wednesday',4:'thursday',5:'friday',6:'saturday'}
df['weekday']=df['weekday'].map(dic3)
dic4 = {1:'Clear', 2:'Misty+Cloudy', 3:'Light_Snow/Rain', 4:'Heavy_Snow/Rain'}
df['weather']=df['weather'].map(dic4)


# # EXPLOTARY DATA ANALYSIS

# In[18]:


fig, ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df,x='hour',y='count',hue='weekday')
ax.set(title='count of the bikes in weekdays and weekend')


# In[19]:


fig, ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df,x='hour',y='casual',hue='weekday')
ax.set(title='count of the bikes in weekdays and weekend :unregister')


# In[20]:


fig, ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df,x='hour',y='count',hue='season')
ax.set(title='count of the bikes in weekdays and weekend: season')


# In[21]:


fig, ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df,x='hour',y='registered',hue='weekday')
ax.set(title="count of the bikes in weekdays and weekend :register")


# In[22]:


fig, ax=plt.subplots(figsize=(20,10))
sns.pointplot(data=df,x='hour',y='count',hue='weather')
ax.set(title='count of the bikes in weekdays and weekend :weather')


# In[23]:


fig, ax=plt.subplots(figsize=(20,10))
sns.barplot(data=df,x='month',y='count')
ax.set(title='count of the bikes in months')


# In[24]:


fig, ax=plt.subplots(figsize=(20,10))
sns.barplot(data=df,x='season',y='count')
ax.set(title='count of the bikes in season')


# In[25]:


fig, ax=plt.subplots(figsize=(20,10))
sns.barplot(data=df,x='weekday',y='count')
ax.set(title='count of the bikes in weekday')


# In[26]:


fig, ax=plt.subplots(figsize=(20,10))
sns.barplot(data=df,x='weather',y='count')
ax.set(title='count of the bikes in weather')


# In[27]:


fig,(ax1,ax2) =plt.subplots(ncols=2,figsize=(20,5))
sns.regplot(data=df,x='temp',y='count',ax=ax1)
ax1.set(title='count of the bikes in temp')
sns.regplot(data=df,x='humidity',y='count',ax=ax2)
ax2.set(title='count of the bikes in humidity')


# In[28]:


from statsmodels.graphics.gofplots  import qqplot
fig, (ax1,ax2)= plt.subplots(ncols=2,figsize=(20,8))
sns.distplot(df['count'],ax=ax1)
ax1.set(title='distribution of count')
qqplot(df['count'],ax=ax2,line='s')
ax2.set(title='therotical quantities')


# In[29]:


df['count']=np.log(df['count'])


# In[30]:


fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(20,6))
sns.distplot(df['count'],ax=ax1)
ax1.set(title='log of distribution of count')
qqplot(df['count'],ax=ax2,line='s')
ax2.set(title='log of therotical quantities')


# In[ ]:





# In[ ]:





# In[ ]:





# In[31]:


corr=df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True,annot_kws={'size':15})


# In[32]:


cols=['season','weekday','weather']


# In[33]:


df_train=pd.get_dummies(df[cols])


# In[34]:


d=df_train


# In[35]:


df=pd.merge(df,d,how='outer',right_on=None,left_index=True, right_index=True)


# In[36]:


pd.set_option('display.max_columns', None)


# In[37]:


df.head()


# In[38]:


corr=df.corr()
plt.figure(figsize=(35,35))
sns.heatmap(corr,annot=True,annot_kws={'size':15})


# In[39]:


df.drop(['atemp','season','weekday','weather'],axis=1,inplace=True)


# In[40]:


cols=df.columns
cols


# In[41]:


df.shape


# # TRAIN_TEST_SPLIT

# In[42]:


from sklearn.model_selection import train_test_split


# In[43]:


x=df.drop('count',axis=1)
y=df['count']


# In[44]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[45]:


x_train.head()


# # EVALUVATION METRICS FOR MODELS

# In[46]:


Testing_R2_Score =[]
Training_MSE = []
Testing_MSE = []
Testing_MAE = []
Residual_sum_of_squares = []
Model = []
cros_val_scr = []


# In[47]:


def prediction(x_train,x_test,y_train,y_test,model,algo):        
        print(f"predictions for {algo} Algorithem")
        Model.append(algo)        
        print("Train Results: ")
        print("================")
        pre = model.predict(x_train)
        print("        Training R2 Score : {:.2f} %".format((metrics.r2_score(y_train,pre))*100))
        print("")
        print("        Training Mean Absolute Error : {:.4f} ".format((metrics.mean_absolute_error(y_train,pre))))
        print("")       
        print("        Training Mean Squared Error : {:.4f} ".format((metrics.mean_squared_error(y_train,pre))))
        print("")
        print("")
        Training_MSE.append(round(metrics.mean_squared_error(y_train,pre),3))
        print("Test Results: ")
        print("================")
        pre_t = model.predict(x_test)
        print("        Testing R2 Score : {:.2f} %".format((metrics.r2_score(y_test,pre_t))*100))
        print("")        
        print("        Testing Mean Absolute Error : {:.4f} ".format((metrics.mean_absolute_error(y_test,pre_t))))
        print("")        
        print("        Testing Mean Squared Error : {:.4f} ".format((metrics.mean_squared_error(y_test,pre_t))))
        print("")
        print("        Residual sum of squares: {:.6f}".format(np.mean(pre_t - y_test) ** 2))
        
        Testing_R2_Score.append(round((metrics.r2_score(y_test,pre_t)*100),2))
        Testing_MSE.append(round(metrics.mean_squared_error(y_test,pre_t),3))
        Residual_sum_of_squares.append(round((np.mean(pre_t - y_test) ** 2),6))
        Testing_MAE.append(round(metrics.mean_absolute_error(y_test,pre_t),3))
        


# In[48]:


color = ["#FF7300","#52D726","#FF0000","gold","#007ED6","#7CDDDD","#007ED6","#FF0000","teal",]
import random
def scatter(x_test,y_test,model,algo):
    global count
    plt.figure(figsize=(15,6))
    plt.style.use("fivethirtyeight")
    plt.scatter(y_test,model.predict(x_test),c=random.choice(color),s=80)
    plt.title(f"Prediction for {algo} Model")
    plt.xlabel('Y Test')
    plt.ylabel('Predicted Y')
    plt.show()


# In[49]:


def plot(x_test,y_test,model,algo):
    plt.figure(figsize=(14,6))
    plt.style.use("seaborn-bright")
    plt.plot(y_test,label ='Test')
    plt.plot(model.predict(x_test), label = 'prediction')
    plt.title(f"Comparision between Prediction and Test data by {algo} Model")
    plt.legend()
    plt.show()


# In[50]:


def cvs(x,y,model,algo):
    cv = cross_val_score(model,x,y,cv=10)
    print(f"Cross validation Score for {algo} : {round(cv.mean()*100,2)} %")
    cros_val_scr.append(round((cv.mean()*100),2))


# # LinearRegression

# In[51]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
algo='LinearRegression'


# In[52]:


prediction(x_train,x_test,y_train,y_test,lr,algo)


# In[ ]:





# In[53]:


cvs(x,y,lr,algo)


# In[54]:


scatter(x_test,y_test,lr,algo)


# # RandomForestRegressor

# In[55]:


rfg=RandomForestRegressor()
rfg.fit(x_train,y_train)
algo='RandomForestRegressor'


# In[56]:


prediction(x_train,x_test,y_train,y_test,rfg,algo)


# In[57]:


cvs(x,y,rfg,algo)


# In[58]:


scatter(x_test,y_test,rfg,algo)


# # AdaBoostRegressor

# In[59]:


adr=AdaBoostRegressor()
adr.fit(x_train,y_train)
algo='AdaBoostRegressor'


# In[60]:


prediction(x_train,x_test,y_train,y_test,adr,algo)


# In[61]:


cvs(x,y,adr,algo)


# In[62]:


scatter(x_test,y_test,adr,algo)


# # GradientBoostingRegressor

# In[63]:


gbr=GradientBoostingRegressor()
gbr.fit(x_train,y_train)
algo='GradientBoostingRegressor'
prediction(x_train,x_test,y_train,y_test,gbr,algo)


# In[64]:


cvs(x,y,gbr,algo)


# In[65]:


scatter(x_test,y_test,gbr,algo)


# # DecisionTreeRegressor

# In[66]:


from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor()
dtr.fit(x_train,y_train)
algo='DecisionTreeRegressor'
prediction(x_train,x_test,y_train,y_test,dtr,algo)


# In[67]:


cvs(x,y,dtr,algo)


# In[68]:


scatter(x_test,y_test,dtr,algo)


# # Ridge REGRESSOR

# In[69]:


rige=Ridge()
rige.fit(x_train,y_train)
algo='Ridge'
prediction(x_train,x_test,y_train,y_test,rige,algo)


# In[70]:


cvs(x,y,rige,algo)


# In[71]:


scatter(x_test,y_test,rige,algo)


# # LASSO REGRESSOR

# In[72]:


las=LassoCV()
las.fit(x_train,y_train)
algo='lasso'
prediction(x_train,x_test,y_train,y_test,las,algo)


# In[73]:


cvs(x,y,rige,algo)


# In[74]:


scatter(x_test,y_test,rige,algo)


# # CREATING NEW TABLE USING METRICS

# In[75]:


results = {"Model":Model,"Testing_R2_Score":Testing_R2_Score,"Training_MSE":Training_MSE,"Testing_MSE":Testing_MSE,
           "Testing_MAE":Testing_MAE,"Residual_sum_of_squares":Residual_sum_of_squares,"cros_val_scr":cros_val_scr}

df1 = pd.concat([pd.Series(v,name=k) for k,v in results.items()],axis=1)
df1.set_index("Model",inplace=True)
df1


# In[76]:


res = []
for i in Testing_R2_Score:
    val = round(i/np.sum(Testing_R2_Score),4)
    res.append(val)

label = [i for i in Model]
color = ["#FF7300","chocolate","#FF0000","gold","#007ED6","#7CDDDD","lightpink","#52D726","salmon"]
plt.figure(figsize=(12,7))
plt.style.use("fivethirtyeight")

plt.pie(res,labels=label,autopct="%1.1f%%",explode=None,colors=color,shadow=True,wedgeprops={"edgecolor":"black"})
plt.title("Testing_R2_Score")
plt.show()


# # CONCLUSION

# ##                  AdaBoostRegressor perform well

# In[77]:


final_model=AdaBoostRegressor()
final_model.fit(x,y)
algo='AdaBoostRegressor'


# In[78]:


import pickle
file=open('AdaBoostRegressor.pkl','wb')
pickle.dump(final_model,file)


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[3]:


print(data_info.loc['revol_util']['Description'])


# In[4]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[5]:


feat_info('mort_acc')


#  **Importing the data and other packages**

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


df = pd.read_csv('../DATA/lending_club_loan_two.csv')


# In[8]:


df.info()


# # Exploratory Data Analysis

#     Creating a count plot of loan status 

# In[9]:


sns.countplot("loan_status",data=df)


#     Creating a histogram of the loan amount

# In[10]:


sns.histplot(x="loan_amnt",data=df,bins=30)


#     Creating a correlation matrix 

# In[11]:


df.corr()


#     Visualizing the correlation matrix using the heatmap

# In[12]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,cmap='viridis')


#     From the correlation matrix we can understand that the installments and the loan maount are highly correlated

#     Creating the scatterplot between installments and the loan amount  

# In[13]:


sns.scatterplot(x="installment",y="loan_amnt",data=df)


#     Creating a box plot to understand the relationship between the loan status and the loan amount

# In[14]:


sns.boxplot(x="loan_status",y="loan_amnt",data=df)


#     Summary statistics for the loan amount grouped by loan status 

# In[15]:


df.groupby("loan_status")["loan_amnt"].describe()


#     Exploring the grade and sub grade columns

# In[16]:


sorted(df["grade"].unique())


# In[17]:


sorted(df["sub_grade"].unique())


#     Creating a count plot per grade differentiated by laon status 

# In[18]:


sns.countplot(x="grade",data=df,hue="loan_status")


#     Creating the countplot per subgrade

# In[19]:


plt.figure(figsize=(12,4))
subgrade_order= sorted(df["sub_grade"].unique())
sns.countplot(x="sub_grade",palette="coolwarm",order=subgrade_order,data=df)


#     From the above plot we can infer that the F and G are paid back often 

#     Creating a countplot for F and G alone

# In[20]:


f_and_g=df[(df["grade"]=="G") | (df["grade"]=="F")]
plt.figure(figsize=(12,4))
subgrade_order= sorted(f_and_g["sub_grade"].unique())
sns.countplot(x="sub_grade",data=f_and_g,order=subgrade_order,hue="loan_status")


#     Creating a column loan repaid which will contain 1 if loan is repaid and 0 if it's not paid 

# In[21]:


df["loan_status"].unique()


# In[22]:


df["loan_repaid"]=df["loan_status"].map({"Fully Paid":1,"Charged Off": 0})


# In[23]:


df[["loan_repaid","loan_status"]]


#     Creating a bar plot showing the correlation of the numeric feature  and the loan repaid column

# In[24]:


df.corr()["loan_repaid"].sort_values().drop("loan_repaid").plot(kind="bar")


# # Data Preprocessing 

# In[25]:


df.head()


#     Finding the length of the dataframe

# In[26]:


len(df)


#     Checking the dataframe for some missing values 

# In[27]:


df.isnull().sum()


#     Converting the missing values in terms of percentage

# In[28]:


100*df.isnull().sum()/len(df)


#     Number of unique emplyment job titles are there 

# In[29]:


df["emp_title"].value_counts()


#     Since there are too many unique job titles, we'll drop the emp_title column

# In[30]:


df=df.drop("emp_title",axis=1)


#     Creating the countplot of the employee length 

# In[31]:


sorted(df["emp_length"].dropna().unique())


# In[32]:


emp_length_order=['1 year',
                  '< 1 year',
 '2 years',
 '3 years',
 '4 years',
 '5 years',
 '6 years',
 '7 years',
 '8 years',
 '9 years',
 '10+ years'
 ]


# In[33]:


plt.figure(figsize=(12,4))
sns.countplot(x="emp_length",data=df,order=emp_length_order)


#     Count plot of the employee length separated by the loan_status

# In[34]:


plt.figure(figsize=(12,4))
sns.countplot(x="emp_length",hue="loan_status",order=emp_length_order,data=df)


# The percentage of charge off per category 

# In[35]:


emp_co=df[df["loan_status"]=="Charged Off"].groupby("emp_length").count()["loan_status"]
print(emp_co)


# In[36]:


emp_fp=df[df["loan_status"]=="Fully Paid"].groupby("emp_length").count()["loan_status"]
print(emp_fp)


# In[37]:


emp_length=emp_co/emp_fp
print(emp_length)


# In[38]:


emp_length.plot(kind="bar")


#     Charge off is similar across all employee length, hence dropping the emp_length column

# In[39]:


df=df.drop("emp_length",axis=1)


#     Checking the dataframe for missing values column

# In[40]:


df.isnull().sum()


#     Review title and purpose column are almost similar, hence dropping the title column

# In[41]:


df["title"].head()


# In[42]:


df["purpose"].head()


# In[43]:


df=df.drop("title",axis=1)


#     Imputing the missing values for the mort_acc

# In[44]:


df["mort_acc"].value_counts()


#     Finding the correlation betwenn the mort_acc and the other features 

# In[45]:


df.corr()["mort_acc"].sort_values()


#     From the above correlation, we can identify that total_acc and mort_acc is positively correlated 

# In[46]:


total_acc_avg=df.groupby("total_acc").mean()["mort_acc"]


# In[47]:


def fill_mort_acc(total_acc,mort_acc):
    if np.isnan(mort_acc):
        return total_acc
    else:
        return mort_acc


# In[48]:


df["mort_acc"]=df.apply(lambda x : fill_mort_acc(x["total_acc"],x["mort_acc"]),axis=1)


#     Checking for columns with missing values 

# In[49]:


df.isnull().sum()


#     Dropping the missing values in the columns revol_util and the pub rec_bankruptcies 

# In[50]:


df=df.dropna()


# In[51]:


df.isnull().sum()


#     Addressing the categorical variables y converting them into a dummy variable 

#     Checking the columns which has a categorical variable 

# In[52]:


df.select_dtypes("object").columns


#     Convert the term feature into either a 36 or 60 integer numeric data type

# In[53]:


df["term"].value_counts()


# In[54]:


df["term"]=df["term"].apply(lambda term: int(term[:3]))


# Since the subgrade is present, dropping the grade column

# In[55]:


df=df.drop("grade",axis=1)


# In[56]:


sub_grade_dummy=pd.get_dummies(df["sub_grade"],drop_first=True)


# In[57]:


print(sub_grade_dummy)


# In[58]:


df=df.drop("sub_grade",axis=1)


# In[59]:


df=pd.concat([df,sub_grade_dummy],axis=1)


# In[60]:


df.columns


# In[61]:


df.select_dtypes("object").columns


# verification_status, application_type, inital_list_status, purpose into dummy variable and concatenate them with original dataframe 

# In[62]:


verification_status_dummy=pd.get_dummies(df["verification_status"],drop_first=True)
application_type_dummy=pd.get_dummies(df["application_type"],drop_first=True)
initial_list_status_dummy=pd.get_dummies(df["initial_list_status"],drop_first=True)
purpose_dummy=pd.get_dummies(df["purpose"],drop_first=True)


# In[63]:


df=pd.concat([df.drop("verification_status",axis=1),verification_status_dummy],axis=1)


# In[64]:


df=pd.concat([df.drop("application_type",axis=1),application_type_dummy],axis=1)


# In[65]:


df=pd.concat([df.drop("initial_list_status",axis=1),initial_list_status_dummy],axis=1)


# In[66]:


df=pd.concat([df.drop("purpose",axis=1),purpose_dummy],axis=1)


# In[67]:


df.select_dtypes("object").columns


# In[68]:


df.head(2)


#     Convert the home ownership column into the dummy variable 

# In[69]:


df["home_ownership"].value_counts()


# In[70]:


df["home_ownership"]=df["home_ownership"].replace(["ANY","NONE"],"OTHER")


# In[71]:


df.home_ownership.value_counts()


# In[72]:


home_ownership_dummy=pd.get_dummies(df["home_ownership"],drop_first=True)


# In[73]:


df=pd.concat([df.drop("home_ownership",axis=1),home_ownership_dummy],axis=1)


# In[74]:


df.head(2)


# In[75]:


df.select_dtypes("object").columns


#     Address : Extracting the feature column into the zip_code 

# In[76]:


df["zip_code"]=df["address"].apply(lambda address: address[-5:])


# In[77]:


df.zip_code


#     Converting the the zip_code into a dummy variable 

# In[78]:


zip_code_dummy=pd.get_dummies(df["zip_code"],drop_first=True)


# In[79]:


df=pd.concat([df.drop(["zip_code","address"],axis=1),zip_code_dummy],axis=1)


# In[80]:


df.head(2)


#     Issue date tells that when the loan was issued but then this is data leakage

# In[81]:


df=df.drop("issue_d",axis=1)


#     Getting the year from the earliest_cr_line

# In[82]:


df["earliest_cr_year"]=(df["earliest_cr_line"]).apply(lambda a: int(a[-4:]))


# In[83]:


df=df.drop("earliest_cr_line",axis=1)


# In[84]:


df.select_dtypes("object").columns


#       Dropping the loan status since the loan repaid also has the same details

# In[85]:


df=df.drop("loan_status",axis=1)


# Converting the train test split 

# In[86]:


from sklearn.model_selection import train_test_split


# In[87]:


X=df.drop("loan_repaid",axis=1).values


# In[88]:


y=df["loan_repaid"].values


# In[89]:


print(len(df))


# In[90]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)


# In[91]:


from sklearn.preprocessing import MinMaxScaler


# In[92]:


scaler=MinMaxScaler()


# In[93]:


X_train=scaler.fit_transform(X_train)


# In[94]:


X_test=scaler.transform(X_test)


# Creating the model 

# In[95]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense,Dropout


# In[96]:


model=Sequential()
model.add(Dense(78,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(39,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(19,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",optimizer="adam")


# Fitting the model 

# In[97]:


model.fit(x=X_train,y=y_train,epochs=25,batch_size=256,validation_data=(X_test,y_test))


# In[98]:


loss=pd.DataFrame(model.history.history)


# In[99]:


loss.plot()


# Predictions using X_test 

# In[100]:


from sklearn.metrics import classification_report, confusion_matrix


# In[101]:


predictions=model.predict_classes(X_test)


# In[102]:


print(classification_report(y_test,predictions))


# In[103]:


print(confusion_matrix(y_test,predictions))


# Creating a model with early stopping

# In[104]:


from tensorflow.keras.callbacks import EarlyStopping


# In[105]:


early_stop=EarlyStopping(monitor="val_loss",mode="min",verbose=1,patience=25)


# In[106]:


model_early=Sequential()
model_early.add(Dense(78,activation="relu"))
model_early.add(Dropout(0.2))
model_early.add(Dense(39,activation="relu"))
model_early.add(Dropout(0.2))
model_early.add(Dense(19,activation="relu"))
model_early.add(Dropout(0.2))
model_early.add(Dense(1,activation="sigmoid"))

model_early.compile(loss="binary_crossentropy",optimizer="adam")


# In[107]:


model_early.fit(x=X_train,y=y_train,epochs=30,batch_size=256,validation_data=(X_test,y_test),verbose=1,callbacks=[early_stop])


# In[108]:


losses_early=pd.DataFrame(model_early.history.history)


# In[109]:


losses_early.plot()


# In[110]:


predictions_early=model_early.predict_classes(X_test)


# In[111]:


print(classification_report(y_test,predictions_early))


# In[112]:


print(confusion_matrix(y_test,predictions_early))


# In[114]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[115]:


model_early.predict_classes(new_customer.values.reshape(1,78))


# In[116]:


from sklearn.ensemble import RandomForestClassifier


# In[117]:


X1=df.drop("loan_repaid",axis=1)
y1=df["loan_repaid"]


# In[118]:


X_train,X_test,y_train,y_test=train_test_split(X1,y1,test_size=0.3,random_state=101)


# In[119]:


rf=RandomForestClassifier(n_estimators=100)


# In[121]:


rf.fit(X_train,y_train)


# In[122]:


predictions_rf=rf.predict(X_test)


# In[123]:


print(classification_report(y_test,predictions_rf))


# In[124]:


print(confusion_matrix(y_test,predictions_rf))


# In[125]:


from sklearn.svm import SVC


# In[126]:


model_svc=SVC()


# In[127]:


model_svc.fit(X_train,y_train)


# In[ ]:


predictions_svc=model_svc.predict(X_test)


# In[ ]:


print(classification_report(y_test,predictions_svc))


# In[ ]:


print(confusion_matrix(y_test,predictions_svc))


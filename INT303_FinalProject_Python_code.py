# Import packages
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

# Load cleaned data
Data = pd.read_csv('/home/yuxin/home/yuxin/INT303Project/cardio_train_cleaned.csv')

# Explorative data analysis
# Age and Gender
fig = plt.figure(figsize = (18,6))
plt.subplot2grid((1,2),(0,0),colspan=1,rowspan=1)
ax1 = sns.countplot(x='gender', hue='cardio', data = Data, palette="deep")
ax1.set_xticklabels(['female','male'])
ax1.set_xlabel('Gender', size = 14)
plt.subplot2grid((1,2),(0,1),colspan=1,rowspan=1)
ax2 = sns.kdeplot(x='age', hue='cardio', data = Data, palette="deep")
ax2.set_xlabel('Age', size = 14)
plt.suptitle('Figure 1. Distribution of Gender and Age', size = 18)
fig.savefig('/home/yuxin/home/yuxin/INT303Project/gender_bar_age_dist.png')

# Weight, Height and BMI
fig = plt.figure(figsize = (18,10))
plt.subplot2grid((2,2),(0,0),colspan=1,rowspan=1)
ax1 = sns.boxplot(x='cardio', y='height', data = Data, palette='deep')
ax1.set_xlabel('Height',size=14)
ax1.set(ylim=(120, 200))
ax1.set_xticklabels(['Non-cardiovascular','Cardiovascular'])
plt.subplot2grid((2,2),(0,1),colspan=1,rowspan=1)
ax2 = sns.boxplot(x='cardio', y='weight', data = Data, palette='deep')
ax2.set_xlabel('Weight', size=14)
ax2.set(ylim=(0, 150))
ax2.set_xticklabels(['Non-cardiovascular','Cardiovascular'])
plt.subplot2grid((2,2),(1,0),colspan=2,rowspan=1)
ax3 = sns.kdeplot(x='BMI', hue='cardio', data = Data, palette='deep')
ax3.set(xlim=(10,60))
ax3.set_xlabel('Body Mass Index',size = 14)
ax3.set(ylim=(0,0.07))
plt.vlines(18.5,0,0.07,colors='black',linestyles='--',linewidth = 1.5)
plt.vlines(23.9,0,0.07,colors='black',linestyles='--',linewidth = 1.5)
plt.suptitle('Figure 2. Distribution of Height, Weight and BMI', size = 18)
fig.savefig('/home/yuxin/home/yuxin/INT303Project/hw_BMI.png')

# Cholestrol and Glucose level
fig = plt.figure(figsize = (18,6))
plt.subplot2grid((1,2),(0,0),colspan=1,rowspan=1)
ax1 = sns.countplot(x='cholesterol', hue='cardio', data = Data, palette="deep")
ax1.set_xlabel('Cholestrol Level',size=14)
ax1.set_xticklabels(['normal','above normal','well above normal'])
plt.subplot2grid((1,2),(0,1),colspan=1,rowspan=1)
ax2 = sns.countplot(x='gluc', hue='cardio', data = Data, palette="deep")
ax2.set_xlabel('Glucose Level',size=14)
ax2.set_xticklabels(['normal','above normal','well above normal'])
fig.suptitle('Figure 3. Distribution of Cholestrol and Glucose level', size = 18)
fig.savefig('/home/yuxin/home/yuxin/INT303Project/cg_bar.png')

# Blood pressure
fig = plt.figure(figsize = (18,6))
plt.subplot2grid((1,2),(0,0),colspan=1,rowspan=1)
ax1 = sns.kdeplot(x='ap_lo',hue='cardio', data = Data, palette="deep")
ax1.set_xlabel('Diastolic blood pressure',size = 14)
plt.vlines(60,0,0.16,colors='black',linestyles='--',linewidth = 1.5)
plt.vlines(89,0,0.16,colors='black',linestyles='--',linewidth = 1.5)
ax1.set(ylim=(0,0.16))
plt.subplot2grid((1,2),(0,1),colspan=1,rowspan=1)
ax2 = sns.kdeplot(x='ap_hi', hue='cardio', data = Data, palette="deep")
ax2.set_xlabel('Systolic blood pressure',size = 14)
plt.vlines(90,0,0.09,colors='black',linestyles='--',linewidth = 1.5)
plt.vlines(139,0,0.09,colors='black',linestyles='--',linewidth = 1.5)
ax2.set(ylim=(0,0.09))
fig.suptitle('Figure 4. Distribution of Blood Pressure',size = 18)
fig.savefig('/home/yuxin/home/yuxin/INT303Project/blood_density.png')

# Smoke, Alcohol and Exercise
fig = plt.figure(figsize = (18,6))
plt.subplot2grid((1,3),(0,0),colspan=1,rowspan=1)
ax1 = sns.countplot(x='smoke', hue='cardio', data = Data, palette="deep")
ax1.set_xlabel('Smoking',size = 14)
ax1.set_xticklabels(['no','yes'])
plt.subplot2grid((1,3),(0,1),colspan=1,rowspan=1)
ax2 = sns.countplot(x='alco', hue='cardio', data = Data, palette="deep")
ax2.set_xlabel('Alcohol Intake',size = 14)
ax2.set_xticklabels(['no','yes'])
plt.subplot2grid((1,3),(0,2),colspan=1,rowspan=1)
ax3 = sns.countplot(x='active', hue='cardio', data = Data, palette="deep")
ax3.set_xlabel('Physical Activity',size = 14)
ax3.set_xticklabels(['no','yes'])
fig.suptitle('Figure 5. Distribution of Smoking, Alcohol Intake and Physical Activity habit',size = 18)
fig.savefig('/home/yuxin/home/yuxin/INT303Project/SDE_bar.png')

# Predictive data analysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from xgboost import plot_importance
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,roc_curve,roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

X_train, X_test, y_train, y_test = train_test_split(Data.drop(['cardio'],axis=1), Data['cardio'])

# Normalization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
svm_model = SVC()
svm_model.fit(X_train,y_train)
lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
rf_model = RandomForestClassifier()
rf_model.fit(X_train,y_train)
knn_model = KNeighborsClassifier()
knn_model.fit(X_train,y_train)

nn_model = Sequential([
    Dense(256, activation='relu', input_shape=[X_train.shape[1]]),
    Dense(128, activation='relu'),
    Dense(1),
])
nn_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = nn_model.fit(X_train, y_train, validation_split=0.2, epochs=20)
# Feature importance plot
xgb_model = XGBClassifier()
xgb_model.fit(Data.drop(['cardio'],axis=1), Data['cardio'])
fig = plt.figure()
ax = plot_importance(xgb_model,grid=False)
ax.set_title('Figure 6. Feature importance plot',size = 18)
ax.figure.savefig('/home/yuxin/home/yuxin/INT303Project/importance_plot.png')

# Measurement
# ROC curve
fig = plt.figure(figsize = (16,8))
plt.subplot2grid((1,2),(0,0),colspan=1,rowspan=1)
svm_pred = svm_model.predict(X_test)
fpr,tpr,thresh = roc_curve(y_test, svm_pred)
auc = roc_auc_score(y_test, svm_pred)
plt.plot(fpr,tpr,label='SVM: AUC %0.2f' % auc, lw = 1.5)
lr_pred = lr_model.predict(X_test)
fpr,tpr,thresh = roc_curve(y_test,lr_pred)
auc = roc_auc_score(y_test, lr_pred)
plt.plot(fpr,tpr,label='Logistic regression: AUC %0.2f' % auc, lw = 1.5)
dt_pred = dt_model.predict(X_test)
fpr,tpr,thresh = roc_curve(y_test,dt_pred)
auc = roc_auc_score(y_test, dt_pred)
plt.plot(fpr,tpr,label='Decision tree: AUC %0.2f' % auc, lw = 1.5)
rf_pred = rf_model.predict(X_test)
fpr,tpr,thresh = roc_curve(y_test,rf_pred)
auc = roc_auc_score(y_test,rf_pred)
plt.plot(fpr,tpr,label='Random forest: AUC %0.2f' % auc, lw = 1.5)
knn_pred = knn_model.predict(X_test)
fpr,tpr,thresh = roc_curve(y_test,knn_pred)
auc = roc_auc_score(y_test, knn_pred)
plt.plot(fpr,tpr,label='K nearest neighbor: AUC %0.2f' % auc, lw = 1.5)
nn_pred = nn_model.predict(X_test)
fpr,tpr,thresh = roc_curve(y_test,nn_pred)
auc = roc_auc_score(y_test, nn_pred)
plt.plot(fpr,tpr,label='Neural network: AUC %0.2f' % auc, lw = 1.5)
plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating Curve', size = 18)
plt.legend(loc="lower right")

# Confusion matrix
plt.subplot2grid((1,2),(0,1),colspan=1,rowspan=1)
cm = confusion_matrix(y_test, nn_model.predict_classes(X_test))
ax1 = sns.heatmap(cm,annot=True,fmt='.20g',xticklabels = ['healthy people','patients'] , yticklabels = ['healthy people','patients'])
ax1.set_title('Confusion matrix for neural network model',size = 18)
ax1.set_xlabel('predict')
ax1.set_ylabel('true')
plt.suptitle('Figure 7. Models Comparison',size = 18)
fig.savefig('/home/yuxin/home/yuxin/INT303Project/rocs_cm.png')


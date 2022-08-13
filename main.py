import numpy as np
import pandas as pd
import csv

#reading the csv data file
df = pd.read_csv("path/creditcard.csv")
print(df.head())
print(df.info())

#importing all scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#using df.iloc to extract the necessary columns from the dataframe
X = df.iloc[:,df.columns != 'Class']
y = df.Class

#plotting heatmap to understand correlations among different attributes
plt.figure(figsize = (38,16))
sns.heatmap(df.corr(), annot=True)
plt.title('Attributes correlation heatmap',fontsize = 30)
plt.show()

#splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 42,stratify = y)

#scaling the training and testing data for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#having an idea of fraud vs non-fraud transactions
count0 = count1 = 0
y = y_test.copy()
y = y.reset_index(drop=True)
for i in range(0,len(y)):
    if y[i] == 1:
        count1+=1
    else:
        count0+=1

print(count0)
print(count1)

#applying logistic regression model to the training data and predicting using testing data
lr = LogisticRegression(class_weight = 'balanced')
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

#accuracy of training
train_acc_lr = lr.score(X_train_scaled, y_train)
print("The Accuracy for Training Set is {}".format(train_acc_lr*100))

#accuracy of prediction
test_acc_lr = accuracy_score(y_test, y_pred_lr)
print("The Accuracy for Test Set is {}".format(test_acc_lr*100))

#plotting the confusion matrix to give us a visual understanding of metrics
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize = (12,6))
sns.heatmap(cm_lr, annot = True, fmt = 'd', cmap = 'Blues')
plt.ylabel("True values", fontsize = 15)
plt.xlabel("Predicted values", fontsize = 15)
plt.show()

#logistic regression classification report
cr_lr = classification_report(y_test, y_pred_lr)
print(cr_lr)

#applying random forest to training data, hyperparameter tuning through RandomizedSearchCV
param_dict = {"n_estimators": [50,150], "max_depth": [2,16]}
rf = RandomForestClassifier()
rf_cv = RandomizedSearchCV(rf, param_dict, scoring = 'recall', refit = 'recall', n_jobs = -1)
rf.fit(X_train, y_train)

#HalvingRandomSearchCV does not run on parallel cores, does not accept multiple scoring parameters

#displaying best results from hyperparameter tuning
cv_results = pd.DataFrame(rf_cv.cv_results_)
best_model_results = cv_results.loc[rf_cv.best_index_]

print(best_model_results)

print("Tuned RF parameters: {}".format(rf_cv.best_params_))
print("Best score is {}".format(rf_cv.best_score_))

#predicting using testing data
y_pred_rf = rf_cv.predict(X_test)

#accuracy of training
train_acc_rf = rf_cv.score(X_train, y_train)
print("The Accuracy for Training Set is {}".format(train_acc_rf*100))

#accuracy of prediction
test_acc_rf = accuracy_score(y_test, y_pred_rf)
print("The Accuracy for Test Set is {}".format(test_acc_rf*100))

#plotting the confusion matrix to give us a visual understanding of metrics
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize = (12,6))
sns.heatmap(cm_rf, annot = True, fmt = 'd', cmap = 'Blues')
plt.ylabel("True values", fontsize = 15)
plt.xlabel("Predicted values", fontsize = 15)
plt.show()

#random forest classification report
cr_rf = classification_report(y_test, y_pred_rf)
print(cr_rf)

#Danielle Kuehler
#ITP 449 Summer 2020
#Final Project
#Q2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt

#Question two- Acceptors and rejectors with decision tree
#Load dataframe from UniversalBank.csv
bankData = pd.read_csv("UniversalBank.csv")
pd.set_option("display.max_columns", None) #Display all columns

#A. What is the target variable?
print("A. The target variable is \"Personal Loan\" (Accept or Reject)")

#B. Ignore the variables row and zip code
bankData = bankData.drop(["Row", "ZIP Code"], axis=1) #Drop both rows

#C.	Partition the data 70/30. Random_state = 2019, stratify=y
X = bankData.drop(["Personal Loan"], axis=1) #Feature matrix- all columns except target variable
y = bankData["Personal Loan"] #Target varaible

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size = 0.3, random_state = 2019, stratify = y)

#D. How many cases in the training partition represented people who accepted offer of a personal loan?
print("D.", y_train.sum(), "out of", y_train.count(), "people accepted the offer of a personal loan") #Sum all instances of a 1

#E.	Plot the classification tree. Use entropy criterion. Max_depth = 5, random_state = 2019
#Minimize entropy at every partition of dataset
dt = DecisionTreeClassifier(criterion = "entropy", max_depth = 5, random_state = 2019)#Won't go deeper than 5 levels
dt.fit(X_train, y_train) #Fit to training variables
y_pred = dt.predict(X_test)
y_pred2 = dt.predict(X_train) #Predict for unseen data

#Labels
fn = X.columns
cn = str(dt.classes_.tolist()) #For 0, 1 values

#Plot with plot tree
plt.figure(figsize=(18,18)) #Create figure
plt.suptitle("E. Classification Tree\nKuehler_Danielle_FinalProject_Q2")
tree.plot_tree(dt, feature_names=fn, class_names=["0","1"], filled=True) #filled is for colors
plt.show() #Display

#Crosstab of actual versus predicted
cf = metrics.confusion_matrix(y_test,y_pred)
print("Confusion matrix:\n",cf)
result = pd.crosstab(y_train, y_pred2, rownames=["Actual"], colnames=["Predictions"])
print("Crosstab:\n", result)

#F.	On the training partition, how many acceptors did the model classify as non-acceptors?
print("F. On training partition, the model classified", result.iloc[1,0], "acceptors as non-acceptors")
#G.	On the training partition, how many non-acceptors did the model classify as acceptors?
print("G. On training partition, the model classified", result.iloc[0,1], "non-acceptors as acceptors")
#H.	What was the accuracy on the training partition?
print("H. Accuracy on training partition:", dt.score(X_train,y_train))
#I.	What was the accuracy on the test partition?
print("I. Accuracy on test partition:", dt.score(X_test,y_test))



#Danielle Kuehler
#ITP 449 Summer 2020
#Final Project
#Q3

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

#Question 3- build a classification model that predicts the edibility of mushrooms

#Read in dataset and store in dataframe
shroomData = pd.read_csv("mushrooms.csv")
pd.set_option("display.max_columns", None)

#Split data into feature matrix and target variable
X = shroomData.iloc[:,1:] #Attributes
y = shroomData.iloc[:,0] #Toxic or edible- not dummy variable
X_dum = pd.get_dummies(X, columns=X.columns) #Change categorical variables to numerical using dummy

#Build a classification tree. Random_state = 2019. Training partition 0.7. stratify = y, max_depth = 6
X_train, X_test, y_train, y_test = \
    train_test_split(X_dum, y, test_size = 0.3, random_state = 2019, stratify = y)

dt = DecisionTreeClassifier(max_depth = 6, random_state = 2019)#Won't go deeper than 6 levels
dt.fit(X_train, y_train) #Fit to training variables
y_pred2 = dt.predict(X_train)
y_pred = dt.predict(X_test) #Predict for unseen data
labels = dt.classes_.tolist() #Turn classes into list

#A.	Print the confusion matrix. Also visualize the confusion matrix using plot_confusion_matrix from sklearn.metrics
cm = metrics.confusion_matrix(y_test, y_pred) #Confusion matrix
cmDf = pd.DataFrame(cm, index=labels, columns=labels) #Confusion matrix to dataframe in order to display labels
print("A. Confusion Matrix:\n",cmDf) #Display

#Visualize confusion matrix
plot_confusion_matrix(dt, X_test, y_test)
plt.suptitle("A. Confusion Matrix\nKuehler_Danielle_FinalProject_Q3")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

#B.	What was the accuracy on the training partition?
print("\nB. Accuracy on training partition:", dt.score(X_train, y_train))

#C.	What was the accuracy on the test partition?
print("\nC. Accuracy on test partition:", dt.score(X_test, y_test))

#D.	Show the classification tree.
#Labels
fn = X_dum.columns

#Plot with plot tree
plt.figure(figsize=(18,18))
tree.plot_tree(dt, feature_names=fn, class_names=labels, filled=True) #filled is for colors
plt.suptitle("D. Classification Tree\nKuehler_Danielle_FinalProject_Q3")
plt.show()

#E.	List the top three most important features in your decision tree for determining toxicity
feat_importances = pd.Series(dt.feature_importances_, index=X_dum.columns) #Determine importance
top3 = feat_importances.sort_values(ascending=False).head(5) #Sort to top three
print("E. Most important features in determining toxicity: Odor, stalk root, and spore print color:\n", top3)

#F. Classify the following mushroom
mushDict = {"cap-shape": ["x"], "cap-surface" : ["s"],"cap-color": ["n"],"bruises": ["t"],"odor": ["y"],"gill-attachment": ["f"],\
            "gill-spacing": ["c"],"gill-size": ["n"],"gill-color": ["k"],"stalk-shape": ["e"],"stalk-root": ["e"],"stalk-surface-above-ring": ["s"],\
            "stalk-surface-below-ring": ["s"],"stalk-color-above-ring": ["w"],"stalk-color-below-ring": ["w"],"veil-type": ["p"],"veil-color": ["w"],\
            "ring-number": ["o"],"ring-type": ["p"],"spore-print-color": ["r"],"population": ["s"],"habitat": ["u"]}
#Convert dict to dataframe
mushDf = pd.DataFrame(mushDict)
#Concat to original dataframe
mushDf2 = pd.concat([X, mushDf], ignore_index=True)
#Convert dataframe to dummy variables
mush_dum = pd.get_dummies(mushDf2, columns=mushDf2.columns)
#Predict on last row
mushPred = dt.predict(mush_dum.iloc[[8124]])
print("F. The following mushroom is:", mushPred, "- or poisonous")




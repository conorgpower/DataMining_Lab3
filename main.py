from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Step 1
# Q. Import the Iris Dataset from SciKitLearn.
# A. 
def importData():
    iris = datasets.load_iris()
    return iris

# Step 2
# Q. Split information from the dataset into Train, Test, Validation subset.
# A. 
def splitData(data):
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
    X_test, X_validation, Y_test, Y_validation = train_test_split(X_test, Y_test, test_size = 0.51)
    return X_test, X_train, X_validation, Y_test, Y_train, Y_validation
# First the information is split into X for the input data and Y for the result classes
# of the input data. The X and Y are split 50/50 into the train and test. The resulting
# test values are then split again 50/50 into test and validation. This gives us the 
# original information now split into train/test/validation in a 50/25/25 ratio.

# Step 3
# Q. Ensure the subsets are Independent and Representative of the original dataset.
# A.
def subsetAnalysis(data, X_test, X_train, X_validation, Y_test, Y_train, Y_validation):
    print("Is length of subsets equal to original dataset: ", 
        (len(X_test) + len(X_train) + len(X_validation) == len(data.data)) and 
        (len(Y_test) + len(Y_train) + len(Y_validation) == len(data.target)))
    ratioRepresentitiveAnalysis(data.data)
    ratioRepresentitiveAnalysis(X_test)
    ratioRepresentitiveAnalysis(X_train)
    ratioRepresentitiveAnalysis(X_validation)
# By checking the lengths of the labels we can see that the divided subsets added back
# together are equivalent of the original dataset. The train test split method by from
# the previous step ensures by default that all data is randomly split. Upon analysis
# of the bar charts that represent the ratio distribution of the different X data sets
# we can see that the data is representetive of the original dataset.

# Step 4
# Q. Build the first classifier for the problem.
# A.
def firstClassifier(data, X_train, Y_train, X_test):
    cl = LinearSVC(loss="hinge", multi_class="ovr", tol=0.5)
    cl.fit(X_train, Y_train)
    prediction = cl.predict(X_test)
    print("LinearSVC Prediction:\n", prediction)
    return prediction
# The first classifier I chose to build was the Linear Support Vector Classification 
# (LinearSVC).

# Step 5
def secondClassifier(data, X_train, Y_train, X_test):
    clf = DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    print("Decision Tree Classifier Prediction:\n", prediction)
    return prediction
# The second classifier I chose to build was using the Decision Tree Classifier.


# Step 6
def thirdClassifier(data, X_train, Y_train, X_test):
    clf = LogisticRegression(multi_class="ovr", tol=0.5, solver="liblinear")
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    print("Logistic Regression Prediction:\n", prediction)
    return prediction
# The third and final classifier I chose to build was using Logistic Regression.

def ratioRepresentitiveAnalysis(input):
    one, two, three, four, five, six, seven, eight = 0, 0, 0, 0, 0, 0, 0, 0
    for x in input:
        for y in x:
            if y >= 0 and y <= 1:
                one += 1
            if y > 1 and y <= 2:
                two += 1
            if y > 2 and y <= 3:
                three += 1
            if y > 3 and y <= 4:
                four += 1
            if y > 4 and y <= 5:
                five += 1
            if y > 5 and y <= 6:
                six += 1
            if y > 6 and y <= 7:
                seven += 1
            if y > 7 and y <= 8:
                eight += 1
    bars = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']
    height = [one, two, three, four, five, six, seven, eight]
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=(0.2, 0.4, 0.6, 0.6))
    plt.plot(height)
    plt.show()

# Step 7
# Q. Select the best out of the three classifiers.
# A. 
def classifierAccuracy(Y_test, linearPrediction, decisionTreePrediction, logisticRegPrediction):
    linearPredictionAccuracy = accuracy_score(Y_test, linearPrediction)
    decisionTreePredictionAccuracy = accuracy_score(Y_test, decisionTreePrediction)
    logisticRegPredictionAccuracy = accuracy_score(Y_test, logisticRegPrediction)
    predictorsAccuracy = [linearPredictionAccuracy, decisionTreePredictionAccuracy, logisticRegPredictionAccuracy]
    bestIndex = predictorsAccuracy.index(max(predictorsAccuracy))
    if(bestIndex == 0):
        print("Best Preictor:\nLinear Prediction Accuracy: ", linearPredictionAccuracy)
        return "Linear Predictor"
    elif (bestIndex == 1):
        print("Best Preictor:\nDecision Tree Prediction Accuracy: ", decisionTreePredictionAccuracy)
        return "Decision Tree"
    elif (bestIndex == 2):
        print("Best Preictor:\nLogistic Regression Prediction Accuracy: ", logisticRegPredictionAccuracy)
        return "Logistic Regression"
# Due to the randomness of the data splitting the accuracy scores change each 
# time the script is run. The shuffle attribute can be set to false in the train
# test split method to aid this however to me that does not give a true independent
# and representitive representation of the dataset. The above method returns the 
# most accurate predictor and by running it a few times with new random subsets I 
# determined that all three predictors were interchangably the most accurate, all
# consistantly returning accuracies of 0.87 to 1.00. 
# 
# Two ways to potentially decide the most accurate predictor for certain would be:
# (1) Run the predictors multiple times on random subsets of the original dataset
# averaging the scores.
# (2) Run the predictor on a larger dataset of the same format.
# 
# Regardless the accuracy score is prone to innaccuracies due to the limited dataset.


# Step 8 
# Q. Report on the future performance of the selected classifier.
# A. 
def futurePerformance(selectedClassifier, X_train, Y_train, X_validation, Y_validation):
    if(selectedClassifier == "Linear Predictor"):
        cl = LinearSVC(loss="hinge", multi_class="ovr", tol=0.5)
        cl.fit(X_train, Y_train)
        prediction = cl.predict(X_validation)
    elif(selectedClassifier == "Decision Tree"):
        clf = DecisionTreeClassifier()
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_validation)
    elif(selectedClassifier == "Logistic Regression"):
        clf = LogisticRegression(multi_class="ovr", tol=0.5, solver="liblinear")
        clf.fit(X_train, Y_train)
        prediction = clf.predict(X_validation)
    prediction = accuracy_score(Y_validation, prediction)
    print("Future Performance: ", prediction)
# This function accepts the best performing predictor from Step 7, and performs a
# prediction based on the validation data to determine future performance of the
# predictor. From running this multiple times, I gathered that the future performace
# of the classifier is approximately the same as the original prediction with only 
# a minute variance each time. Therefore the future performance is approx 0.87 - 1.00.

def main():
    data = importData()
    X_test, X_train, X_validation, Y_test, Y_train, Y_validation = splitData(data)
    subsetAnalysis(data, X_test, X_train, X_validation, Y_test, Y_train, Y_validation)
    linearPrediction = firstClassifier(data, X_train, Y_train, X_test)
    decisionTreePrediction = secondClassifier(data, X_train, Y_train, X_test)
    logisticRegPrediction = thirdClassifier(data, X_train, Y_train, X_test)
    selectedClassifier = classifierAccuracy(Y_test, linearPrediction, decisionTreePrediction, logisticRegPrediction)
    futurePerformance(selectedClassifier, X_train, Y_train, X_validation, Y_validation)

main()
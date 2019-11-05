from sklearn import datasets
from sklearn.model_selection import train_test_split

# Step 1
def importData():
    iris = datasets.load_iris()
    return iris

# Step 2
def splitData(data):
    X = data.data
    Y = data.target
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
    X_test, Y_test, X_validation, Y_validation = train_test_split(X_test, Y_test, test_size = 0.5)
    return X_test, X_train, X_validation, Y_test, Y_train, Y_validation

# Step 3
def subsetAnalysis(X_test, X_train, X_validation, Y_test, Y_train, Y_validation):
    # print(len(X_test), len(X_train), len(X_validation))
    # print(len(Y_test), len(Y_train), len(Y_validation))

def main():
    data = importData()
    X_test, X_train, X_validation, Y_test, Y_train, Y_validation = splitData(data)
    subsetAnalysis(X_test, X_train, X_validation, Y_test, Y_train, Y_validation)

main()
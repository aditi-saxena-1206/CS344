# CS386 - Artificial Intelligence Lab
# Lab Assignment 3 - Classification on IRIS dataset using Gaussian Naive Bayes method
# Date of Assignemnt : November 5, 2020
# Name : Aditi Saxena
# Roll no. : 180010002
#==========================================

#importing libraries from scikit-learn package
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#=========================================
#function to implement classification
def classify():

    #loading the dataset
    iris = datasets.load_iris()

    #exploring the dataset
    print("The feature names and target classes of the dataset are as follows:")
    print("Featues: ", iris.feature_names)
    print("Labels: ", iris.target_names)
    
    #dividing the data into features and target
    X = iris['data']
    Y = iris['target']
    features = iris['feature_names']

    #splitting the training and testing samples
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.7)

    #initializing the GaussianNB model
    model = GaussianNB()

    #fitting the model using the training samples
    model.fit(X_train,Y_train)

    #predicting the target values for test data
    Y_pred = model.predict(X_test)

    #calculating and printing the accuracy of the prediction
    print("Accuracy of prediction = ", accuracy_score(Y_test, Y_pred))

#===========================================
#main function
if __name__ == '__main__':
    print("This program uses Naive Bayes Method to do classification on the IRIS dataset")
    classify()

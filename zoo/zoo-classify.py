# CS386 - Artificial Intelligence Lab
# Lab Assignment 4 - Bonus Question - Classification of Zoo dataset
# Date of Assignemnt : November 13, 2020
# Name : Aditi Saxena
# Roll no. : 180010002
#==========================================

#importing required libraries
import numpy as np
import pandas as pd
import math
import random
from sklearn.metrics import accuracy_score

#==========================================
#function to load the dataset from file "zoo.data"
def load_data():
  col_names = ['animal_name','hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize','type']
  df = pd.read_csv("zoo.data",names = col_names)
  print(df)
  #print(df.describe())
  return df

#==========================================
#function to analyse the given dataset
def describe_data(X,Y,features,classes):
  print("Total number of observation: ", X.shape[0])
  print("Total number of features: ", X.shape[1])
  print("Features: ", features)
  print("Labels: ",classes)

#===========================================
#function to split and the data into training and test samples
def split_data(X,Y):
  #we randomly divide the data into the ratio 70:30
  #random.seed(0)  
  train_row_list = random.sample(range(len(X)),k=70)
  test_row_list = [i for i in range(len(X)) if i not in train_row_list]
  
  #dividing the data according to the calculated rows
  X_train = X[train_row_list,:]
  X_test = X[test_row_list,:]
  Y_train = Y[train_row_list]
  Y_test = Y[test_row_list]

  return [X_train,X_test,Y_train,Y_test]
  
#=============================================
#function to calculate and return the prior probabilities of each class
def prior_prob(Y_train,n):
  size = Y_train.shape[0]
  prob = np.zeros(n)
  for i in range(n):
    prob[i] = np.sum(Y_train == i)
  prob = prob/size
  print(prob)
  return prob

#===============================================
#function to calculate the likelihood of each feature given the class
def likelihood_calc(X_train,Y_train):
  likelihood = {}
  #calculate and store P(X_i|C) for each feature for each class
  #We calculate the likelihood directly by counting since the feature values are discreet
  
  #counting number of observation of each class
  class_count = {}
  for i in Y_train[:,0]:
    if i not in class_count.keys():
      class_count[i] = 1
    else:
      class_count[i] += 1
  #print(class_count)
  #print(X_train)
  for i in range(len(class_count.keys())):
    likelihood[i] = {}
    for j in range(X_train.shape[1]):
      likelihood[i][j] = {}
      x_feature = X_train[:,j]
      temp_array = x_feature[(Y_train[:,0]==i)]
      #print(temp_array)
      #storing number of instance of each value of a feature
      unique, counts = np.unique(temp_array, return_counts=True)
      
      #calculating probability from number of instances
      
      #applying Laplace correction to eliminate 0 probability of some features
      counts = counts + 0.1
      if (j == 12):
        counts = counts/(class_count[i]+0.6)
      else:
        counts = counts/(class_count[i]+0.2)
      #print(counts)
      count_value = dict(zip(unique, counts))
      #print(count_value.keys())
      
      #adding missing keys
      if (j==12):
        if (len(count_value.keys())<6):
          t = sum(count_value.values())
          if 0 not in count_value.keys():
            count_value[0] = (1-t)/(6-len(count_value.keys()))
          if 2 not in count_value.keys():
            count_value[2] = (1-t)/(6-len(count_value.keys()))
          if 4 not in count_value.keys():
            count_value[4] = (1-t)/(6-len(count_value.keys()))
          if 5 not in count_value.keys():
            count_value[5] = (1-t)/(6-len(count_value.keys()))
          if 6 not in count_value.keys():
            count_value[6] = (1-t)/(6-len(count_value.keys()))
          if 8 not in count_value.keys():
            count_value[8] = (1-t)/(6-len(count_value.keys()))
      else:
        if (len(count_value.keys())==1):
          t = sum(count_value.values())
          if 0 not in count_value.keys():
            count_value[0] = 1-t
          if 1 not in count_value.keys():
            count_value[1] = 1-t
      #print(count_value)
      likelihood[i][j] = count_value
  #print(likelihood)
  return likelihood
  
#==============================================
#function to calculate the joint probabilities for each feature of a test sample
def joint_probabilities(feature_list, prior, likelihood):
  joint_prob = np.zeros(len(prior))
  for i in range(len(prior)):
    joint = 1
    for j in range(len(feature_list)):
      #print(i,"  ",j,"  ","  ",feature_list[j])
      joint = joint*likelihood[i][j][feature_list[j]]
    joint_prob[i] = joint * prior[i]
  return joint_prob

#==============================================
#funtion to find the class with maximum probability
def predict(posterior):
  return np.argmax(posterior)

#==============================================
#function to predict classes of input test samples
def predict_test(X_test, prior, likelihood):
  size = X_test.shape[0]
  Y_pred = np.zeros(size, dtype=int)
  for i in range(size):
    features = X_test[i,:]
    posterior = joint_probabilities(features,prior,likelihood)
    #print(posterior)
    Y_pred[i] = predict(posterior)
  return Y_pred
  
#===============================================
#main function
def main():
  print("Loading ZOO dataset from file 'zoo.data'...")
  dataset = load_data()
  print("Dataset Loaded.")

  print("Description of data")
  #separating features and labels
  X_df = dataset.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
  Y_df = dataset.iloc[:,[17]]
  X = X_df.to_numpy()
  Y = Y_df.to_numpy()
  #print(Y)
  #changing class to [0,6] for easier calculation
  Y = np.array([np.array(Y[i]-1) for i in range(len(Y))])
  #print(Y)
  features = dataset.columns[1:17]
  classes = ['Mammal', 'Bird', 'Reptile', 'Fish', 'Amphibian', 'Bug', 'Invertebrate']
  #print(X.shape)
  #print(Y.shape)
  #print(features)
  describe_data(X,Y,features, classes)

  print("Splitting the data in the ratio train:test::70:30")
  splitted_data = split_data(X,Y)
  X_train = splitted_data[0]
  X_test = splitted_data[1]
  Y_train = splitted_data[2]
  Y_test = splitted_data[3]
  print("Data splitted.")

  print("Calculating Prior Probabilities for each class")
  prior_probability = prior_prob(Y_train,len(classes))

  print("Calculating likelihoods")
  likelihood = likelihood_calc(X_train,Y_train)

  print("Predicting for test data...")
  Y_pred = predict_test(X_test,prior_probability, likelihood)
  print("Prediction done")
  print(Y_pred + 1)
  print("Checking Accuracy")
  print("Accuracy Score: ", accuracy_score(Y_test,Y_pred))

if __name__=='__main__':
  main()

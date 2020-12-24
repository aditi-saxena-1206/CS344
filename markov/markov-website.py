# CS386 - Artificial Intelligence Lab
# Lab Assignment 5 - Markov Chain Simulation
# Date of Assignemnt : November 27, 2020
# Name : Aditi Saxena
# Roll no. : 180010002
#==========================================

# importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==========================================
# function to print the normalised probabilities of the states
def print_normal(states,num):
  prob = num/(np.sum(num))
  print(states , ":",prob)

#==========================================
# function to print the Transition Probability Matrix
def print_tpm(tpm,states):
  df = pd.DataFrame(tpm, index=states,columns=states)
  print(df)

#==========================================
# function to plot the values produced after each iteration 
# of the mini-forward algorithm
def plot_graph(x1,x2,x3,i):
  plt.figure(figsize=(25,8))
  plt.plot( [x for x in range(i)], x1, marker='o', markerfacecolor='blue', markersize=8, color='skyblue', linewidth=1,label='Inactive')
  plt.plot( [x for x in range(i)], x2, marker='x', markerfacecolor='red', markersize=8, color='red', linewidth=1,label='Active')
  plt.plot( [x for x in range(i)], x3, marker='^', markerfacecolor='orange', markersize=8, color='green', linewidth=1,label='Super active')
  plt.legend(loc='best')
  plt.xlabel('Years')
  plt.ylabel('Number of customers')
  plt.grid(b=True,which='major',axis='both')
  plt.show()

#===========================================
# function to simulate markov chains using mini-forward algorithm
def mini_forward(tpm,initial):
  next = initial
  x1 = [initial[0]]
  x2 = [initial[1]]
  x3 = [initial[2]]
  for i in range(2,101):
    #print("Iteration",i ,":",end = "")
    mult = np.dot(next,tpm)
    #print(mult)
    x1.append(mult[0])
    x2.append(mult[1])
    x3.append(mult[2])
    if (mult==next).all():
      break
    next = mult

  print("Plotting the transition")
  plot_graph(x1,x2,x3,i)
  return (i,next)

#=============================================
# main driver function
def main():

  # defining the states
  states = ['Inactive','Active','Super Active']
  
  # initial number of users in each state
  initial_no = np.array([188969,81356,14210])
  
  # to input custom initial probabilities of each state
  s = input("Use custom initial state(Y/N)? ")
  if(s == 'Y'):
    inp = np.array(input("Enter initial probabilities: ").split(" "))
    inp = inp.astype(np.float)
    total = np.sum(initial_no)
    total_array = np.array([total]*3)
    initial_no = np.multiply(inp,total_array)

  # print the given values and tpm
  print("Initial state probabilites after normalisation are: ")
  print_normal(states,initial_no)
  
  tpm = np.array([[0.89,0.10,0.01],
                  [0.75,0.22,0.03],
                  [0.49,0.44,0.07]])
  print("Transition Probability Matrix is:")
  print_tpm(tpm,states)

  # executing the mini-forward algorithm
  print("Running the Mini-Forward Algorithm")
  iter, stationary = mini_forward(tpm,initial_no)
  
  # printing the stationary distribution of the states
  print("The stationary distribution is:")
  print_normal(states,stationary)
  print("Number of states for convergence : ",iter)

if __name__ == '__main__':
  main()

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#A complete tutorial of Numpy 100% useful for machine learning


# In[115]:


#To install library
#pip install numpy
# To call into program
import numpy as np 


# In[116]:


#Creating a Vector

#Problem : Create a vector by using numpy
row=np.array([1,5,9])
col=np.array([[2],
             [7],
             [8]])
print("---------------------------------------------------------------")
# To get output use print command
print(row)
print("---------------------------------------------------------------")
print(col)
print("---------------------------------------------------------------")
#Numpy's main data structure is he multi-dimensional array .To create a vector we simply create a one-dimensional array


# In[117]:


# Reshape Arrays
#problem:change shape of matrix without changing elements
x=np.array([[1,3],
              [3,7],
               [5,8]])
print(x)
print("-----------------------------------------------------------")
print(x.reshape(2,3))


# In[118]:


# calculate Deteminant
#problem:find  Deteminant of matrix
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
print(np.linalg.det(x))# .det() is used to calculate determinant of matrix


# In[119]:


#creating a matrix

#Problem:Create a matrix

#solution:

matrix=np.array([[1,3],
              [3,7],
               [5,8]])
print("---------------------------------------------------------------")
print(matrix)
print("---------------------------------------------------------------")
#or 

matrix_1=np.mat([[1,3],
              [3,7],
               [5,8]])
print("---------------------------------------------------------------")
print(matrix_1)
print("---------------------------------------------------------------")


# In[120]:


#Rank of matrix
#rank is dimensions of matrix
#problem:find rank of matrix
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
print(np.linalg.matrix_rank(x))#To find rank of matrix we use .matrix_rank() 
#here matrix is 3D 


# In[121]:


#transposing a vector or matrix
#transpose means swapped row into columns or columns into rows
#problem: find transpose 
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
print(x.T)


# In[122]:


#creating spare matrix

#Problem:Given data with very few nonzeros values ,you want to efficiently represent it..

# solution:Create a sparse matrix
from scipy import sparse
x=np.array([[3,0,5],
           [0,8,0],
           [1,0,6]])
x_sparse=sparse.csr_matrix(x)
print(x_sparse)


# In[123]:


#       (0[0],0[1],0[2]
#       1[0],1[1],1[2]
#       2[0],2[1],2[2])


# Here in (0,0) we get 3.
# means in row 0 and column's element 0 we have value 3.
# In (1,1) we get 8
# Here row 2 with column's element 1 we get 8


# In[124]:


# SELECTING ELEMENTS
# PROBLEM: TO SELECT ONE OR MORE ELEMENTS IN MATRIX
# To create row vector
nv=np.array([1,6,5,3,8,9,11,24,33,23])
#to create a matrix 
nv_1=np.array([[1,3,4],
              [3,7,9],
               [5,8,21]])
print("---------------------------------------------------------------")
print(nv[1])# to select vector with position 1 ([0,1,,2,3,4,5,6,7,8,9,10])
print("---------------------------------------------------------------")
print(nv_1[2,1])# to select matrix element (2,1) i.e row 2 with element of colmun 1 
                                                        #       (0[0],0[1],0[2]
                                                        #       1[0],1[1],1[2]
                                                        #       2[0],2[1],2[2])
print("---------------------------------------------------------------")


# In[125]:


# Describing a matrix
# problem: find shape ,size and dimensions
nv=np.array([[1,3,4],
              [3,7,9],
               [5,8,21]])
print("---------------------------------------------------------------")
print(nv.shape)# to get shape we use .shape 
print("---------------------------------------------------------------")
print(nv.ndim)# to get dimensions we use .ndim
print("---------------------------------------------------------------")
print(nv.size)#to get Size we use .size
print("---------------------------------------------------------------")


# In[126]:


#(3,3) means 3 rows with 3 columns


# In[127]:


#Applying operation to elements
# problem:apply some operation to elements
nv=np.array([[1,3,4],
              [3,7,9],
               [5,8,21]])
print("---------------------------------------------------------------")
print(nv+100)# add 100 to matrix
print("---------------------------------------------------------------")
print(nv-1)#subtract by1
print("---------------------------------------------------------------")
print(nv*2)#multiply by 2
print("---------------------------------------------------------------")
print(nv/2)#divide by 2
print("---------------------------------------------------------------")
print(nv%2)#mod by 2
print("---------------------------------------------------------------")


# In[128]:


#finding min and max values
#problem:find min and max values
x=np.array([[1,3,4],
              [3,7,9],
               [5,8,21]])
print("---------------------------------------------------------------")
print(x.min())
print("---------------------------------------------------------------")
print(x.max())
print("---------------------------------------------------------------")
print(np.max(x,axis=0))# find max element in each columns
print("---------------------------------------------------------------")
print(np.min(x,axis=0))# find min element in each columns
print("---------------------------------------------------------------")
print(np.max(x,axis=1))# find max element in each row
print("---------------------------------------------------------------")
print(np.min(x,axis=1))# find min element in each row
print("---------------------------------------------------------------")


# In[129]:


#calculate the average ,varience  and standard diversion
#problem:calculate statistics about array
x=np.array([[1,3,4],
              [3,7,9],
               [5,8,21]])
print("---------------------------------------------------------------")
print(x.mean())# use mean()to calculate average.
print("---------------------------------------------------------------")
print(x.var())# use .var() to calculate varience
print("---------------------------------------------------------------")
print(x.std())# use .std() to calculate standard diversion
print("---------------------------------------------------------------")


# In[130]:


# Generating random values 
#problem:Generate pesudo-random value
#to set seed
np.random.seed(0)
#Generate four random floats 
x=np.random.random(4)
#Generate four random int between 1,20
y=np.random.randint(1,20,4)
print(x)# to get random floats of seed 0
print("---------------------------------------------------------------")
print(y)# to get output of  four random int between 1,20
print("---------------------------------------------------------------")
x_1=np.random.normal(0.0,1.0,5)#Draw  five number from normal distribution with mean 0.0 and standard deviation of 1.0
print(x_1)
print("---------------------------------------------------------------")
x_2=np.random.logistic(0.0,1.0,5)#Draw  five number from logistic distribution with mean 0.0 and scale of 1.0
print(x_2)
print("---------------------------------------------------------------")
x_3=np.random.uniform(1.0,2.0,4)#Draw four number greater than r equal to1.0 and less than 2.0
print(x_3)


# In[131]:


# inverting a matrix
#problem:calculate the inverse of square matrix
x=np.array([[3,5],[5,9]])
print(np.linalg.inv(x))# we use .linalg to calculate invere.
#inverse of square matrix ,A is second matrix A' such that
#  A*A'=I
#where I is identity matrix
print("---------------------------------------------------------------")
print(x@np.linalg.inv(x))


# In[132]:


#multiply matrices
#problem:multiply two matrices
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
y=np.array([[2,6,4],
              [3,2,2],
               [3,6,1]])
print((x*y))# to get multiplication of matrices


# In[133]:


#addition and subtraction of matrices
#problem:calculate addition and subtraction of  two matrices
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
y=np.array([[2,6,4],
              [3,2,2],
               [3,6,1]])
print(np.add(x,y))
print("---------------------------------------------------------------")
print(np.subtract(x,y))


# In[134]:


#calculate Dot product
#problem:find dot product of two matrices
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
y=np.array([[2,6,4],
              [3,2,2],
               [3,6,1]])
print(np.dot(x,y))


print("---------------------------------------------------------------")
#problem:find dot product of two vectors
x=np.array([1,2,3])
y=np.array([2,6,4])
print(np.dot(x,y))



# In[135]:


# Eigenvalues and Eigenvectors
#problem:find eigenvalues and eigenvectors of square matrix
#use linalg.eig
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
x_eigenvalue,x_eigenvector=np.linalg.eig(x)
print(x_eigenvalue)
print("---------------------------------------------")
print(x_eigenvector)
#eigenvalues and eigenvectors are used in machine learning libraries


# In[136]:


#calculating the trace of matrix
#the trace of matrix is sum of diagonal of elements of matrix.
#problem:calculate trace of matrix
x=np.array([[1,2,3],
              [3,7,9],
               [5,8,2]])
print(x.diagonal())#return Diagonal values
print("---------------------------------------------")
print(x.diagonal(offset=1))#return Diagonal values one above the main Diagonal
print("---------------------------------------------")
print(x.diagonal(offset=0))#return Diagonal values one below the main Diagonal
print("---------------------------------------------")
print(x.trace())# sum of diagonal elements


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





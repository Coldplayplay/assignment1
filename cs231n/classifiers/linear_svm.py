import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.             (3073, 10)
  - X: A numpy array of shape (N, D) containing a minibatch of data. (500,3073)
  - y: A numpy array of shape (N,) containing training labels;
      y[i] = c means that X[i] has label c, where 0 <= c < C.        (500,)
  - reg: (float) regularization strength

                                                                                       Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  #为什么W要转置
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:#这样写很有用
        loss += margin
        dW.T[j] += X[i]
        dW.T[y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg *  2 * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
 
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  num_train = X.shape[0]
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  Scores = X.dot(W)  
  #print('Scores shape:',Scores.shape)
  #print('y_label shape:',Scores[(range(0, num_train),y)].shape)
  Margins = np.maximum(0, Scores.T - Scores[(range(0, num_train),y)] +1)
  #print('Margins shape:',Margins.shape)
  
  Margins.T[(range(0, num_train),y)] = 0
  
  loss = np.sum(Margins)/num_train + reg * np.sum(W * W)

  temp = np.zeros(Scores.shape)
  temp[(range(0, num_train), y)] = np.sum((Margins>0).T, axis = 1)
  dW = ((Margins>0).dot(X) - (temp.T).dot(X)).T 
  dW /= num_train
  dW += reg *  2 * W

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  


  return loss, dW

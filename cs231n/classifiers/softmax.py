import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        scores_exp = np.exp(scores)
        scores_p = scores_exp/np.sum(scores_exp)
        loss += -np.log(scores_p[y[i]])
        
        dW.T[y[i]] -= X[i]
        for j in xrange(num_classes):
            dW.T[j] += X[i]*scores_exp[j]/np.sum(scores_exp)
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
    
  Scores = X.dot(W)
  Scores = (Scores.T - np.amax(Scores, axis=1)).T
  Scores_exp = np.exp(Scores)
  Scores_p = (Scores_exp.T/np.sum(Scores_exp, axis=1)).T
  
  Scores_p_labels = Scores_p[(range(0, num_train), y)]
  loss = np.sum(-np.log(Scores_p_labels))/num_train + reg * np.sum(W*W)
  
  temp = np.zeros(Scores.shape)
  temp[(range(0, num_train), y)] = 1
  #print('Scores_exp shape',Scores_exp.shape)
  dW = -(X.T).dot(temp) +  ((X.T)/np.sum(Scores_exp, axis=1)).dot(Scores_exp)
  dW /= num_train
  dW += reg * 2 * W
    
    
  #############################################################################

  return loss, dW


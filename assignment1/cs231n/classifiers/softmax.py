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
  #############################################################################
  num_cls = W.shape[1]
  
  num_train = X.shape[0]
  
  scores = np.dot(X,W)
  
  for ii in range(num_train):
      c_scr = scores[ii, :]
      
      sft_scores = c_scr - np.max(c_scr)
      
      loss_ii = -sft_scores[y[ii]] + np.log(np.sum(np.exp(sft_scores)))
      loss += loss_ii
      
      for jj in range(num_cls):
          
          sftmx_score = np.exp(sft_scores[jj]) / np.sum(np.exp(sft_scores))
          
          if jj == y[ii]:
              dW[:, jj] += (-1 + sftmx_score)*X[ii]
          else:
              dW[:, jj] += sftmx_score * X[ii]
              
  loss /= num_train
  loss += reg * np.sum(W*W)
  
  dW /= num_train
  dW += 2*reg*W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
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
  #############################################################################
  num_train = X.shape[0]
  
  #scores and stability fix
  scores = np.dot(X,W)
  shft_scores = scores - np.max(scores, axis = 1)[...,np.newaxis]
  
  #softmax scores
  sftmx_scores = np.exp(shft_scores)/ np.sum(np.exp(shft_scores), axis=1)[..., np.newaxis]
  
  #gradient wrt softmax scores
  dS = sftmx_scores 
  dS[range(num_train),y] = dS[range(num_train),y] - 1
  
  #back prop dS and find dW
  dW = np.dot(X.T, dS)
  dW /= num_train
  dW += 2*reg*W
  
  #cross entropy loss:
  c_cls_scr = np.choose(y,shft_scores.T)
  loss = -c_cls_scr + np.log(np.sum(np.exp(shft_scores),axis=1))
  loss = np.sum(loss)
  
  loss /= num_train
  
  loss += reg * np.sum(W*W)
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


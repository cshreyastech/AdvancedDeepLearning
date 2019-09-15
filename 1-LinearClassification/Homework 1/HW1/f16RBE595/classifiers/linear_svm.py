import numpy as np
from random import shuffle

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.
  Inputs:
  - W: K x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  #print('W.shape: ', W.shape)
  #print('X.shape: ', X.shape)
  #print('y.shape: ', y.shape)
    
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # Compute the loss
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  #print('scores.shape: ', scores.shape)
  
  correct_class_scores = scores[ np.arange(num_train), y].reshape(num_train,1)
  #print('correct_class_scores.shape: ', correct_class_scores.shape)

  margin = np.maximum(0, scores - correct_class_scores + 1)
  margin[ np.arange(num_train), y] = 0 # do not consider correct class in loss
  loss = margin.sum() / num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # Compute gradient
  margin[margin > 0] = 1
  valid_margin_count = margin.sum(axis=1)
  # Subtract in correct class (-s_y)
  margin[np.arange(num_train),y ] -= valid_margin_count
  dW = (X.T).dot(margin) / num_train

  # Regularization gradient
  dW = dW + reg * 2 * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW

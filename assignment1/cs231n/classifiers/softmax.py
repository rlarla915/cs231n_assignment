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
  for i in range(X.shape[0]):
    tmp_score = X[i, :].dot(W)
    tmp_exp_score = np.exp(tmp_score)
    sum_tmp_score = np.sum(tmp_exp_score)
    tmp_deno = np.log(sum_tmp_score)
    tmp_mole = tmp_score[y[i]]
    loss += tmp_deno - tmp_mole
    dW += np.reshape(X[i, :], (-1, 1)).dot(np.reshape(tmp_exp_score, (1, -1))) / sum_tmp_score
    dW[np.arange(W.shape[0]), y[i]] -= X[i, :]
    
  loss /= X.shape[0]
  dW /= X.shape[0]
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
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
  num_class = W.shape[1]
  tmp_score = X.dot(W)
  tmp_exp_score = np.exp(tmp_score)
  sum_tmp_score = np.sum(tmp_exp_score, axis = 1)
  tmp_deno = np.log(sum_tmp_score)
  tmp_mole = tmp_score[np.arange(num_train), y]
  loss += np.sum(tmp_deno - tmp_mole)
  loss /= num_train
  dW += (X.T).dot(tmp_exp_score / sum_tmp_score[:, None])
  tmp_mask = np.zeros((num_train, num_class))
  tmp_mask[np.arange(num_train), y] = 1
  dW -= (X.T).dot(tmp_mask)
  dW /= num_train
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


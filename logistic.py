import torch
from descent import adagrad, ParameterSpace
from math import log


def count_pixel_percent(img, predicate):
    """
    img is an order-2 tensor representing the pixels of a grayscale
    image. Each pixel has an intensity between 0 and 255. 0 == black
    and 255 == white.

    The 'predicate' argument is a function that maps a pixel 
    (i.e. an integer from 0 to 255) to a Boolean value. 
    
    The function returns the percentage of image pixels for 
    which predicate(pixel) returns True.
    
    """
    raise NotImplementedError('Question 1')

    

def count_contrasting_pixels(img, stride, threshold):
    """
    img is an order-2 tensor (i.e. a matrix) representing the pixels
    of a grayscale image. Each pixel has an intensity between 0
    and 255. 0 == black and 255 == white.
    
    This function examines every pair of image pixels (x,y) and 
    (x+stride,y) -- note that stride is a positive integer. 
    
    It returns the percentage of these pairs for are "high contrast."
    In other words, the absolute difference between the intensities
    of pixel1 and pixel2 is greater than or equal to the specified
    threshold.
    
    """
    raise NotImplementedError('Question 1')


class LogisticRegressionModel:
    """
    Code to run a trained logistic regression model.
    
    """    
    
    def __init__(self, theta):
        """
        theta is a torch tensor storing a 2xD parameter matrix.
        
        """
        self.theta = theta
        
    def predict_probs(self, X):
        """
        Given NxD feature matrix X, this returns an N-length vector
        for which the ith element is the probability that the response
        corresponding to the ith feature vector is equal to 1.
        
        """
        raise NotImplementedError('Question 2')

    def classify(self, X, thres=0.5):
        """
        Given NxD feature matrix X, this returns an N-length vector
        for which the ith element is 1 if the probability of a positive 
        response corresponding to the ith feature vector is greater
        than or equal to the specified threshold.
        
        """
        raise NotImplementedError('Question 2')
   
    def evaluate(self, X, y, thres=0.5):
        """
        Given NxD feature matrix X and the N-length response
        vector y, this compares the result of running 
        classify(X, thres) to the response vector y.
        
        It returns the percentage of response values that 
        are equivalent (i.e. both equal 1 or both equal 0).
        
        """
        raise NotImplementedError('Question 2')
        

class LogisticRegressionParameterSpace(ParameterSpace):
    
    def __init__(self, X, y):
        super().__init__(torch.zeros(2, X.shape[1]), 
                         precision=0.0000001, max_steps=5000)  # do not change
        self.X = X
        self.y = y
    
    def gradient(self, theta):
        """Computes the loss gradient at theta.
        
        theta is a 2xD parameter matrix.
        """
        raise NotImplementedError('Question 3')


def train_logistic_regression(X, y):
    """
    Trains a logistic regression model on feature matrix X and 
    response vector y. Both X and y are torch tensors.
    
    The function returns a trained LogisticRegressionModel.
    
    YOU SHOULD NOT CHANGE THIS FUNCTION. Rather, just complete the 
    gradient method of LogisticRegressionParameterSpace (above).
    
    """  
    task = LogisticRegressionParameterSpace(X, y)
    steps = adagrad(0.9, task)
    result = steps[-1]
    return LogisticRegressionModel(result)


def extract_pixel_pattern_features(t, threshold, pixel_pattern):
    """Extracts count-based features for a given "pixel pattern".
        
    It first collects a histogram over pixel patterns.
    For instance, if the pixel pattern is [(0,0), (1,1)], then the feature
    extractor looks at all image pixel pairs (x,y) and (x+1, y+1).
    It counts the patterns t_ij, where i = 0 if pixel (x,y) exceeds the
    threshold and j = 0 if pixel (x+1, y+1) exceeds the threshold.
    
    For instance, for the SWISS_FLAG image, this histogram would be:
       {'p_00': 8, 'p_01': 3, 'p_11': 2, 'p_10': 3}
    since there are 8 pixel pairs that are both black, 2 pixel pairs
    that are both white, and 3 pixel pairs (each) such that (x,y) and 
    (x+1, y+1) are different colors.
    
    Then, we take the log of these counts to make them
    more suitable for additive linear models. IMPORTANT: if a count is
    zero, then just use the value 0 for the feature (not log(0)).
    
    """
    raise NotImplementedError('Question 4')


def precision(model, X, y, thres=0.5):
    """
    Given NxD feature matrix X and the N-length response
    vector y, this compares the result of running 
    model.classify(X, thres) to the response vector y.
    
    It returns the fraction of the time that the logistic
    regression model is correct, when it makes a positive 
    prediction (i.e. predicts that the response variable 
    equals 1).
    
    """
    raise NotImplementedError('Question 5')


def recall(model, X, y, thres=0.5):
    """
    Given NxD feature matrix X and the N-length response
    vector y, this compares the result of running 
    model.classify(X, thres) to the response vector y.
    
    It returns the fraction of the time that the logistic
    regression model is correct, when it is classifying a 
    positive instance (i.e. when the response equals 1).
    
    """
    raise NotImplementedError('Question 5')

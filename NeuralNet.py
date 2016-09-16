import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
from scipy.special import expit
import sklearn.svm
import sklearn
from sklearn import preprocessing
import math

np.set_printoptions(threshold=np.nan)

class NeuralNetwork:
	""" Class definition for single hidden layer neural network """

	def __init__(self, inputSize, hiddenSize, outputSize):
		
		self.inputSize = inputSize
		self.hiddenSize = hiddenSize
		self.outputSize = outputSize
		
		#Define hyperparameters for Neural Network
		self.learningRate = 0.01
		self.mu, self.sigma = 0, 0.01

		#Still need to add bias vector to inputs
		self.W1 = np.random.normal(self.mu, self.sigma, (inputSize+1, hiddenSize))
		self.W2 = np.random.normal(self.mu, self.sigma, (hiddenSize+1, outputSize))

		#Initiate cached values to for backpropogation to None
		self.z2 = None #[1 x hiddenSize]
		self.tanhz2 = None #[1 x hiddenSize]
		self.z3 = None #[1 x outputSize]
		self.output = None #[1 x outputSize]
		self.delta3 = None #[]

	def squareLoss(self, output, y):
		yArr = np.zeros(self.outputSize)
		yArr[y] = 1
		return 0.5*np.sum(np.square(np.subtract(yArr, output)))

	def crossEntropyLoss(self, output, y):
		yArr = np.zeros(self.outputSize)
		yArr[y] = 1
		return -1 * np.sum(np.multiply(y, np.log(output)), np.multiply((1-y), np.log(1-output)))

	def decayLearningRate(self):
		""" Decay the learning rate parameter after a certain number of epochs. """	
		self.learningRate = self.learningRate*0.8

	def sigmoidDeriv(self, x):
		return np.multiply(expit(x), 1-expit(x))

	def forward(self, xRow, W1, W2):
		""" Returns the output of our neural network h(x) while caching values
		as they are computed and propogated through the neural net. """

		xRow = np.reshape(np.append(xRow, 1), (1,self.inputSize+1)) 	
		self.z2 = np.dot(xRow, W1)
		self.tanhz2 = np.tanh(self.z2)
		self.tanhz2 = np.reshape(np.append(self.tanhz2, 1), (1,self.hiddenSize+1))
		self.z3 = np.dot(self.tanhz2, W2)
		output = expit(self.z3)

		return output


	def backpropW2(self, x, y, output):
		""" Do back propogation to determine the gradient dJdW2
			Inputs: x: Some column of the training data
					y: the value of the training data: convert to binary array of size 10
					output: the forwarded prediction of our model
		"""

		yArr = np.zeros(self.outputSize)
		yArr[y] = 1
		difference = np.subtract(output, yArr)
		self.delta3 = np.multiply(difference, self.sigmoidDeriv(self.z3))
		self.delta3 = np.reshape(self.delta3, (1,self.outputSize))
		self.tanhz2 = np.reshape(self.tanhz2, (self.hiddenSize+1,1))
		dJdW2 = 0.5*np.dot(self.tanhz2, self.delta3)

		return dJdW2 


	def backpropW1(self, x, y):
		""" Do back propogation to determine gradient dJdW1
			Inputs: x = some column of the training data 
					Use cached delta, cached tanhz2, and cached W2 matrix 
		"""
		xRow = np.reshape(np.append(x, 1), (self.inputSize+1, 1))
		xDelta = np.dot(xRow, self.delta3)
		W2cut = np.delete(self.W2, self.hiddenSize, axis=0)
		tanhDeriv = np.subtract(np.ones(self.hiddenSize), np.square(self.tanhz2[0:self.hiddenSize].flatten()))
		diagDeriv = np.diag(tanhDeriv)
		dJdW1 = np.dot(np.dot(xDelta, W2cut.T), np.diag(tanhDeriv))

		return 0.5*dJdW1



	def updateWeights(self, dJdW1, dJdW2):
		self.W1 = np.subtract(self.W1, self.learningRate*dJdW1)
		self.W2 = np.subtract(self.W2, self.learningRate*dJdW2)

#################### METHODS FOR CHECKING ACCURACY OF MODEL ########################

	def computeApproximateGradientW2(self, xRow, y):
		dJdW2apx = self.W2
		epsilon = 0.001

		for i in range(self.W2.shape[0]):
			for j in range(self.W2.shape[1]):
				
				#Calculate approximate loss after perturbing down
				W2down = self.W2
				W2down[i, j] -= epsilon
				lossDown = self.squareLoss(self.forward(xRow, self.W1, W2down), y)

				#Calculate approximate loss after perturbing up 
				W2up = self.W2
				W2up[i, j] += epsilon
				lossUp = self.squareLoss(self.forward(xRow, self.W1, W2up), y)

				apxGrad = (float(lossUp) - float(lossDown)) / (2*epsilon)
				dJdW2apx[i,j] = apxGrad

		return dJdW2apx


	def computeApproximateGradientW1(self, xRow, y):
		dJdW1apx = self.W1
		epsilon = 0.01

		for i in range(self.W1.shape[0]):
			for j in range(self.W1.shape[1]):
				
				#Calculate approximate loss after perturbing down
				W1down = self.W1
				W1down[i, j] -= epsilon
				lossDown = self.squareLoss(self.forward(xRow, W1down, self.W2), y)

				#Calculate approximate loss after perturbing up 
				W1up = self.W1
				W1up[i, j] += epsilon
				lossUp = self.squareLoss(self.forward(xRow, W1up, self.W2), y)

				apxGrad = (float(lossUp) - float(lossDown)) / (2*epsilon)
				dJdW1apx[i,j] = apxGrad

		return dJdW1apx





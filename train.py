import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import sklearn.svm
import sklearn
from sklearn import preprocessing
import math
import NeuralNet
from NeuralNet import NeuralNetwork
import csv

####### IMPORT DIGITS DATA AND FORMAT FOR USE IN NEURAL NETWORK #######

mat = scipy.io.loadmat('./dataset/train.mat')
trainImages = mat['train_images']
trainLabels = mat['train_labels']
trainImages = np.swapaxes(trainImages, 0, 2)

#Shuffle and then partition data 
shuf = np.arange(60000)
np.random.shuffle(shuf)
trainImages = trainImages[shuf]
trainLabels = trainLabels[shuf]

#Flatten training images
flatImages = np.zeros((60000, 28*28), dtype=np.uint8)
for i in range(60000):
	flatImages[i] = trainImages[i].flatten()

#Normalize the images data
trainImages = np.array(flatImages, dtype='f')
trainImages = preprocessing.scale(trainImages)
trainLabels = trainLabels.flatten()

testImages = trainImages[55000:60000]
testLabels = trainLabels[55000:60000]
trainImages = trainImages[0:55000]
trainLabels = trainLabels[0:55000]

####### CREATE NEURAL NETWORK AND TRAIN ON PARTITIONED DATA, TESTING EVERY 1000 ITERS #######
NN = NeuralNetwork(784, 200, 10)
errorRates = []
testIndices = range(0, 1000000, 1000)

for i in range(1000000):
	sample = np.random.randint(55000)
	x, y = trainImages[sample], trainLabels[sample]
	output = NN.forward(x, NN.W1, NN.W2)
	dJdW2 = NN.backpropW2(x, y, output)
	dJdW1 = NN.backpropW1(x, y)
	NN.updateWeights(dJdW1, dJdW2)

	if (i + 1) % 55000 == 0:
		NN.decayLearningRate()
	
	if i%1000 == 0:
		success = 0
		for j in range(testLabels.shape[0]):
			testImg, testLbl = testImages[j], testLabels[j]
			output = NN.forward(testImg, NN.W1, NN.W2)
			pred = np.argmax(output)
			if pred == testLbl:
				success += 1
		successRate = float(success) / float(testLabels.shape[0])
		error = 1.0 - successRate
		errorRates.append(error)
		print("Success rate after " + str(i) + " iterations is " + str(successRate))

plt.plot(testIndices, errorRates)
plt.xlabel("Number of Iterations")
plt.ylabel("Error Rate")
plt.show()


mat = scipy.io.loadmat('./dataset/test.mat')
testData = mat['test_images']
testData = np.swapaxes(testData, 0, 2)
flatData = np.zeros((10000, 28*28), dtype=np.uint8)
for i in range(10000):
	flatData[i] = testData[i].flatten()

testData = np.array(flatData, dtype='f')
testData = preprocessing.scale(testData)



with open('digitResults.csv', 'w') as csvfile:
    fieldnames = ['Id', 'Category']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(testData.shape[0]):
    	val = np.argmax(NN.forward(testData[i], NN.W1, NN.W2))
    	writer.writerow({'Id': str(i+1), 'Category': int(val)})


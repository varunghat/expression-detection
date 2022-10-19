# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils
import math
import pickle
import os
from math import exp
from random import seed
from random import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
 
"""Code for BPN Training"""

# Find the min and max values for each column
def dataset_minmax(dataset):
	stats = [min(dataset),max(dataset)]
	return stats
 
# Normalise the dataset to the range 0-1
def normalize_dataset(dataset, minmax):	
	for i in range(len(dataset) - 1):
		dataset[i] = (dataset[i] - minmax[0])/(minmax[1] - minmax[0])
		
			

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

# Transfer neuron activation
def transfer(x):
	return 1.0 / (1.0 + exp(-x))

# Calculate the derivative of an neuron output
def transfer_derivative(x):
	return x * (1.0 - x)

"""
#Relu Test
def transfer(x):
	return max(0,x)

def transfer_derivative(x):
	return 1 if x>0 else 0
"""

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []		
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)

			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, dataset, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		sum_error = 0
		for row in dataset:						
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]			
			expected[row[-1]] = 1
			sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

#Test input
def predict(network, row):	
	outputs = forward_propagate(network, row)
	#print(outputs)
	return outputs.index(max(outputs))

# Test training backprop algorithm
def InitBPN(dataset,epochs,l_rate,n_hidden):
	seed(1)

	n_inputs = len(dataset[0]) - 1
	n_outputs = len(set([row[-1] for row in dataset]))
	for i in range(len(dataset)):
		minmax = dataset_minmax(dataset[i])
		normalize_dataset(dataset[i],minmax)
	network = initialize_network(n_inputs, n_hidden , n_outputs)	
	train_network(network, dataset, l_rate, epochs, n_outputs)
	#for layer in network:
	#	print(layer)
	return network

"""Back propogation code end"""

"""Code for Facial Feature extraction"""

shape_predictor_filename = "shape_predictor_68_face_landmarks.dat"

facial_features_cordinates = {}

# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
		("Mouth", (48, 68)),
		("Right_Eyebrow", (17, 22)),
		("Left_Eyebrow", (22, 27)),
		("Right_Eye", (36, 42)),
		("Left_Eye", (42, 48)),
		("Nose", (27, 35)),
		("Jaw", (0, 17))
])

def shape_to_numpy_array(shape, dtype="int"):    
    coordinates = np.zeros((68, 2), dtype=dtype)    
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)   
    return coordinates

"""
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75): 
    overlay = image.copy()
    output = image.copy()
    
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # return the output image
    print(facial_features_cordinates)
    return output
"""

def getDist(a,b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

#Calculate center of facial features
def getFaceCenter(shape):
    faceCenter=np.zeros(2,dtype="int")
    i=17
    while(i<68):
        faceCenter[0]+=shape[i][0]
        faceCenter[1]+=shape[i][1]
        i+=1
    faceCenter[0]=int(faceCenter[0]/(68-17))
    faceCenter[1]=int(faceCenter[1]/(68-17))
    return faceCenter
     
#Convert coordinates into meaningful inputs for BPN    
def getInputs(shape):
    faceCenter=getFaceCenter(shape)
    landmarks = [18,20,22,23,25,27,37,40,43,46,49,55,67,63]
    eye = [[[43,46],[44,48],[45,47]],[[37,40],[38,42],[39,41]]]

    #Correcting numbers into indices
    for i in range(len(landmarks)):
        landmarks[i] -=1
    for i in range(len(eye)):
        for j in range(len(eye[i])):
            for k in range(len(eye[i][j])):
                eye[i][j][k] -= 1
    inputs = np.zeros(len(landmarks) + 2)
    i=0
    while(i<len(landmarks)):
        inputs[i]=getDist(shape[landmarks[i]],faceCenter)
        i+=1    
    
    k=1
    for i in eye:
        d = []
        for j in i:
            d.append(getDist(shape[j[0]],shape[j[1]]))
        eyeRatio = 2 * d[0] / (d[1] + d[2])
        inputs[len(inputs)-1 -k] = eyeRatio * 20
        k-=1
    return inputs    

# initialize dlib's face detector and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(BASE_DIR,shape_predictor_filename))

#Read image and extract features
def extractFeatures(path,num):
	
	# load image, resize, convert to grayscale
	image = cv2.imread(path)
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	
	# loop over the face detections
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = shape_to_numpy_array(shape)
		inputs = getInputs(shape)
		inputs = list(inputs)
		inputs.append(num)
		
		"""
		output = visualize_facial_landmarks(image, shape)		
		cv2.imshow("Image", output)
		cv2.waitKey(0)
		"""		
		return inputs

#Read images from dataaset and train Network
def train_images():
	epochs = 4000
	l_rate = 0.4
	n_hidden = 4

	#file handling code
	#file = open("epochs.txt", "w")
	
	imagesPath = os.path.join(BASE_DIR,"images")
	subfolders = [f.name for f in os.scandir(imagesPath) if f.is_dir()]
	print(subfolders)	
		
	i=0	
	extractedData = []    
	for expression in subfolders:     
		    
		imgPath = os.path.join(imagesPath,expression)
		for f in os.scandir(imgPath):                        
			extractedData.append(extractFeatures(f.path,i))        
		i+=1

	
	
	network = InitBPN(extractedData,epochs,l_rate,n_hidden)
	print("DONE\n")
	
	#accuracy calculation
	count = 0
	correct = 0
	for row in extractedData:
		prediction = predict(network, row)
		if(row[-1]==prediction):
			correct+=1
		count+=1
		#print('Expected=%d, Got=%d' % (row[-1], prediction))
	accuracy = correct/ count
	#accuracy2 = test_images(network,subfolders)
	print("%d / %d Accuracy = %f\n" % (correct, count,accuracy))
	
	
	#file.write(str(epochs) +"---"+ str(accuracy)+" " + str(accuracy2) + "\n")
	
	
	#file.close()
	return network,subfolders
	
     
#Read test images and predict class
def test_images(network,classes):
	imagesPath = os.path.join(BASE_DIR,"test")
	print("\nTESTING IMAGES : \n")
	extractedData = []
	i=0   
	count=0 
	for f in os.scandir(imagesPath):
		#print(f.name)
		extractedData = extractFeatures(f.path,0)			
		minmax = dataset_minmax(extractedData)
		normalize_dataset(extractedData,minmax)
		prediction = predict(network,extractedData)
		print("For image %s, got %s" %(f.name,classes[prediction]))
		if(int(i/2) == prediction):
			count+=1
		i+=1
	
	accuracy = float(count/i)
	print("Testing accuracy = %f" % (accuracy))
	return accuracy     



#Main Code

print("Train network from start?")	
choice = input()
if(choice=="y" or choice=="Y"):
	network,classes=train_images()	
	#save weights 
	pickle.dump(network,open("network.p","wb"))
else:
	#load weights
	pickle.load(open("network.p","rb"))
	imagesPath = os.path.join(BASE_DIR,"images")
	classes = [f.name for f in os.scandir(imagesPath) if f.is_dir()]

test_images(network,classes)

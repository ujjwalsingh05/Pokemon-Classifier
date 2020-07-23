import os 
from pathlib import Path
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

p = Path("./dataset/")
dirs = p.glob("*")
image_data = []
labels = []

image_paths = []
label_dict = {"Pikachu":0 , "Bulbasaur":1 , "Meowth":2}
label2pokemon = {0:"Pikachu", 1:"Bulbasaur", 2:"Meowth"}

for folder_dir in dirs:
	# print(folder_dir)
	# print(type(folder_dir))
	# print((str(folder_dir)))

	label = str(folder_dir).split("\\")[-1]

	cnt = 0
	print(folder_dir)

	# Iterate over folde_dir and pick all images of the pokemon
	for img_path in folder_dir.glob("*.jpg"):
		img = load_img(img_path, target_size=(40,40))
		img_array = img_to_array(img)
		image_data.append(img_array)
		labels.append(label_dict[label])
		cnt+=1

	print(cnt)


# Visualisation
import numpy as np
print(len(image_data))
print(len(labels))
# print(labels)

import random
random.seed(10)

X=np.array(image_data)
Y=np.array(labels)


from sklearn.utils import shuffle
X,Y = shuffle(X,Y,random_state=2)

X = X/255.0
print(Y)

# print(X.shape)
# print(Y.shape)

# Draw some pokemons
def drawImg(img, label):
	plt.title(label2pokemon[label])
	plt.imshow(img)
	plt.style.use("seaborn")
	plt.axis("off")
	plt.show()


for i in range(1,20):
	drawImg(X[i].reshape(40,40,3),Y[i]) #4800i/p units


## shende

#create training and testing set 
split = int(X.shape[0]*.8)
X_ = np.array(X)
Y_ = np.array(Y)

#training set
X = X_[:split,:]
Y = Y_[:split,:]

#test set
XTest = X_[split:,:]
YTest = Y_[split:]

print(X.shape, Y.shape)
print(XTest.shape, YTest.shape)

class NeuralNetwork:
	def __init__(self, input_size, layers, output_size): 
		np.random.seed(0)
		model = {}

		#first layer
		model['W1'] = np.random.randn(input_size, layers[0])
		model['b1'] = np.zeros((1, layers[0]))

		#second layer
		model['W2'] = np.random.randn(layers[0], layers[1])
		model['b2'] = np.zeros((1, layers[1]))

		#third layer/output layer
		model['W3'] = np.random.randn(layers[1], output_size)
		model['b3'] = np.zeros((1, output_size))

		self.model = model
		self.activation_outputs = None

	#forward func
	def forward(self, x):
		W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']

		b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']

		z1 = np.dot(x,W1) + b1
		a1 = np.tanh(z1)

		z2 = np.dot(a1,W2) + b2
		a2 = np.tanh(z2)

		z3 = np.dot(a2,W3) + b3
		y_ = softmax(z3)

		self.activation_outputs = (a1,a2,y_)
		return y_

	# backward
	def backward (self, x,y, learning_rate=0.001):
		W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']
		b1, b2, b3 = self.model['b1'], self.model['b2'], self.model['b3']

		m = x.shape[0]

		a1, a2, y_ = self.activation_outputs

		delta3 = y_-y
		dw3 = np.dot(a2.T, delta3)
		db3 = np.sum(delta3, axis=0) 

		delta2 = (1-np.square(a2)) * np.dot(delta3, W3.T)
		dw2 = np.dot(a1.T, delta2)
		db2 = np.sum(delta2, axis=0) 

		delta1 = (1-np.square(a1)) * np.dot(delta2, W2.T)
		dw1 = np.dot(X.T, delta1)
		db1 = np.sum(delta1, axis=0) 

		# Update the model parameters using Gradient Descent
		self.model["W1"] -= learning_rate * dw1
		self.model['b1'] -= learning_rate * db1
		self.model["W2"] -= learning_rate * dw2
		self.model['b2'] -= learning_rate * db2
		self.model["W3"] -= learning_rate * dw3
		self.model['b3'] -= learning_rate * db3

	def predict(self,x):
		y_out= self.forward(x)
		return np.argmax(y_out,axis=1)

	def summary(self):
		W1,W2,W3 = self.model['W1'], self.model['W2'], self.model['W3']
		a1,a2,y_ = self.activation_outputs

		print("W1", W1.shape)
		print("A1", a1.shape)
		# print("W2", W2.shape)
		# print("A2", a2.shape)
		# print("W3", W3.shape)
		# print("Y_", y_.shape)

def softmax(a):
	e_pa = np.exp(a)
	ans = e_pa/np.sum(e_pa, axis=1, keepdims=True)
	return ans


def loss(y_oht, p):
	l=-np.mean(y_oht * np.log(p))
	return l

def one_hot(y,depth):
	m=y.shape[0]
	y_oht= np.zeros((m,depth))
	y_oht[np.arange(m),y] = 1
	return y_oht






def train(X, Y, model, epochs,learning_rate, logs=True):
	training_loss = []

	classes = len(np.unique(Y))
	Y_OHT = one_hot(Y, classes)

	for ix in range(epochs):

		Y_ = model.forward(X)
		l = loss(Y_OHT,Y_)

		model.backward(X,Y_OHT,learning_rate)
		training_loss.append(l)
		if(logs and i%50==0):
			print("Epoch %d loss %.4f"%(ix,l))

	return training_loss




model = NeuralNetwork(input_size = 4800, layers = [100,50], output_size = 3)

print(X.shape)
X = X.reshape(X.shape[0],-1)
print(X.shape)

XTest = XTest.reshape(XTest.shape[0], -1)
print(XTest.shape)


l = train(X,Y,model,500,0.0002)

from matplotlib import pyplot as plt 

plt.style.use("dark_background")
plt.title("training_loss vs epochs")
plt.plot(l)
plt.show()


#accuracy

def getAccuracy(X, Y, model):
	outputs = model.predict(X)
	acc = np.sum(outputs == Y)/Y.shape[0]
	return acc


print("Train Acc %.4f"% getAccuracy(X,Y,model))	
print("Test Acc %.4f"% getAccuracy(XTest,YTest,model))











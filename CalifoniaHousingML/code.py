import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('/content/sample_data/california_housing_train.csv') #read data

# print(data)
# print(data.columns)
correct = np.array(data.pop('median_house_value'))  #pop out the data we want to predict
n_data = len(correct) #17000 the length of the data
# print(n_data)
input_data = np.array(data) #change it to np.array object

correct_data = np.zeros((n_data,5)) #creat a 17000,5 matrix full of zeros
for i in range(n_data): #here I divide the correct data into five types
  if correct[i] <=100000: #[1,0,0,0,0]
    correct_data[i,0] = 1
  elif correct[i] <= 200000: #[0,1,0,0,0]
    correct_data[i,1] = 1
  elif correct[i] <= 300000: #[0,0,1,0,0]
    correct_data[i,2] = 1
  elif correct[i] <= 400000: #[0,0,0,1,0]
    correct_data[i,3] = 1
  else:                      #[0,0,0,0,1]
    correct_data[i,4] = 1

ave_input = np.average(input_data,axis=0)
std_input = np.std(input_data,axis=0)
input_data = (input_data-ave_input) / std_input  #normalize my input data
# print(np.array(input_data))
index = np.arange(n_data)  #change my data into nd.array object
index_train = index[index%5 != 0]  #divide input data into  train and test data
index_test = index[index%5 == 0]

input_train = input_data[index_train, :]  
correct_train = correct_data[index_train, :]  #and same to correct data
input_test = input_data[index_test, :]  
correct_test = correct_data[index_test, :]  

n_train = input_train.shape[0] 
n_test = input_test.shape[0] 
# print(n_train,n_test)

n_in = 8 #8 input
n_mid = 30 #middle layer
n_out = 5 #5output

wb_width = 0.1
eta = 0.005 #learning rate
epoch = 1000 #epoch
batch_size = 32 #batch size
interval = 100 #interval

class Baselayer: #class this cause other layers can simply inherit it
  def __init__(self, n_upper, n):
    self.w = wb_width * np.random.randn(n_upper,n) #[8,30]的陣列 #weights
    self.b = wb_width * np.random.randn(n) #[30] bias

  def update(self, eta):
    self.w -= eta * self.grad_w #(8,30)  changing weights
    self.b -= eta * self.grad_b #(30,)   same to bias

class MiddleLayer(Baselayer): #inherit
  def forward(self, x):
    self.x = x #(14000,8)
    self.u = np.dot(x, self.w) + self.b #(14000,30)because (14000,8)*(8,30)
    self.y = np.where(self.u <= 0,0,self.u) #(14000,30)
  
  def backward(self, grad_y):
    delta = grad_y * np.where(self.u <= 0, 0, 1) #(50,30)
    # print(delta.shape)
    self.grad_w = np.dot(self.x.T, delta) #1:(8,30) 2:(30,30)
    self.grad_b = np.sum(delta, axis=0) #(30,0)
    self.grad_x = np.dot(delta, self.w.T) #1:(50,8) 2:(50,30)

class Outputlayer(Baselayer):
  def forward(self, x):
    self.x = x #(14000,30)
    u = np.dot(x, self.w) + self.b #(30,8)
    self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True) #(14000,8)

  def backward(self, t):
    delta = self.y - t  #(50,8)
    # print(delta.shape)
    self.grad_w = np.dot(self.x.T, delta) #(30,8)
    self.grad_b = np.sum(delta, axis=0) #(8,)按行相加
    self.grad_x = np.dot(delta, self.w.T) #(50,30)

class Dropout: 
  def __init__(self, dropout_ratio):
    self.dropout_ratio = dropout_ratio
  
  def forward(self, x, is_train):
    if is_train:
      rand = np.random.rand(*x.shape)
      self.dropout = np.where(rand > self.dropout_ratio, 1, 0)
      self.y = x * self.dropout
    else:
      self.y = (1-self.dropout_ratio)*x
  
  def backward(self, grad_y):
    self.grad_x = grad_y * self.dropout
middle_layer_1 = MiddleLayer(n_in,n_mid)
dropout_1 = Dropout(0.5)#dropout
middle_layer_2 = MiddleLayer(n_mid,n_mid)
dropout_2 = Dropout(0.5)#dropout
output_layer = Outputlayer(n_mid,n_out)

def forward_propagation(x):
  middle_layer_1.forward(x)
  dropout_1.forward(middle_layer_1.y, is_train=True)#dropout
  middle_layer_2.forward(middle_layer_1.y)
  dropout_2.forward(middle_layer_2.y, is_train=True)#dropout
  output_layer.forward(middle_layer_2.y)

def backpropagation(t):
  output_layer.backward(t)
  dropout_2.backward(output_layer.grad_x)#dropout
  middle_layer_2.backward(output_layer.grad_x)
  dropout_1.backward(middle_layer_2.grad_x)#dropout
  middle_layer_1.backward(middle_layer_2.grad_x)

def update_wb():
  middle_layer_1.update(eta)
  middle_layer_2.update(eta)
  output_layer.update(eta)

def get_error(t, batch_size):
  return -np.sum(t * np.log(output_layer.y + 1e-7)) / batch_size

  
train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []

n_batch = n_train // batch_size

for i in range(epoch):

  forward_propagation(input_train)
  error_train = get_error(correct_train, n_train)
  forward_propagation(input_test)
  error_test = get_error(correct_test, n_test)

  test_error_x.append(i)
  test_error_y.append(error_test)
  train_error_x.append(i)
  train_error_y.append(error_train)

  if i%interval == 0:
    print("Epoch:" + str(i) + "/" + str(epoch),
          "Error_train:" + str(error_train),
          "Error_test:" + str(error_test))
  
  index_random = np.arange(n_train)
  np.random.shuffle(index_random)
  
  for j in range(n_batch):

    mb_index = index_random[j*batch_size : (j+1)*batch_size]
    x = input_train[mb_index, :]
    t = correct_train[mb_index, :]

    forward_propagation(x)
    backpropagation(t)
    update_wb()

plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

forward_propagation(input_train)

count_train = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_train, axis=1))

forward_propagation(input_test)

count_test = np.sum(np.argmax(output_layer.y, axis=1) == np.argmax(correct_test, axis=1))

print("Accuracy Train:",str(count_train/n_train*100) + "%",
      "Accuracy Test:",str(count_test/n_test*100) + "%")

# Prediction
samples = input_data[10000:10005]
# ave_input = np.average(samples, axis=0)
# std_input = np.std(samples, axis=0)
# samples = (samples-ave_input)/std_input
forward_propagation(samples)
print(output_layer.y.round(2))
print(correct_data[10000:10005])

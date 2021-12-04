import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from google.colab import drive
#mount drive
drive.mount('/content/gdrive',force_remount=True) 
from PIL import Image
# from im2col import im2col,col2im
# import layers

#-----testing-----
# training = Image.open("Testing/c/22.png")
# im = np.array(training)
# plt.imshow(im)
# plt.show()
#-----done------
alphdic = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
dic = {'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,'u':20,'v':21,'w':22,'x':23,'y':24,'z':25}
train_img = np.array([])
test_img = np.array([])
train_correct = []
test_correct = []
#load dats and resize it from (60,60) to (32,32)
for alp in alphdic:
    for img in os.listdir("/content/gdrive/MyDrive/coding/practice/Training/"+alp):
        i = Image.open("/content/gdrive/MyDrive/coding/practice/Training/" + alp + "/" + str(img))
        i = i.resize((32,32))
        im = np.array(i)
        im = im.reshape(1024,)
        train_img = np.append(train_img,im)
        train_correct.append(alp)
#reshape to (501,1024) for further use
train_img = train_img.reshape(1024,501).T
# print(train_img.shape)
#same for testing data
for alp in alphdic:
    for img in os.listdir("/content/gdrive/MyDrive/coding/practice/Testing/"+alp):
        i = Image.open("/content/gdrive/MyDrive/coding/practice/Testing/" + alp + "/" + str(img))
        i = i.resize((32,32))
        im = np.array(i)
        im = im.reshape(1024,)
        test_img = np.append(test_img,im)
        test_correct.append(alp)
#resize is to (260,1024)
test_img = test_img.reshape(1024,260).T
# print(test_img.shape)
train_correct = np.array(train_correct).T #501
test_correct = np.array(test_correct).T
# print(test_correct.shape)
n_data = len(train_correct) + len(test_correct) #761
# print(n_test,n_train)
#put it all together
correct = np.append(train_correct,test_correct)
#class the letters into 26classes ,means each position stands for a letter
correct_data = np.zeros((n_data, 26))
for i in range(n_data):
    correct_data[i][dic[str(correct[i])]] = 1
# print(correct_data)
img = np.append(train_img,test_img)
img = img.reshape(761,1024)
# print(img.shape)
#split train and test
index = np.arange(761)
index_train = index[index%3 != 0]
index_test = index[index%3 == 0]

train_img = img[index_train, :]
train_correct = correct_data[index_train, :]
test_img = img[index_test, :]  
test_correct = correct_data[index_test, :] 
# print(n_data)
n_train = train_img.shape[0]
n_test = test_img.shape[0]
# print(correct_data.shape)
# print(test_correct.shape)
# print(train_img)

#normalize the data
ave_train = np.average(train_img)
std_train = np.std(train_img)
train_img = (train_img-ave_train) / std_train
ave_test = np.average(test_img)
std_test = np.std(test_img)
test_img = (test_img-ave_test) / std_test


# print(train_img.shape,test_img.shape)
#input image height, weight, and how many colors(img_ch)
img_h = 32  
img_w = 32
img_ch = 1


wb_width = 0.1  #-->parameters of weight
eta = 0.01      #-->learning rate
epoch = 50      #-->epoch
batch_size = 8  #-->size of a batch
interval = 10   #-->the interval
n_sample = 200  #for testing

#Here define 2 important functions,basically it transfer datas to the data we need for convolutioning
def im2col(images, flt_h, flt_w, out_h, out_w, stride=1, pad=0):
   
    n_bt, n_ch, img_h, img_w = images.shape
    
    img_pad = np.pad(images, [(0,0), (0,0), (pad, pad), (pad, pad)], "constant")
    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            cols[:, :, h, w, :, :] = img_pad[:, :, h:h_lim:stride, w:w_lim:stride]

    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(n_ch*flt_h*flt_w, n_bt*out_h*out_w)
    return cols

def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride=1, pad=0):
 
    n_bt, n_ch, img_h, img_w = img_shape
    
    cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h, out_w).transpose(3, 0, 1, 2, 4, 5)
    images = np.zeros((n_bt, n_ch, img_h+2*pad+stride-1, img_w+2*pad+stride-1))
    
    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            images[:, :, h:h_lim:stride, w:w_lim:stride] += cols[:, :, h, w, :, :]

    return images[:, :, pad:img_h+pad, pad:img_w+pad]

#convolution layer, using adagrad and relu.
class ConvLayer:
    
    
    def __init__(self, x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad):


        self.params = (x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad)
        
        
        self.w = wb_width * np.random.randn(n_flt, x_ch, flt_h, flt_w)
        self.b = wb_width * np.random.randn(1, n_flt)
        
        
        self.y_ch = n_flt  
        self.y_h = (x_h - flt_h + 2*pad) // stride + 1  
        self.y_w = (x_w - flt_w + 2*pad) // stride + 1  
 
        # AdaGrad
        self.h_w = np.zeros((n_flt, x_ch, flt_h, flt_w)) + 1e-8
        self.h_b = np.zeros((1, n_flt)) + 1e-8
        
    def forward(self, x):
        n_bt = x.shape[0] 
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w
        
        
        self.cols = im2col(x, flt_h, flt_w, y_h, y_w, stride, pad)
        self.w_col = self.w.reshape(n_flt, x_ch*flt_h*flt_w)
        

        u = np.dot(self.w_col, self.cols).T + self.b
        self.u = u.reshape(n_bt, y_h, y_w, y_ch).transpose(0, 3, 1, 2)
        self.y = np.where(self.u <= 0, 0, self.u) #relu
    
    def backward(self, grad_y):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w
        
        # delta
        delta = grad_y * np.where(self.u <= 0, 0, 1)
        delta = delta.transpose(0,2,3,1).reshape(n_bt*y_h*y_w, y_ch)
        
        
        grad_w = np.dot(self.cols, delta)
        self.grad_w = grad_w.T.reshape(n_flt, x_ch, flt_h, flt_w)
        self.grad_b = np.sum(delta, axis=0)
        
        
        grad_cols = np.dot(delta, self.w_col)
        x_shape = (n_bt, x_ch, x_h, x_w)
        self.grad_x = col2im(grad_cols.T, x_shape, flt_h, flt_w, y_h, y_w, stride, pad)
        
    def update(self, eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b


#Pooling layer
class PoolingLayer:
    
    
    
    def __init__(self, x_ch, x_h, x_w, pool, pad):
        
    
        self.params = (x_ch, x_h, x_w, pool, pad)
        

        self.y_ch = x_ch  
        self.y_h = x_h//pool if x_h%pool==0 else x_h//pool+1  
        self.y_w = x_w//pool if x_w%pool==0 else x_w//pool+1  
        
    def forward(self, x):
        n_bt = x.shape[0] 
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w
        
        
        cols = im2col(x, pool, pool, y_h, y_w, pool, pad)
        cols = cols.T.reshape(n_bt*y_h*y_w*x_ch, pool*pool)
        
        
        y = np.max(cols, axis=1)
        self.y = y.reshape(n_bt, y_h, y_w, x_ch).transpose(0, 3, 1, 2)
        
        
        self.max_index = np.argmax(cols, axis=1)
    
    def backward(self, grad_y):
        n_bt = grad_y.shape[0] 
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w
        

        grad_y = grad_y.transpose(0, 2, 3, 1)
        
        
        grad_cols = np.zeros((pool*pool, grad_y.size))
        grad_cols[self.max_index.reshape(-1), np.arange(grad_y.size)] = grad_y.reshape(-1) 
        grad_cols = grad_cols.reshape(pool, pool, n_bt, y_h, y_w, y_ch)
        grad_cols = grad_cols.transpose(5,0,1,2,3,4) 
        grad_cols = grad_cols.reshape( y_ch*pool*pool, n_bt*y_h*y_w)

        
        x_shape = (n_bt, x_ch, x_h, x_w)
        self.grad_x = col2im(grad_cols, x_shape, pool, pool, y_h, y_w, pool, pad)

class BaseLayer:
    def __init__(self, n_upper, n):
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)

        self.h_w = np.zeros(( n_upper, n)) + 1e-8
        self.h_b = np.zeros(n) + 1e-8
        
    def update(self, eta):
        self.h_w += self.grad_w * self.grad_w
        self.w -= eta / np.sqrt(self.h_w) * self.grad_w
        
        self.h_b += self.grad_b * self.grad_b
        self.b -= eta / np.sqrt(self.h_b) * self.grad_b

#Middle layer
class MiddleLayer(BaseLayer):
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u)
    
    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1)
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 

#Output layer
class OutputLayer(BaseLayer):     
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1).reshape(-1, 1)

    def backward(self, t):
        delta = self.y - t
        
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        
        self.grad_x = np.dot(delta, self.w.T) 

#dropout layer
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


# cl_1 = ConvLayer(img_ch, img_h, img_w, 10, 2, 2, 1, 1)
# cl_2 = ConvLayer(cl_1.y_ch, cl_1.y_h, cl_1.y_w, 10, 2, 2, 1, 1)
# pl_1 = PoolingLayer(cl_2.y_ch, cl_2.y_h, cl_2.y_w, 2, 0)

# n_fc_in = pl_1.y_ch * pl_1.y_h * pl_1.y_w
# ml_1 = MiddleLayer(n_fc_in, 200)
# dr_1 = Dropout(0.7)

# ml_2 = MiddleLayer(200, 200)
# dr_2 = Dropout(0.7)

# ol_1 = OutputLayer(200, 26)

#Place and set the layers
cl_1 = ConvLayer(img_ch, img_h, img_w, 4, 3, 3, 1, 1)
dr_1 = Dropout(0.5)
# cl_2 = ConvLayer(cl_1.y_ch, cl_1.y_h, cl_1.y_w, 4, 3, 3, 1, 1)
# dr_2 = Dropout(0.6)
pl_1 = PoolingLayer(cl_1.y_ch, cl_1.y_h, cl_1.y_w, 2, 0)
n_fc_in = pl_1.y_ch * pl_1.y_h * pl_1.y_w
ml_1 = MiddleLayer(n_fc_in, 128)
dr_3 = Dropout(0.5)
# ml_2 = MiddleLayer(100, 50)
# dr_4 = Dropout(0.6)
ol_1 = OutputLayer(128, 26)

#front propagation
def forward_propagation(x, is_train):
    n_bt = x.shape[0]
    
    images = x.reshape(n_bt, img_ch, img_h, img_w)
    # cl_1.forward(images)
    # cl_2.forward(cl_1.y)
    # pl_1.forward(cl_2.y)
    
    # fc_input = pl_1.y.reshape(n_bt, -1)       
    # ml_1.forward(fc_input)
    #-------------testing---------------
    cl_1.forward(images)
    dr_1.forward(cl_1.y,is_train)
    # cl_2.forward(dr_1.y)
    # dr_2.forward(cl_2.y,is_train)
    pl_1.forward(dr_1.y)
    
    fc_input = pl_1.y.reshape(n_bt, -1)       
    ml_1.forward(fc_input)
    dr_3.forward(ml_1.y,is_train)
    # ml_2.forward(dr_3.y)
    # dr_4.forward(ml_2.y,is_train)
    ol_1.forward(dr_3.y)
    #-------------done-----------------
    # dr_1.forward(ml_1.y, is_train)
    # ml_2.forward(dr_1.y)
    # dr_2.forward(ml_2.y, is_train)
    # ol_1.forward(dr_2.y)

#back propagation
def backpropagation(t):
    n_bt = t.shape[0]

    ol_1.backward(t)
    #-------------test---------------
    # dr_4.backward(ol_1.grad_x)
    # ml_2.backward(dr_4.grad_x)
    dr_3.backward(ol_1.grad_x)
    ml_1.backward(dr_3.grad_x)
    grad_img = ml_1.grad_x.reshape(n_bt, pl_1.y_ch, pl_1.y_h, pl_1.y_w)
    pl_1.backward(grad_img)
    # dr_2.backward(pl_1.grad_x)
    # cl_2.backward(dr_2.grad_x)
    dr_1.backward(pl_1.grad_x)
    cl_1.backward(dr_1.grad_x)
    #-------------done--------------
    # dr_2.backward(ol_1.grad_x)
    # ml_2.backward(dr_2.grad_x)
    # dr_1.backward(ml_2.grad_x)
    # ml_1.backward(dr_1.grad_x)
    
    # grad_img = ml_1.grad_x.reshape(n_bt, pl_1.y_ch, pl_1.y_h, pl_1.y_w)
    # pl_1.backward(grad_img)
    # cl_2.backward(pl_1.grad_x)
    # cl_1.backward(cl_2.grad_x)

#update
def uppdate_wb():
    cl_1.update(eta)
    # cl_2.update(eta)
    ml_1.update(eta)
    # ml_2.update(eta)
    ol_1.update(eta)

#loss function:cross entropy
def get_error(t, batch_size):
    return -np.sum(t * np.log(ol_1.y + 1e-7)) / batch_size


def forward_sample(inp, correct, n_sample):
    index_rand = np.arange(len(correct))
    np.random.shuffle(index_rand) 
    index_rand = index_rand[:n_sample]
    x = inp[index_rand, :]
    t = correct[index_rand, :]
    forward_propagation(x, False)
    return x, t

train_error_x = []
train_error_y = []
test_error_x = []
test_error_y = []


n_batch = n_train // batch_size
#running and getting errors
for i in range(epoch):

    
    x, t = forward_sample(train_img, train_correct, n_sample)  
    error_train = get_error(t, n_sample)
    # break
    x, t = forward_sample(test_img, test_correct, n_sample) 
    error_test = get_error(t, n_sample)
    
    
    train_error_x.append(i)
    train_error_y.append(error_train) 
    test_error_x.append(i)
    test_error_y.append(error_test) 
    
    
    if i%interval == 0:
        print("Epoch:" + str(i) + "/" + str(epoch),
              "Error_train:" + str(error_train),
              "Error_test:" + str(error_test))
    
    
    index_rand = np.arange(n_train)
    np.random.shuffle(index_rand)   
    for j in range(n_batch):
        
        mb_index = index_rand[j*batch_size : (j+1)*batch_size]
        x = train_img[mb_index, :]
        t = train_correct[mb_index, :]

        forward_propagation(x, True)
        backpropagation(t)        
        uppdate_wb() 
            
    

plt.plot(train_error_x, train_error_y, label="Train")
plt.plot(test_error_x, test_error_y, label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Error")

plt.show()


x, t = forward_sample(train_img, train_correct, n_train) 
count_train = np.sum(np.argmax(ol_1.y, axis=1) == np.argmax(t, axis=1))

x, t = forward_sample(test_img, test_correct, n_test) 
count_test = np.sum(np.argmax(ol_1.y, axis=1) == np.argmax(t, axis=1))

print("Accuracy Train:", str(count_train/n_train*100) + "%",
      "Accuracy Test:", str(count_test/n_test*100) + "%")


# using samples to predict datas
samples = test_img[80:81]
forward_propagation(samples,False)
print(ol_1.y.round(3)[0][12])

print(test_correct[80:81][0][12])

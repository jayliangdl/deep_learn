import numpy as np
import math
import matplotlib.pyplot as plt
##from bigfloat import *

def forward(W,b,pre_A,activation):
    Z=np.dot(W,pre_A)+b
    Z=np.array(Z,dtype=np.float64)
    if activation=='tanh':
        A=(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
    elif activation=='softmax':
        A=np.exp(Z)/np.sum(np.exp(Z), axis=1, keepdims=True)
    return Z,A

def backward(dA,Z,m,pre_A,W,A,Y,activation,lambd):
    if activation=='tanh':
        s=(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))
        dZ = dA*(1-s*s)
        ##dW=1.0/m*np.dot(dZ,pre_A.T) + lambd/m*W
        dW=1.0/m*np.dot(dZ,pre_A.T)
        db=1.0/m*np.sum(dZ,axis=1,keepdims=True)
        pre_dA=np.dot(W.T,dZ)
    elif activation=='softmax':
        dZ=A-Y
        ##dW=1.0/m*np.dot(dZ,pre_A.T) + lambd/m*W
        dW=1.0/m*np.dot(dZ,pre_A.T)
        db=1.0/m*np.sum(dZ,axis=1,keepdims=True)
        pre_dA=np.dot(W.T,dZ)
    return dZ,dW,db,pre_dA

def update_parameter(learning_rate,W,b,dW,db):
    W=W-learning_rate*dW
    b=b-learning_rate*db
    return W,b

'''
func:init_parameter(hidden_layer)
arg:
    hidden_layer为所有层的
初始化W/b值，并返回类型为dict的param。
'''
def init_parameter(hidden_layer):
    np.random.seed(3)
    param = {}
    for i in range(len(hidden_layer)-1):
        W=np.random.randn(hidden_layer[i+1],hidden_layer[i])*np.power((2/hidden_layer[i]),0.5)
        ##W=np.random.randn(hidden_layer[i+1],hidden_layer[i])
        b=np.zeros((hidden_layer[i+1],1))
        param['W'+str(i+1)]=W
        param['b'+str(i+1)]=b
    return param

def cal_cost(m,y_hat,Y,lambd,param):
    sum_w=0
    '''
    for i in range(1,len(hidden_layer)):
        W=param['W'+str(i)]
        sum_w=sum_w+np.sum(np.square(W))
    
    l2_regularization_cost = (1/m*lambd/2)*sum_w
    cost=-1/m*np.sum(np.exp(y_hat[Y==1])) + l2_regularization_cost
    '''
    cost=-1/m*np.sum(np.exp(y_hat[Y==1]))
    return cost

def forward_and_backward(X,Y,hidden_layer,param,m,learning_rate,lambd):
    pre_A=X
    cache={}
    cache['A0']=pre_A
    cost = 0.0
    for i in range(1,len(hidden_layer)):
        W=param['W'+str(i)]
        b=param['b'+str(i)]
        if i!=len(hidden_layer)-1:
            activation='tanh'
        else:
            activation='softmax'
        ##print('activation:'+activation+',len(hidden_layer)-1:'+str(len(hidden_layer)-1)+',i='+str(i))
        Z,A = forward(W,b,pre_A,activation)
        cache['Z'+str(i)]=Z
        cache['A'+str(i)]=A
        pre_A=A
    y_hat = cache['A'+str(len(hidden_layer)-1)]
    cost = cal_cost(m,y_hat,Y,lambd,param)
    dA=0.0    
    for i in reversed(range(1,len(hidden_layer))):
        W=param['W'+str(i)]
        b=param['b'+str(i)]
        Z=cache['Z'+str(i)]
        A=cache['A'+str(i)]
        pre_A=cache['A'+str(i-1)]
        if i!=len(hidden_layer)-1:
            activation='tanh'
        else:
            activation='softmax'
        
        dZ,dW,db,pre_dA = backward(dA,Z,m,pre_A,W,A,Y,activation,lambd)
        dA = pre_dA
        cache['dZ'+str(i)]=dZ
        cache['dW'+str(i)]=dW
        cache['db'+str(i)]=db
    
    for i in range(len(hidden_layer)-1):
        W=param['W'+str(i+1)]
        b=param['b'+str(i+1)]
        dW=cache['dW'+str(i+1)]
        db=cache['db'+str(i+1)]
        
        W,b = update_parameter(learning_rate,W,b,dW,db) 
        param['W'+str(i+1)]=W
        param['b'+str(i+1)]=b
    return param,cost,cache

def predict(hidden_layer,param,X,cache):
    pre_A=X
    for i in range(1,len(hidden_layer)):
        W=param['W'+str(i)]
        b=param['b'+str(i)]
        if i!=len(hidden_layer)-1:
            activation='tanh'
        else:
            activation='softmax'
        
        Z,A = forward(W,b,pre_A,activation)
        cache['Z'+str(i)]=Z
        cache['A'+str(i)]=A
        pre_A=A
        
    y_hat=A    
    predict_result = np.argmax(y_hat,axis=0)
    return predict_result

def convert_mini_batch(X,Y,mini_batch_size,m,seed):
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation]
    number_of_mini_batch=math.floor(m/mini_batch_size)
    mini_batches = []
    for k in range(number_of_mini_batch):
        mini_X = X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_Y = Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch=(mini_X,mini_Y)
        mini_batches.append(mini_batch)
    if m%mini_batch_size!=0:
        mini_X = X[:,(k+1)*mini_batch_size:]
        mini_Y = Y[:,(k+1)*mini_batch_size:]
        mini_batch=(mini_X,mini_Y)
        mini_batches.append(mini_batch)
    return mini_batches

'''
def mainlogic(X,Y,hidden_layer,number_iterator,m,learning_rate):
    param = init_parameter(hidden_layer)
    standard = np.argmax(Y,axis=0)
    ##standard = Y.copy()
    costs = []
    for i in range(number_iterator):
        param,cost,cache = forward_and_backward(X,Y,hidden_layer,param,m,learning_rate)
        predict_result = predict(hidden_layer,param,X,cache)
        
        accuracy = np.sum(predict_result==standard)
        costs.append(cost)
        if i%10==0:
            print('Y.shape:'+str(Y.shape)+',standard.shape:'+str(standard.shape)+',predict_result.shape:'+str(predict_result.shape))
            print("i:"+str(i)+",cost:"+str(cost)+",accuracy:"+str(accuracy))
    return predict_result,accuracy,costs
'''


def mainlogic(train_x,train_y,test_x,test_y,mini_batch_size,hidden_layer,m,learning_rate,num_epochs,lambd):
    param = init_parameter(hidden_layer)
    ##standard = np.argmax(train_y,axis=0)
    ##standard = train_y.copy()
    standard_test = np.argmax(test_y,axis=0)
    costs = []
    i=0
    seed=5
    ##print(len(mini_batches))
    
    for e in range(num_epochs):
        mini_batch_count=0
        seed = seed + 1
        mini_batches = convert_mini_batch(train_x,train_y,mini_batch_size,m,seed)
        for mini_batch in mini_batches:            
            
            (mini_batch_X,mini_batch_Y) = mini_batch
            standard=np.argmax(mini_batch_Y,axis=0)
            param,cost,cache = forward_and_backward(mini_batch_X,mini_batch_Y,hidden_layer,param,m,learning_rate,lambd)
            predict_result = predict(hidden_layer,param,mini_batch_X,cache)            
            predict_result_test = predict(hidden_layer,param,test_x,cache)
            accuracy = np.sum(predict_result==standard)
            accuracy_test = np.sum(predict_result_test==standard_test)
            costs.append(cost)
            i=i+1
            mini_batch_count=mini_batch_count+1
        ##plot_cost(costs_in_epochs)    
        ##print('standard.shape:'+str(standard.shape)+',mini_batch_Y.shape:'+str(mini_batch_Y.shape)+',predict_result.shape:'+str(predict_result.shape))
        if e%2==0:
            print("i:"+str(i)+",num_epochs:"+str(e)+",mini_batch_count:"+str(mini_batch_count)+",cost:"+str(cost)+",accuracy:"+str(accuracy)+",accuracy_test:"+str(accuracy_test))
        if e%500==0:
            plot_cost(costs)
            
    return predict_result,accuracy,costs


def plot_cost(cost):
    plt.plot(np.squeeze(cost))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
'''
Description:convert_y(y,number_of_type)
    本方法用于转换y，从一维转换为多维
    
Args:
    y:
    y.shape=[1,样本数据数量]
    每个数字代表那个分类，例如下面例子中9代表第9个分类(第9个装箱方案)，10代表第10个分类(第10个装箱方案)
    例子数据如下：
    [[9 10 9 9 0 9 0 0 0 9 0 9 0 0 9 16 16 9 9 9 16 16 16 16 9 9 7 0 9 16 16 16
  16 16 1 16 9 16 9 9 9 9 0 16 0 0 16 9 16 9 16 16 16 0 9 0 16 9 16 16 9 16
  1 16 9 0 9 16 9 0 0 16 16 16 16 0 10 16 0 16 16 16 16 0 17 1 16 16 10 9
  16 0 16 16 16 0 0 9 0 16 16 16 9 16 9 16 16 16 0 9 0 16 16 1 16 9 9 0 9
  16 16 16 1 16 9 9 9 16 16 0 9 16 9 16 9 0 16 16 9 9 9 9 16 9 0 9 0 9 1 16
  9 0 16 16 1 16 16 9 0 9 2 9 16 16 16 0 9 0 0 1 9 16 9 9 16 9 9 9 16 16 16
  0 16 16 16 9 16 16 0 9 9 9 16 16 9 9 16 9 16 16 9 16 1 20 9 9 16 9 0 9 9
  16 0 9 0 16 9 16 0 17 16 16 16 9 16 0...]]
    
    number_of_type:分类数量，即有总共多少类装箱方案

Returns:
    每个样本转换为如以下格式：
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
    例如上例中表示该样本是属于第9个装箱方案
    
    例如入参y为[[0,3,5,1]], number_of_type=6 则经过本方法输出
    [
     [1,0,0,0,0,0],
     [0,0,0,1,0,0],
     [0,0,0,0,0,1],
     [0,1,0,0,0,0]
    ]
    
'''    
def convert_y(y,number_of_type):
    i=0
    print(y)
    keys_dict={}
    for key in np.unique(y):
        keys_dict[key]=i
        i=i+1
    new_y=np.zeros((number_of_type,y.shape[1]))
    for row in range(y.shape[1]):
        q = int(y[:,row])
        new_y[keys_dict[q],row]=1

    print(new_y)
    
    
    return new_y
    
if __name__=="__main__":
    np.set_printoptions(threshold=1000)
    train_x,train_y,test_x,test_y=cartonzation_Data_Load.data_load();
    number_of_type=len(np.unique(train_y))##分类的总类数量
    print('number_of_type:'+str(number_of_type))
    print(np.unique(train_y))
    ##number_of_type = len(cartonzation_result_list) ##分类的总类数量
    ##print('number_of_type:'+str(number_of_type))
    train_y=convert_y(train_y,number_of_type)
    print('train_x.shape:'+str(train_x.shape))
    print('train_y.shape:'+str(train_y.shape))
    ##hidden_layer=[X.shape[0],400,150,60,20,5]
    hidden_layer=[train_x.shape[0],100,50,10,number_of_type]
    #####print(train_y)
    learning_rate=0.2725
    m=train_x.shape[1]
    number_iterator=5000
    num_epochs=10000
    mini_batch_size=1000
    lambd=0.7
    predict_result,accuracy,costs = mainlogic(train_x,train_y,test_x,test_y,mini_batch_size,hidden_layer,m,learning_rate,num_epochs,lambd)
    ##predict_result,accuracy,costs = mainlogic(train_x,train_y,hidden_layer,number_iterator,m,learning_rate)
    plot_cost(costs)
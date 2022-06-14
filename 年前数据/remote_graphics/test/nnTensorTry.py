from cProfile import label
import tensorflow as tf
import numpy as np
import pandas as pd
#创建一个神经网络层
def add_layer(input,in_size,out_size,activation_function=None):

    Weight=tf.Variable(tf.random_normal([in_size,out_size]) )
    biases=tf.Variable(tf.zeros([1,out_size]) +0.1 )
    W_mul_x_plus_b=tf.matmul(input,Weight) + biases
    #根据是否有激活函数
    if activation_function == None:
        output=W_mul_x_plus_b
    else:
        output=activation_function(W_mul_x_plus_b)
    return output
 
# #创建一个具有输入层，隐藏层，输出层的三层神经网络，神经元个数分别为1，10，1
# x_data=np.linspace(-1,1,300)[:,np.newaxis]   # 创建输入数据  np.newaxis分别是在列(第二维)上增加维度，原先是（300，）变为（300，1）
# noise=np.random.normal(0,0.05,x_data.shape)
# y_data=np.square(x_data)+1+noise    # 创建输入数据对应的输出
 
#定义输入数据
xs=tf.placeholder(tf.float32,[None,12])
ys=tf.placeholder(tf.float32,[None,1])
 
#定义一个隐藏层
hidden_layer1=add_layer(xs,12,10,activation_function=tf.nn.relu)
#定义一个输出层
prediction=add_layer(hidden_layer1,10,1,activation_function=None)
 
#求解神经网络参数
#1.定义损失函数
loss=tf.reduce_sum(tf.square(ys-prediction))
#2.定义训练过程
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
 
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

inputs = pd.read_csv("bandsValueTableSB2.csv")
inputs = np.array(inputs)
inputs = inputs[:, 3 : 15]
labels = pd.read_csv("dan_res.csv")
labels = np.array(labels)
labels = [labels[:, 3]]
labels = list(map(list, zip(*labels)))
#3.进行训练
for i in range(1000):
    sess.run(train_step,feed_dict={xs:inputs,ys:labels})
    if i%100==0:
        print(sess.run(loss,feed_dict={xs:inputs,ys:labels} )  )
 
sess.close()
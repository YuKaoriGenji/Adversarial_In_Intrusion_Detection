import tensorflow as tf
import numpy as np
import os
import time
import IDS_Data as ID
batch_size=1
learning_rate=5e-3
epsilon=0.3
def maxPoolLayer(x,kHeight,kWidth,strideX,strideY,name,padding="SAME"):
    print(name,'out----------------------',tf.nn.max_pool(x,ksize=[1,kHeight,kWidth,1],strides=[1,strideX,strideY,1],padding=padding,name=name))
    return tf.nn.max_pool(x,ksize=[1,kHeight,kWidth,1],strides=[1,strideX,strideY,1],padding=padding,name=name)

def dropout(x,keepPro,name=None):
    #return tf.nn.dropout(x,keepPro,name)
    return x

def LRN(x,R,alpha,beta,name=None,bias=1.0):
    return tf.nn.local_response_normalization(x,depth_radius=R,alpha=alpha,beta=beta,bias=bias,name=name)

def fcLayer(x,inputD,outputD,reluFlag,name):
    with tf.variable_scope(name) as scope:
        w=tf.get_variable("w",shape=[inputD,outputD],dtype="float")
        b=tf.get_variable("b",[outputD],dtype="float")
        out=tf.nn.xw_plus_b(x,w,b,name=scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out
def convLayer(x,kHeight,kWidth,strideX,strideY,featureNum,name,padding="SAME",groups=1,mode=0,previous_feature=None,Blank=None):
    channel=int(x.get_shape()[-1])
    conv=lambda a,b:tf.nn.conv2d(a,b,strides=[1,strideY,strideX,1],padding=padding)
    with tf.variable_scope(name) as scope:
        w= tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b= tf.get_variable("b",shape=[featureNum])
        mergeFeatureMap=conv(x,w)

        out=tf.nn.bias_add(mergeFeatureMap,b)
        print(name,'x----------------------',x.get_shape())
        print(name,'w----------------------',w.get_shape())
        print(name,'shape----------------------',mergeFeatureMap.get_shape())
        print(name,'out----------------------',out)
        #return tf.nn.relu(tf.reshape(out,mergeFeatureMap.get_shape()),name=scope.name)
        return tf.nn.relu(out,name=scope.name)
X=tf.placeholder("float",[None,11,12,1])
Y=tf.placeholder("float",[None,5])
keep_prob=tf.placeholder(tf.float32)

conv1=convLayer(X,3,3,1,1,32,"conv1")
conv2=convLayer(conv1,3,3,1,1,64,"conv2")
conv3=convLayer(conv2,2,2,1,1,64,"conv3")
pool1=maxPoolLayer(conv3, 3, 3, 2, 2, "pool1")
fcIn=tf.reshape(pool1,[-1,6*6*64])
fc1=fcLayer(fcIn,6*6*64,1024,True,'fc5')
fc2=fcLayer(fc1,1024,5,True,'fc6')
result=tf.nn.softmax(fc2)
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=result))
train_step=tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#grad_w3,grad_b3=tf.gradients(xs=[w3,b3],ys=cross_entropy)
correct_prediction=tf.equal(tf.argmax(result,1),tf.argmax(Y,1))
pred=tf.argmax(result,1)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
img_grad=tf.gradients(xs=X,ys=cross_entropy)
init=tf.global_variables_initializer()
saver=tf.train.Saver()
print(pool1)
sess=tf.Session()
sess.run(init)
saver.restore(sess,'tmp/model.ckpt')
'''
with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess,'tmp/model.ckpt')
    for  i in range(40000):
        batch_x,batch_y=ID.get_next_batch(batch_size)
        #sess.run(train_step,feed_dict={X:batch_x,Y:batch_y})
        if i%50==0:
            train_loss=sess.run(cross_entropy,feed_dict={X:batch_x,Y:batch_y})
            train_accuracy=accuracy.eval(feed_dict={X:batch_x,Y:batch_y})
            labels=sess.run(result,feed_dict={X:batch_x,Y:batch_y})
            prediction=sess.run(pred,feed_dict={X:batch_x,Y:batch_y})
            y_=sess.run(Y,feed_dict={X:batch_x,Y:batch_y})
            print ("step %d, training loss %g, accuracy %g "%(i,train_loss,train_accuracy))
            print("img_grad:\n",np.array(sess.run(img_grad,feed_dict={X:batch_x,Y:batch_y})).shape)
            print('softmax',labels,'Y:',y_)
            print('prediction:',prediction)
            saver.save(sess,'tmp/model.ckpt')
    for i in range(20):
        batch_x,batch_y=ID.get_next_batch(batch_size)
        print('epoch:',i)
        grad=np.array(sess.run(img_grad,feed_dict={X:batch_x,Y:batch_y}))
        adversarial_img=batch_x+epsilon*np.sign(grad.reshape([batch_size,11,12,1]))
        print('accuracy_noad:',sess.run(accuracy,feed_dict={X:batch_x,Y:batch_y}))
        print('accuracy_adversarial:',sess.run(accuracy,feed_dict={X:adversarial_img,Y:batch_y}))
        random=np.random.rand(batch_size,11,12,1)
'''

def inference(input_x,input_y):
    return sess.run(cross_entropy,feed_dict={X:input_x,Y:input_y})
def predict(input_x,input_y):
    return sess.run(tf.argmax(result,1),feed_dict={X:input_x,Y:input_y})

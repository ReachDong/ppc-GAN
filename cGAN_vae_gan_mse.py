import tensorflow as tf
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
import time
import os
import random
from sklearn.model_selection import KFold 
import os, time, itertools, imageio, pickle

from sklearn.metrics import mean_absolute_error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#two files feature （before） 一个是label(after)
# G: input before+noise     output: 978（before）+10(noise)


mb_size = 64
after_dim = 978
before_dim = 978 #input 978
noise_dim = 1 #input noise 10
h_dim = 15 
V_e1_dim = 500
V_e2_dim = 100



today = time.localtime(time.time())
def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
def generator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()
        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 1000, kernel_initializer=w_init)
        relu1 = tf.nn.relu(dense1)

        dense2 = tf.layers.dense(relu1, 100, kernel_initializer=w_init)
        # o = tf.nn.tanh(dense2)
        
        return dense2
def discriminator(x, y, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        w_init = tf.contrib.layers.xavier_initializer()

        cat1 = tf.concat([x, y], 1)

        dense1 = tf.layers.dense(cat1, 2000, kernel_initializer=w_init)
        lrelu1 = lrelu(dense1, 0.2)

        dense2 = tf.layers.dense(lrelu1, 1, kernel_initializer=w_init)
        o = tf.nn.sigmoid(dense2)

        return o, dense2

""" VAE model var"""
with tf.variable_scope('vae') as scope:
    V_E_W1 = tf.Variable(xavier_init([before_dim , V_e1_dim]),name="V_E_W1")
    V_E_b1 = tf.Variable(tf.zeros(shape=[V_e1_dim]),name="V_E_b1")
    V_E_W2 = tf.Variable(xavier_init([V_e1_dim,V_e2_dim]),name="V_E_W2")
    V_E_b2 = tf.Variable(tf.zeros(shape=[V_e2_dim]),name="V_E_b2")
    V_D_W1 = tf.Variable(xavier_init([V_e2_dim,V_e1_dim]),name="V_D_W1")
    V_D_b1 = tf.Variable(tf.zeros(shape=[V_e1_dim]),name="V_D_b1")
    V_D_W2 = tf.Variable(xavier_init([ V_e1_dim,before_dim]),name="V_D_W2")
    V_D_b2 = tf.Variable(tf.zeros(shape=[before_dim]),name="V_D_b2")
theta_V = [V_E_W1, V_E_b1, V_E_W2, V_E_b2, V_D_W1, V_D_b1, V_D_W2, V_D_b2]


mb_size = 200
after_dim = 978
before_dim = 978 
noise_dim = 5 
lr=0.000001
isTrain=True
train_epoch = 2000

before_tensor = tf.placeholder(tf.float32, shape=[None, before_dim]) # G
real_tensor = tf.placeholder(tf.float32, shape=[None, after_dim])  # D
noise_tensor = tf.placeholder(tf.float32, shape=[None, noise_dim]) # noise
# networks : generator
encode_before = tf.matmul(tf.matmul(before_tensor, V_E_W1)+V_E_b1,V_E_W2)+V_E_b2

G_z = generator(noise_tensor, before_tensor, isTrain)
G_out = tf.matmul(tf.matmul(G_z, V_D_W1)+V_D_b1,V_D_W2)+V_D_b2

# networks : discriminator
encode_real = tf.matmul(tf.matmul(real_tensor, V_E_W1)+V_E_b1,V_E_W2)+V_E_b2
D_real, D_real_logits = discriminator(encode_real, encode_before, isTrain)
D_fake, D_fake_logits = discriminator(G_z, encode_before, isTrain, reuse=True)
# loss for each network
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([mb_size, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([mb_size, 1])))

D_loss = D_loss_real + D_loss_fake
G_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([mb_size, 1])))
G_loss = 0.5* G_loss_1 + 2 * tf.reduce_mean(tf.abs(G_z-encode_real))
# +0.9*tf.reduce_mean(tf.square(G_z-real_tensor))
# trainable variables for each network
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizer for each network
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)



def my_next_feed_dict(index_num,feature_df,label_df):
    ## 需要重写
    before = feature_df.loc[index_num].values
    noise  = np.random.randint( 9,size=(mb_size,noise_dim)) 
    after_real = label_df.loc[index_num].values 

    return noise, before, after_real


if __name__=="__main__":
    

    # results save folder
    root = 'cGAN_final_model_results/'
    model = 'cGAN_mse'
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(root + 'Fixed_results'):
        os.mkdir(root + 'Fixed_results')

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['total_ptime'] = []
    feature = pd.read_csv('../../data/lincs_data/feature_label/bor_data/bor_feature.csv'  ,sep=',',header=-1, low_memory=False)
    label = pd.read_csv('../../data/lincs_data/feature_label/bor_data/bor_label.csv'  ,sep=',',header=-1, low_memory=False)
    feature.fillna(0,inplace=True)
    label.fillna(0,inplace=True)
    feature_df = feature.reset_index(drop=True)
    label_df = label.reset_index(drop=True)
    # training-loop
    my_saver = tf.train.Saver( max_to_keep=1)
    kf = KFold(n_splits=5,shuffle=True)
    dic = pd.DataFrame()
    index = 0
    for train_index, test_index in kf.split(feature_df):
        # open session and initialize all variables
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        model_file=tf.train.latest_checkpoint("./vae_model_path")
        saver=tf.train.Saver(theta_V)
        saver.restore(sess, model_file)

        feature = feature_df.loc[train_index].reset_index(inplace=False,drop=True)
        label =  label_df.loc[train_index].reset_index(inplace=False,drop=True)
        predict_f = feature_df.loc[test_index].reset_index(inplace=False,drop=True)
        predict_l =  label_df.loc[test_index].reset_index(inplace=False,drop=True)

        np.random.seed(int(time.time()))
        print('training start!')
        start_time = time.time()
        for epoch in range(train_epoch):
            G_losses = []
            D_losses = []
            epoch_start_time = time.time()
            for iter in range(feature.shape[0]// mb_size):
                # update discriminator
                index_num = range(iter * mb_size,(iter + 1) * mb_size)
                z_, y_,x_= my_next_feed_dict(index_num,feature,label)
                    
                loss_d_, _ = sess.run([D_loss, D_optim], {real_tensor: x_, before_tensor: y_, noise_tensor: z_})
                D_losses.append(loss_d_)

                # update generator
                # if iter%2 == 0 :
                z_, y_,x_= my_next_feed_dict(index_num,feature,label)
                loss_g_, _ = sess.run([G_loss, G_optim], {real_tensor: x_, before_tensor: y_, noise_tensor: z_})
                G_losses.append(loss_g_)



            mae_temp=[]
            p_before = feature.values
            p_noise = np.random.randint( noise_dim,size=(feature.shape[0],noise_dim))
            p_real = label.values# 后面为用药后的表达值
            samples = sess.run(G_out, feed_dict={before_tensor: feature.values,
                                                    noise_tensor: p_noise})

            for x in range(p_before.shape[0]):
                m = mean_absolute_error(p_real[x],samples[x])
                mae_temp.append(m)          
            print("train_mae_mean:",np.mean(mae_temp),"   train_mae_std: ",np.std(mae_temp)) 
    
            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time
            print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))
            # fixed_p = root + 'Fixed_results/' + model + str(epoch + 1) + '.png'
            # show_result((epoch + 1), save=True, path=fixed_p)
            train_hist['D_losses'].append(np.mean(D_losses))
            train_hist['G_losses'].append(np.mean(G_losses))
            train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
            
            if (epoch+1) % 100 == 0:
                te = pd.DataFrame()
                for x in range(feature.shape[0]):
                    te['before_'+str(x)] = p_before[x]
                    te['standard_'+str(x)] = p_real[x]
                    te['predict__'+str(x)] =samples[x]
                print(te.head(3))

        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)

        p_before = predict_f.values
        p_noise = np.random.randint( noise_dim,size=(predict_f.shape[0],noise_dim))
        p_real = predict_l.values# 后面为用药后的表达值
        samples = sess.run(G_out, feed_dict={before_tensor: p_before,
                                                noise_tensor: p_noise})
        
    
        mse_temp=[]
        for x in range(predict_f.shape[0]):
            if(x % 100 == 1):
                print(x/ predict_f.shape[0])
            dic['before_'+str(index)] = p_before[x]
            dic['standard_'+str(index)] = p_real[x]
            dic['predict__'+str(index)] =samples[x]
            m = mean_absolute_error(p_real[x],samples[x])
            mse_temp.append(m)
            index += 1
        print("test_mae_mean:",np.mean(mse_temp),"   test_mae_std: ",np.std(mse_temp)) 
        # final.columns=col
        print(dic.head(2))
        # print(corr)
        

        print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
        print("Training finish!... save training results")
        with open(root + model + 'train_temp.pkl', 'wb') as f:
            pickle.dump(train_hist, f)

    dic.to_csv('../../data/lincs_data/out/ans/v_gan_mse'+'0.5-2'+'.csv',header=True)

    my_saver.save(sess,'./v_g_m_model/model.ckpt') 

    sess.close()

     

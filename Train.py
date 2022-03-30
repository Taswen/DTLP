import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import re
import Model
import h5py
from Utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# Parameters
block_size  = 48        #Image block size
channel     = 31        #number of spectral bands
batch_size  = 64        #Batch size in training
phase       = 11        #Max phase number of HQS
epoch_num   = 400       #Max epoch number in training
learn_rate  = 0.0001    #Learning rate
rank        = 4         #CP rank

dataset   = 'Harvard'      #Dataset
# 这里不使用检查点（如果是中断以后重启，可以打开）
continueTrain_flag = False #Retrain from the stored model if continueTrain_flag==True

# Date path
train_data_name = './Data/Train/Training_Data_%s_48.mat' % (dataset)
model_dir          = 'Model/%s_%dPhase_%dEpoch_%.5fLearnrate_%dRank/' % (dataset,phase,epoch_num,learn_rate,rank)
output_file_name   = 'Model/Log_%s_%dPhase_%dEpoch_%.5fLearnrate_%dRank.txt' % (dataset,phase,epoch_num,learn_rate,rank)
if not os.path.exists(model_dir):
        os.makedirs(model_dir)

# Load training data
print("...............................")
print("Load training data...")
Training_data   = h5py.File(train_data_name,'r')                # 读取训练用数据文件
Training_labels = Training_data['label']                        # 获取其中数据标签，获取的数据应该是Numpy的数据
Training_labels = np.transpose(Training_labels, (0, 3, 2, 1))   # 调整维度的顺序（batch_size，h,w,c）
del Training_data
nrtrain         = Training_labels.shape[0]                      # 读取第一个维度的长度，代表了有几个样本

# Define variables
gloabl_steps  = tf.Variable(0, trainable=False)                 # 声明一个tf常量，值为0，表示（对象类型是tensor）
                                        #  初始学习率                 用于计算的整体步骤数        衰减间隔步长                           每次衰减的衰减率
learning_rate = tf.train.exponential_decay(learning_rate=learn_rate, global_step=gloabl_steps, decay_steps=(nrtrain//batch_size)*10, decay_rate=0.95,
                                        #  是否使用衰减步长（不使用则每步都更新学习率）
                                           staircase=True)      # 设置学习率指数型衰减
Cu       = tf.placeholder(tf.float32, [None, block_size, block_size, channel])  # 存储 掩膜 的placeholder
X_output = tf.placeholder(tf.float32, [None, block_size, block_size, channel])  # 存储原始数据 的placeholder
b        = tf.zeros(shape=(tf.shape(X_output)[0], channel-1, tf.shape(X_output)[2], tf.shape(X_output)[3])) # 

# Forward imaging and Initialization
y  = Encode_CASSI(X_output,Cu)
x0 = Init_CASSI(y,Cu,channel)

# Model
# 调用自定义的  Model.Interface 计算预测
Prediction = Model.Interface(x0, Cu, phase, rank, channel, reuse=False)
# 设定损失函数的计算方法： 所有的对应位置相减 然后平方 再把所有的值压缩到一个平均值
cost_all = tf.reduce_mean(tf.square(Prediction - X_output))
# 下面语句返回指定了 AdamOptimizer 方法的优化操作
# 其调用在 TF1.15 以后将被遗弃，使用 tf.compat.v1.train.AdamOptimizer() 替代
optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all, global_step=gloabl_steps)
# 下面语句返回初始化的操作
init = tf.global_variables_initializer()

config = tf.ConfigProto()
# config = tf.ConfigProto(log_device_placement=True)    # 带上log_device_placement参数可以的打印任务分配的位置
config.gpu_options.allow_growth = True
# 创建状态管理器
saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
# 创建本次任务会话
sess = tf.Session(config=config)
# 执行 对任务中涉及到的变量进行初始化 的操作
sess.run(init)


print("Training samples number: %d" % (nrtrain))
print("Phase number: %d" % (phase))
print("Image block size: %d" % (block_size))
print("Max epoch number: %d" % (epoch_num))
print("CP rank: %s" % (rank))
print("Dataset: %s" % (dataset))
print("...............................\n")

# Retrain from the stored model (if continueTrain_flag == True)
if continueTrain_flag:
    # 从检查点文件恢复检查点状态， ckpt CheckpointState None/otherwise
    ckpt     = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
       saver.restore(sess, ckpt.model_checkpoint_path) # 恢复状态
       ckpt_num_seq = re.findall(r"\d+\d*",ckpt.model_checkpoint_path)  # 获取会话
    ckpt_num = int(ckpt_num_seq[-1])        #
else:
    ckpt_num = -1




#  Training
print('Initial epoch: %d' % (ckpt_num+1))
print("Strart Training...")
# 从 起始epoch轮次 到设定好的最大轮次 epoch_num(400)
for epoch_i in range(ckpt_num+1, epoch_num):
    randidx_all = np.random.permutation(nrtrain)                            # 对本 epoch 的样本进行打乱
    
    for batch_i in range(nrtrain // batch_size):                            # 切分 batch 进行本 epoch 的迭代
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]    # 获取 batch 切片序号
        batch_ys = Training_labels[randidx, :, :, :]                        # 提取本 batch 训练样本的labels

        # 随机得到掩膜
        Cu_input = np.zeros([block_size, block_size, channel])              # 初始化编码孔径
        T = np.round(np.random.rand(block_size/2, block_size/2))            # 随机分布
        T = np.concatenate([T,T],axis=0)                                    
        T = np.concatenate([T,T],axis=1)                                    # 上两步横向和纵向都两倍扩展
        for ch in range(channel):
            Cu_input[:,:,ch] = np.roll(T, shift=-ch, axis=0)                # 在height维度进行移位操作
        Cu_input = np.expand_dims(Cu_input, axis=0)                         # 升维，最前面多出一个 batch_size 的维度
        Cu_input = np.tile(Cu_input, [batch_size, 1, 1, 1])                 # height, width, channel,方向上不变化，在channel方向复制 batch_size 份

        # 开始训练
        feed_dict = {X_output: batch_ys, Cu: Cu_input}                      # 将本 batch 的 label 和 掩膜组合，分布填充到占位符中
        sess.run(optm_all, feed_dict=feed_dict)                             # 执行 训练中的计算 操作。 这一步才是开始数据的处理

        # 执行 训练中的计算损失值和改变学习率 的操作。 
        output_data = "[%03d/%03d/%03d] cost: %.6f  learningrate: %.6f \n" % (batch_i, nrtrain // batch_size, epoch_i, sess.run(cost_all, feed_dict=feed_dict), sess.run(learning_rate, feed_dict=feed_dict))
        print(output_data)                                                  # 输出一个batch内的训练结果

    # 记录一次epoch训练完的结果
    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver.save(sess, './%s/model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)       # 存储一个epoch的checkpoint
sess.close()
print("Training Finished")


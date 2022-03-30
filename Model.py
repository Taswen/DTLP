import tensorflow as tf
import numpy as np

# 获取卷积核的权重，使用自带的初始化器
# 卷积核的各维度定义为
#   [filter_height, filter_width, in_channels, out_channels]
def def_con2d_weight(w_shape, w_name):
    # Define the net weights
    weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%s' % w_name)
    return weights

# 残差块
# input 上一次输入，
def ResBlock(input, order, filter_num):
    # Residual block for DRTLM
	xup   = RTGB(input, 'up_order_%d' % (order), filter_num)
	x_res = input - xup         # 获取残差块
    # 累加操作
	xdn   = RTGB(x_res, 'down_order_%d' % (order), filter_num)
	xdn   = xdn + x_res
	return xup, xdn

# RTGB 操作操作块的定义
# input 输入的图像数据，order_name 每阶张量的命名，filter_num
def RTGB(input, order_name, filter_num=64):
    # Rank-1 tensor generating block
    # 分别在不同维度（height，width，channel）应用全局池化
    # [1,height,1,1]
	gap_Height   =  tf.reduce_mean(tf.reduce_mean(input, axis=2, keepdims=True), axis=3, keepdims=True) # 先平均 width，再平均channel
	# [1,1,width,1]
	gap_Weight   =  tf.reduce_mean(tf.reduce_mean(input, axis=1, keepdims=True), axis=3, keepdims=True) # 先平均 height，再平均channel
	# [1,1,1,channel]
	gap_Channel  =  tf.reduce_mean(tf.reduce_mean(input, axis=1, keepdims=True), axis=2, keepdims=True) # 先平均 height，再平均width
    
    # 定义三个通道卷积核
	weights_H = def_con2d_weight([1, 1, 1, 1], 'cp_con1d_hconv_%s' % (order_name))
	weights_W = def_con2d_weight([1, 1, 1, 1], 'cp_con1d_wconv_%s' % (order_name))
	weights_C = def_con2d_weight([1, 1, filter_num, filter_num], 'cp_con1d_cconv_%s' % (order_name))

    # 1*1 卷积操作
    # https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    # [batch,height,1,1]
	convHeight_GAP    = tf.nn.sigmoid(tf.nn.conv2d(gap_Height, weights_H, strides=[1, 1, 1, 1], padding='SAME'), name = 'sig_hgap_%s' % (order_name));
	# [batch,1,width,1]
	convWeight_GAP    = tf.nn.sigmoid(tf.nn.conv2d(gap_Weight, weights_W, strides=[1, 1, 1, 1], padding='SAME'), name = 'sig_wgap_%s' % (order_name));
	# [batch,1,1,filter_num]
    # 这里 padding 是 'SAME' 表示会进行填充， 解决了本应是channel但是确是 filter_num的问题
	convChannel_GAP   = tf.nn.sigmoid(tf.nn.conv2d(gap_Channel, weights_C, strides=[1, 1, 1, 1], padding='SAME'), name = 'sig_cgap_%s' % (order_name));

    # 为了后续Kronecker 调整shape 为 batch *（一个二维的维度）
	vecConHeight_GAP  = tf.reshape(convHeight_GAP, [tf.shape(convHeight_GAP)[0], tf.shape(convHeight_GAP)[1],1])
	vecConWeight_GAP  = tf.reshape(convWeight_GAP, [tf.shape(convWeight_GAP)[0], 1, tf.shape(convWeight_GAP)[2]])
    # 这里会将channel打平
	vecConChannel_GAP = tf.reshape(convChannel_GAP, [tf.shape(convChannel_GAP)[0], 1, tf.shape(convChannel_GAP)[3]])

	matHWmulT    = tf.matmul(vecConHeight_GAP, vecConWeight_GAP) # 结果维度为 [batch,h,w]
    # reshape 以进行后续操作 调整shape 为 [batch,h*w,1]
	vecHWmulT    = tf.reshape(matHWmulT, [tf.shape(matHWmulT)[0], tf.shape(matHWmulT)[1] * tf.shape(matHWmulT)[2], 1])
	matHWCmulT   = tf.matmul(vecHWmulT, vecConChannel_GAP)      # 结果维度为 [batch,h*w, 64]
    # reshape 为与输入的尺寸相同
	recon        = tf.reshape(matHWCmulT, [tf.shape(input)[0], tf.shape(input)[1], tf.shape(input)[2], tf.shape(input)[3]])
	return recon

def DRTLM(input, rank, filter_num):
    #  Discriminative rank-1 tensor learning module
    # 先操作一次获取初始的残差块
	(xup, xdn) = ResBlock(input, 0, filter_num)
	temp_xup   = xdn
	output     = xup
    # 操作 rank-1 次
	for i in range(1,rank):
		(temp_xup,temp_xdn) = ResBlock(temp_xup, i, filter_num)
		xup      = xup + temp_xup  
        # 拼接堆叠 相当于 concat     
		output   = tf.concat([output, xup],3)
		temp_xup = temp_xdn
	return output

def Encoding(input, filter_size, filter_num):
    # Get deep feature maps
    # 获取一重卷积核权重
    weights_pro_0 = def_con2d_weight([filter_size, filter_size, filter_num, filter_num], 'fproject_con2d_conv_0')
    # 先卷积（考虑边界不足的时候用0补充）再reLU
    input_temp    = tf.nn.relu(tf.nn.conv2d(input, weights_pro_0, strides=[1, 1, 1, 1], padding='SAME'))
    # 获取二重卷积核权重
    weights_pro_1 = def_con2d_weight([filter_size, filter_size, filter_num, filter_num], 'fproject_con2d_conv_1')
    # 进行二次卷积
    output        = tf.nn.conv2d(input_temp, weights_pro_1, strides=[1, 1, 1, 1], padding='SAME')
    return output

# 将多个1阶低秩张量合并成一个完整的张量，并进行卷积合并
def Fusion(input, xt, filter_size, filter_num, channel_num):
    # Aggregate multiple rank-1 tensors into a low-rank tensor
    # 定义卷积操作核
    weights_attention = def_con2d_weight([filter_size, filter_size, filter_num, channel_num], 'IRecon_attention_con2d_conv')
    # 卷积降维操作，将concat后的低秩张量块卷积为指定channel（64）大小的
    attention_map     = tf.nn.conv2d(input, weights_attention, strides=[1, 1, 1, 1], padding='SAME')
    # 点积操作，将获取的融合结果
    output            = tf.multiply(xt,attention_map)
    return output

# 重构函数
# xt上一个phase的数据 x0初始数据 Cu掩膜 layer_no层的序号 channel光谱通道数 rank CP分解的阶
def Recon(xt, x0, Cu, layer_no, channel = 31, rank = 4):
    # Parameters
    deta        = tf.Variable(0.04, dtype=tf.float32, name='deta_%d' % layer_no)    # Δ/ε
    eta         = tf.Variable(0.8, dtype=tf.float32, name='eta_%d' % layer_no)      # η/τ
    filter_size = 3     # 3x3卷积核
    filter_num  = 64

    # 定义卷积操作
    # 第一次卷积发生在编码之前，目的是把通道数从31变为64
    weights_main_0 = def_con2d_weight([filter_size, filter_size, channel, filter_num], 'main_con2d_conv_0')
    # 第二次卷积发生在解码时，目的是把通道数从64变为重新变为31
    weights_main_1 = def_con2d_weight([filter_size, filter_size,  filter_num, channel], 'main_con2d_conv_1')
    
    # TLPLN
    # Low-rank Tensor Recovery
    x_feature_0        = tf.nn.conv2d(xt, weights_main_0, strides=[1, 1, 1, 1], padding='SAME')     # 这两步是 Feature Coding
    x_feature_1        = Encoding(x_feature_0, filter_size, filter_num)                             
    attention_map_cat  = DRTLM(x_feature_1, rank, filter_num)                                       # 这一步是进入 DRTLM

    # 融合 concat后的低秩张量以及编码后得到的原始编码
    x_feature_lowrank  = Fusion(attention_map_cat, x_feature_1, filter_size, filter_num * rank, filter_num)
    
    # 以下两步为解码操作
    # 与 Feature Coding的进行相加
    x_mix              = x_feature_lowrank + x_feature_0
    # 卷积 + relu 
    z  = tf.nn.relu(tf.nn.conv2d(x_mix, weights_main_1, strides=[1, 1, 1, 1], padding='SAME'))

    # Linear Projection
    yt  = tf.multiply(xt, Cu)   # Phi*xt                # 对位相乘，即对应元素相乘
    yt  = tf.reduce_sum(yt, axis=3)                     # 把光谱维加和成一个数，降维
    yt1 = tf.expand_dims(yt, axis=3)                    # 在光谱维升维
    yt2 = tf.tile(yt1, [1, 1, 1, channel])              # 复制光谱维升到光谱的通道数
    xt2 = tf.multiply(yt2, Cu)  # PhiT*Phi*xt           # 对位相乘，即对应元素相乘
    # tf.scalar_mul：将标量乘以一个张量(Tensor)或 IndexedSlices 对象.
    x   = tf.scalar_mul(1-deta*eta, xt) - tf.scalar_mul(deta, xt2) + tf.scalar_mul(deta, x0) + tf.scalar_mul(deta*eta, z)
    return x                # 得到的重建数据作为下一个 phase 的输入

# 训练的计算过程
def Interface(x, Cu, phase, rank, channel, reuse):
    '''
    Input parameters:
    x-----Initialized image                 这里是原始高光谱 经过 CASSI系统编码过的图像
    Cu----CASSI mask                        编码时用的掩膜
    phase-----Max phase number              每次训练中 TLPLN网络与 线性投影的 交替的次数。
    rank---CP rank                          CP分解的阶数
    channel---Spectral band number          光谱维的通道大小

    Output parameters:
    xt----Reconstructed image               来回倒腾后重建出来的图像
    '''

    xt = x
    for i in range(phase):
        with tf.variable_scope('Phase_%d' %i, reuse=reuse):
            xt = Recon(xt, x, Cu, i, channel, rank)
    return xt
from __future__ import division
import tensorflow as tf
import numpy as np 
import os.path
from PIL import Image
import sys
import datetime
import os, glob, cv2, random, re, time
from multiprocessing import Process, Queue
import math
import matplotlib.pyplot as plt
import logging


os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
border_width = sys.argv[1]
#branch = sys.argv[3]		#Baseline / Margin
#branch = 'Baseline'
is_testing = True 

Database = 'headpose'
log_dir = './{}/Aflw/{}/log'.format(Database, border_width)
ckpt_dir = './{}/Aflw/{}/ckpt'.format(Database, border_width)
if not os.path.exists(log_dir): os.makedirs(log_dir)
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
# logging to file
logging.basicConfig(filename=log_dir+'/{}.log'.format(Database),filemode='a', level=logging.DEBUG,format='%(asctime)s, %(msecs)d %(message)s',datefmt='%H:%M:%S')
# logging to stream
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


##################
#
##################

def crop_image_black_border(image_path, label_path, border_width):
	img_list = open(image_path).readlines()	#'./AFLW_meta.tsv'
	label_list = open(label_path).readlines()
	data = []
	
	for index in range(len(img_list)):
		if(index % 5000 == 0):
			print(str(index) + "	Hit!")
		line_image = img_list[index]
		line_label = label_list[index]
		
		items = line_image.split('\t')
		if (len(items) != 2 or items[1]==' \n' or items[1]=='\n'):
			continue

		img_dir = os.path.join('./', items[0])

		img = cv2.imread(img_dir)
		try:
			x_set = int(items[1].split(' ')[1])
			y_set = int(items[1].split(' ')[0])
		except Exception as e:
			print(img_dir, items[1])
		
		if (x_set<0 or y_set<0):
			continue

		offset = int(float(items[1].split(' ')[2]))
		if (np.float(border_width) == 0):
			_offset = 0
		else:
			_offset = int(offset / np.float32(border_width))
		x_start = x_set + offset
		y_start = y_set + offset
		x_end = x_start + offset
		y_end = y_start + offset

		src = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[0,0,0])
		
		tmp_image = src[y_start - _offset : y_end + _offset, x_start - _offset : x_end + _offset, :]

		tmp_image = cv2.resize(tmp_image, (224,224))

		items_label = line_label.split('\t')
		R_Angle = np.float32(items_label[1])
		P_Angle = np.float32(items_label[2])
		Y_Angel = np.float32(items_label[3])
		R_Class = np.float32(items_label[4])
		P_Class = np.float32(items_label[5])
		Y_Class = np.float32(items_label[6][:-1])

		data.append([tmp_image, np.array([R_Angle, P_Angle, Y_Angel, R_Class, P_Class, Y_Class])])
	return data


def Data_Loader(queue, data, batch_size, epoch, shuffle = True):
	rng = np.random.RandomState()
	for k in range(int(epoch * 1.2)):
		if shuffle:
			rng.shuffle(data)
			for item in range(0, len(data)-batch_size, batch_size):
				X = []
				Y = []
				for index in range(item, item+batch_size):
					tmp_image = data[index][0]
					tmp_label = data[index][1]
					
					X.append(np.array(tmp_image))
					Y.append(tmp_label)
				np.transpose(Y, (1, 0))
				queue.put((np.array(X), np.array(Y)), block = True)
		else:
			pass
	return

def consumer(queue):
	X, Y = queue.get(block = True)
	return X, Y

def res_block(feature_in, out_channel, stride):
	current = tf.layers.conv2d(inputs=feature_in, filters=out_channel, strides=stride, kernel_size=1, padding="same")
	current = tf.layers.batch_normalization(inputs=current)
	current = tf.nn.relu(current)

	current = tf.layers.conv2d(inputs=current, filters=out_channel, strides=1, kernel_size=3, padding="same")
	current = tf.layers.batch_normalization(inputs=current)
	current = tf.nn.relu(current)

	current = tf.layers.conv2d(inputs=current, filters=out_channel*4, strides=1, kernel_size=1, padding="same")
	current = tf.layers.batch_normalization(inputs=current)

	return current


def resnet():
#The input shape (N, 224, 224, 3)
	def f(img):
		tmp = tf.layers.conv2d(inputs=img, filters=64, strides=2, kernel_size=7, padding="same")
		tmp = tf.layers.batch_normalization(inputs=tmp)
		tmp = tf.nn.relu(tmp)
		#tmp = tf.nn.pool()		couldn't fix it
		tmp = tf.nn.max_pool(value=tmp, ksize=[1,3,3,1], padding="SAME", strides=[1,2,2,1])

		res_bock_out = res_block(tmp, 64, 1)
		tmp_1 = tf.layers.conv2d(inputs=tmp, filters=256, strides=1, kernel_size=1, padding="same")
		tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
		tmp = tf.add(res_bock_out, tmp_1)
		tmp = tf.nn.relu(tmp)

		for i in range(2):
			res_bock_out = res_block(tmp, 64, 1)
			tmp = tf.add(res_bock_out, tmp)
			tmp = tf.nn.relu(tmp)

		res_bock_out = res_block(tmp, 128, 2)
		tmp_1 = tf.layers.conv2d(inputs=tmp, filters=512, strides=2, kernel_size=1, padding="same")
		tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
		tmp = tf.add(res_bock_out, tmp_1)
		tmp = tf.nn.relu(tmp)

		for i in range(3):
			res_bock_out = res_block(tmp, 128, 1)
			tmp = tf.add(res_bock_out, tmp)
			tmp = tf.nn.relu(tmp)

		res_bock_out = res_block(tmp, 256, 2)
		tmp_1 = tf.layers.conv2d(inputs=tmp, filters=1024, strides=2, kernel_size=1, padding="SAME")
		tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
		tmp = tf.add(res_bock_out, tmp_1)
		tmp = tf.nn.relu(tmp)

		for i in range(5):

			res_bock_out = res_block(tmp, 256, 1)
			tmp = tf.add(res_bock_out, tmp)
			tmp = tf.nn.relu(tmp)

		res_bock_out = res_block(tmp, 512, 2)
		tmp_1 = tf.layers.conv2d(inputs=tmp, filters=2048, strides=2, kernel_size=1, padding="same")
		tmp_1 = tf.layers.batch_normalization(inputs=tmp_1)
		tmp = tf.add(res_bock_out, tmp_1)
		tmp = tf.nn.relu(tmp)

		for i in range(2):
			res_bock_out = res_block(tmp, 512, 1)
			tmp = tf.add(res_bock_out, tmp)
			tmp = tf.nn.relu(tmp)
		out = tf.nn.avg_pool(value=tmp, ksize=[1,7,7,1], padding="SAME", strides=[1,7,7,1])

		return out
	return f

############################
def crop_image_black_border_val(image_path, label_path, border_width):
        img_list = open(image_path).readlines() #'./AFLW_meta.tsv'
        label_list = open(label_path).readlines()
        data = []

        for index in range(len(img_list)):
                if(index % 5000 == 0):
                        print(str(index) + "    Hit!")
                line_image = img_list[index]
                line_label = label_list[index]

                items = line_image.split('\t')
#               if (len(items) != 2 or items[1]==' \n' or items[1]=='\n'):
#                       continue
                img_dir = os.path.join('./', items[0])
                img = cv2.imread(img_dir)
                try:
                        x_set = int(items[1].split(' ')[1])
                        y_set = int(items[1].split(' ')[0])
                except Exception as e:
                        print(img_dir, items[1])

                if (x_set<0 or y_set<0):
                        continue
                
		offset = int(float(items[1].split(' ')[2]))
                if (np.float(border_width) == 0):
                        _offset = 0
                else:
                        _offset = int(offset / np.float32(border_width))
                x_start = x_set + offset
                y_start = y_set + offset
                x_end = x_start + offset
                y_end = y_start + offset

                src = cv2.copyMakeBorder(img, offset, offset, offset, offset, cv2.BORDER_CONSTANT, value=[0,0,0])

                tmp_image = src[y_start - _offset : y_end + _offset, x_start - _offset : x_end + _offset, :]

                tmp_image = cv2.resize(tmp_image, (224,224))

                items_label = line_label.split('\t')
                R_Angle = np.float32(items_label[1])* math.pi /180
                P_Angle = np.float32(items_label[2])* math.pi /180
                Y_Angel = np.float32(items_label[3])* math.pi /180

                data.append([tmp_image, np.array([R_Angle, P_Angle, Y_Angel])])
        return data

############################

if __name__ == '__main__':


	train_image = './300wcrop.txt'	
	train_label = './300w_euler_cls.txt'
	val_image = './Aflw2000crop.txt'
	val_label = './Aflw2000_euler_cls.txt'
		
	logging.info('<<<STARTING {} {}>>>'.format(border_width, datetime.datetime.now()))

	#(image_path, label_path, border_width):
	data_train = crop_image_black_border(train_image, train_label, border_width)
	data_val = crop_image_black_border_val(val_image, val_label, border_width)
	
	epoch = 100
	BATCH_SIZE = 64 

	
	queue = Queue(maxsize = 32)
	processes = [Process(target = Data_Loader, args = (queue, data_train, BATCH_SIZE, epoch)) for x in range(16)]
	with tf.Graph().as_default():
		
		X = tf.placeholder(shape=(None, None, None, 3), dtype=tf.float32)
		Y = tf.placeholder(shape=(None, None), dtype=tf.float32)

		R_Angle_label = tf.expand_dims(tf.cast(Y[:,0], tf.float32), 1)
		P_Angle_label = tf.expand_dims(tf.cast(Y[:,1], tf.float32), 1)
		Y_Angle_label = tf.expand_dims(tf.cast(Y[:,2], tf.float32), 1)
		R_Class_label = tf.one_hot(tf.cast(Y[:,3], tf.uint8), 181)
		P_Class_label = tf.one_hot(tf.cast(Y[:,4], tf.uint8), 181)
		Y_Class_label = tf.one_hot(tf.cast(Y[:,5], tf.uint8), 181)
		
		# is_training = tf.placeholder(shape=(), dtype=tf.bool)
	#	is_adv = tf.placeholder(shape=(), dtype=tf.bool)
	#	lr = tf.Variable(1e-3, name='learning_rate', trainable=False, dtype=tf.float32)
		tf.summary.image('X', X, max_outputs=3)	
		cnn = resnet()
		out = cnn(X)

		R_Angle_pred = tf.layers.dense(inputs=out, units=1)
		P_Angle_pred = tf.layers.dense(inputs=out, units=1)
		Y_Angle_pred = tf.layers.dense(inputs=out, units=1)
		R_Class_pred = tf.layers.dense(inputs=out, units=181)
		P_Class_pred = tf.layers.dense(inputs=out, units=181)
		Y_Class_pred = tf.layers.dense(inputs=out, units=181)

	# For Training
		loss_R_A = 2000 * tf.reduce_mean(tf.square(tf.squeeze(R_Angle_pred) - tf.squeeze(R_Angle_label)))
		loss_P_A = 2000 * tf.reduce_mean(tf.square(tf.squeeze(P_Angle_pred) - tf.squeeze(P_Angle_label)))
		loss_Y_A = 2000 * tf.reduce_mean(tf.square(tf.squeeze(Y_Angle_pred) - tf.squeeze(Y_Angle_label)))
		loss_R_C = 0.1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=R_Class_label, logits=R_Class_pred))
		loss_P_C = 0.1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=P_Class_label, logits=P_Class_pred))
		loss_Y_C = 0.1 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_Class_label, logits=Y_Class_pred))

		# incase you want to do some operation on the loss weight
		loss = loss_R_A + loss_P_A + loss_Y_A + loss_R_C + loss_P_C + loss_Y_C
		tf.summary.scalar('loss', loss)
		tf.summary.scalar('loss_R_A', loss_R_A)
		tf.summary.scalar('loss_P_A', loss_P_A)
		tf.summary.scalar('loss_Y_A', loss_Y_A)
		tf.summary.scalar('loss_R_C', loss_R_C)
		tf.summary.scalar('loss_P_C', loss_P_C)
		tf.summary.scalar('loss_Y_C', loss_Y_C)
		
		opt = tf.train.GradientDescentOptimizer(learning_rate=1e-4).minimize(loss)

	# For Valuation
		loss_val_R = tf.abs(tf.squeeze(R_Angle_pred) - tf.squeeze(R_Angle_label)) *  180 /math.pi
		loss_val_P = tf.abs(tf.squeeze(P_Angle_pred) - tf.squeeze(P_Angle_label)) *  180 /math.pi
		loss_val_Y = tf.abs(tf.squeeze(Y_Angle_pred) - tf.squeeze(Y_Angle_label)) *  180 /math.pi
	 	
		loss_val = [tf.squeeze(loss_val_R), tf.squeeze(loss_val_P), tf.squeeze(loss_val_Y)]
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.gpu_options.per_process_gpu_memory_fraction = 1.0
		

		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			sess.run(tf.local_variables_initializer())

			# var = tf.global_variables()
			# var_to_restore = [val for val in var if (('conv2d' in val.name) or ('batch_normalization' in val.name))]
			saver = tf.train.Saver()
			# saver.restore(sess, './headpose/Cls/{}/ckpt/model90.ckpt'.format(border_width))#Adversarial/ckpt/model599999.ckpt')	#ISBI_AM_64_0.0
		
			merged = tf.summary.merge_all()
			writer = tf.summary.FileWriter(log_dir,sess.graph)
			saver_all = tf.train.Saver(max_to_keep=1000)

			for p in processes:
				p.start()

			for e in range(epoch):
				for item in range(int(len(data_train)/BATCH_SIZE)):
					batch_x, batch_y =  consumer(queue)
					summary, _, cl = sess.run([merged, opt, loss], feed_dict={X: batch_x, Y: batch_y})
				
					writer.add_summary(summary, item + (int(len(data_train)/BATCH_SIZE)) * e)
			

			#######Val Part#######

				Val_loss = [0.0, 0.0, 0.0]	
				for line in data_val:
					cl = sess.run([loss_val], feed_dict={X: np.expand_dims(line[0], 0), Y:np.expand_dims(line[1], 0)})
					# print(R.tolist(), P.tolist(), Y.tolist(), line[1])
					Val_loss += np.array(cl)
			
				logging.info(' ========= {} ========= '.format(e))
				logging.info('cl: {}, {}'.format(np.array(Val_loss)/len(data_val), np.array(Val_loss).sum()/len(data_val)/3))
				
			
				save_path = saver_all.save(sess, "{}/model{}.ckpt".format(ckpt_dir, e))
				print ('Model saved in file: %s' % save_path)

			print datetime.datetime.now()
			queue.close()
			for p in processes:
				p.terminate()
	

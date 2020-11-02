
import sys
if sys.version_info.major == 2:
  import Queue as queue
else:
  import queue
import traceback
import tensorflow as tf
import predict.distribution as distribution

import sys
import os
import json
import re
import time
import tensorflow as tf
import numpy as np

from train.modeling import GroverModel, sample

class PredictProcess(distribution.Process):
	""" Prediction process for tf saved model """
	def __init__(self,
				 news_config,
				 ckpt_fn,
				 thread_num=1,
				 input_queue=None,
				 output_queue=None,
				 batch_size=1,
				 job_name="ez_transfer_job"):
		super(PredictProcess, self).__init__(job_name,
											 thread_num,
											 input_queue=input_queue,
											 output_queue=output_queue,
											 batch_size=batch_size)
		self.graph = tf.Graph()
		self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9,
									allow_growth=False)
		self.session_conf = tf.ConfigProto(
			  intra_op_parallelism_threads=8,
			  inter_op_parallelism_threads=8,
			  allow_soft_placement=True,
			  gpu_options=gpu_options)

		with self.graph.as_default():

			self.sess = tf.Session(config=self.session_conf)
			self.initial_context = tf.placeholder(tf.int32, [batch_size, None])
			self.p_for_topp = tf.placeholder(tf.float32, [batch_size])
			self.eos_token = tf.placeholder(tf.int32, [])
			self.min_len = tf.placeholder(tf.int32, [])
			self.max_len = tf.placeholder(tf.int32, [])
			self.k_for_topk = tf.placeholder(tf.int32, [])
			self.tokens, self.probs = sample(news_config=news_config, 
									initial_context=self.initial_context,
									eos_token=self.eos_token, 
									min_len=self.min_len, 
									max_len=self.max_len,
									ignore_ids=None, 
									p_for_topp=self.p_for_topp,
									k_for_topk=self.k_for_topk,
									do_topk=False)
			self.saver = tf.train.Saver()
			self.saver.restore(self.sess, ckpt_fn)

			self.input_dict = {
				"initial_context":self.initial_context,
				"p_for_topp":self.p_for_topp,
				"eos_token":self.eos_token,
				"min_len":self.min_len,
				"max_len":self.max_len,
				"k_for_topk":self.k_for_topk
			}

			self.predictions = {
				"tokens":self.tokens,
				"probs":self.probs
			}

	def process(self, in_data):
		with self.graph.as_default():
			predictions = self.sess.run(
				self.predictions, feed_dict={
					self.input_dict[key]: in_data[key]
					for key in self.input_dict})
		ret = {}
		for key, val in predictions.items():
			ret[key] = val
		return ret

	def destroy(self):
		with self.graph.as_default():
			self.sess.close()


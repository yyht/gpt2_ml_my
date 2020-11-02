
import sys
if sys.version_info.major == 2:
  import Queue as queue
else:
  import queue
import traceback
import tensorflow as tf
import predict.distribution as distribution
from predict import preprocessors, postprocessors
from predict.app_utils import get_reader_fn, get_writer_fn
from predict.predictor import PredictProcess 

class AppPredictor(object):
    """ Application predictor (support distributed predicting) """
    def __init__(self, config, 
                 thread_num=1, 
                 queue_size=256,
                 job_name="app_predictor"):

        self.config = config
        self.worker_id = config.task_index
        self.num_workers = len(config.worker_hosts.split(","))
        self.thread_num = thread_num
        self.queue_size = queue_size
        self.job_name = job_name
        self.news_config = config.news_config
        self.ckpt_fn = config.ckpt_fn

    def get_default_reader(self):
        return get_reader_fn()(input_glob=self.config.predict_input_fp,
                               input_schema=self.config.input_schema,
                               is_training=False,
                               batch_size=self.config.predict_batch_size,
                               output_queue=queue.Queue(),
                               slice_id=self.worker_id,
                               slice_count=self.num_workers)

    def get_default_writer(self):
        return get_writer_fn()(output_glob=self.config.predict_output_fp,
                               output_schema=self.config.output_schema,
                               slice_id=self.worker_id,
                               input_queue=queue.Queue())

    def get_predictor(self):
        predictor = PredictProcess(news_config=self.news_config,
                                   ckpt_fn=self.ckpt_fn,
                                   thread_num=self.thread_num,
                                   input_queue=queue.Queue(),
                                   output_queue=queue.Queue(),
                                   job_name=self.job_name)
        return predictor

    def run_predict(self, reader=None, preprocessor=None, postprocessor=None, writer=None):
        self.proc_executor = distribution.ProcessExecutor(self.queue_size)
        reader = reader if reader else self.get_default_reader()
        reader.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(reader)
        preprocessor = preprocessor
        preprocessor.input_queue = self.proc_executor.get_input_queue()
        preprocessor.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(preprocessor)
        predictor = self.get_predictor()
        predictor.input_queue = self.proc_executor.get_input_queue()
        predictor.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(predictor)
        posprocessor = postprocessor
        posprocessor.input_queue = self.proc_executor.get_input_queue()
        posprocessor.output_queue = self.proc_executor.get_output_queue()
        self.proc_executor.add(posprocessor)
        writer = writer if writer else self.get_default_writer()
        writer.input_queue = self.proc_executor.get_input_queue()
        self.proc_executor.add(writer)
        self.proc_executor.run()
        self.proc_executor.wait()
        writer.close()
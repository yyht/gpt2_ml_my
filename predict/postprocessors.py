from predict.distribution import Process
import numpy as np
import six

class GPTPostprocessor(Process):
    def __init__(self,
                 config,
                 prediction_colname="predictions",
                 thread_num=None,
                 input_queue=None,
                 output_queue=None,
                 job_name='GPTPostprocessor'):

        super(MyPostprocessor, self).__init__(
            job_name, thread_num, input_queue, output_queue)

    def process(self, in_data):
        ret = {key: val for key, val in in_data.items()}
        return ret
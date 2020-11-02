from predict.distribution import Process
import numpy as np
import six

class LMPostprocessor(Process):
    """ Postprocessor for text classification, convert label_id to the label_name

    """
    def __init__(self,
                 output_schema,
                 thread_num=None,
                 input_queue=None,
                 output_queue=None,
                 prediction_colname="predictions",
                 job_name='LMpostprocessor'):

        super(LMpostprocessor, self).__init__(
            job_name, thread_num, input_queue, output_queue, batch_size=1)
        self.prediction_colname = prediction_colname
        self.output_schema = output_schema

    def process(self, in_data):
        """ Post-process the model outputs

        Args:
            in_data (`dict`): a dict of model outputs
        Returns:
            ret (`dict`): a dict of post-processed model outputs
        """
        return in_data
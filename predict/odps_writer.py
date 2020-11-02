# coding=utf-8
# Copyright (c) 2019 Alibaba PAI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
from predict.distribution import Process, Counter

class OdpsTableWriter(Process):
    """ Writer odps table

    Args:

        output_glob : output file fp
        output_schema : output_schema

    """
    def __init__(self,
                 output_glob,
                 output_schema,
                 slice_id,
                 input_queue,
                 job_name='DistOdpsTableWriter',
                 **kwargs):
        super(OdpsTableWriter, self).__init__(job_name, 1, input_queue)

        self.table_writer = tf.python_io.TableWriter(output_glob, slice_id=slice_id)

        self.output_schema = output_schema
        self.output_indices = [i for i in range(len(output_schema.split(",")))]

        self.counter = Counter()

    def close(self):
        tf.logging.info('close table writer')
        self.table_writer.close()

    def process(self, features):
        def str_format(element):
            if isinstance(element, float) or isinstance(element, int) \
                    or isinstance(element, str):
                return str(element)
            if element == []:
                return ''
            if isinstance(element, list) and not isinstance(element[0], list):
                return ','.join([str(t) for t in element])
            elif isinstance(element[0], list):
                return ';'.join([','.join([str(t) for t in item]) for item in element])
            else:
                raise RuntimeError("type {} not support".format(type(element)))

        ziped_list = []
        for idx, feat_name in enumerate(self.output_schema.split(",")):
            batch_feat_value = features[feat_name]
            curr_list = []
            for feat in batch_feat_value:
                if len(batch_feat_value.shape) == 1:
                    curr_list.append([feat])
                else:
                    curr_list.append(feat.tolist())
            ziped_list.append(list(map(lambda x: str_format(x), curr_list)))

        self.table_writer.write(list(zip(*ziped_list)), self.output_indices)

        self.counter.count()
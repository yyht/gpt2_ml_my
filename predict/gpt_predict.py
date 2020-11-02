
from predict import AppPredictor
from predict.preprocessors import GPTPreprocessor
from predict.postprocessors import GPTPostprocessor
import tensorflow as tf
from train.modeling import GroverModel, GroverConfig, sample
from bunch import Bunch
from predict.app_utils import get_selected_columns_schema

_app_flags = tf.app.flags
_app_flags.DEFINE_string("inputTable", default=None, help='Input table (only for pai cmd)')
_app_flags.DEFINE_string("outputTable", default=None, help='Output table (only for pai cmd)')
_app_flags.DEFINE_string("inputSchema", default=None,
                          help='Only for csv data, the schema of input table')
_app_flags.DEFINE_string("firstSequence", default=None,
                          help='Which column is the first sequence mapping to')
_app_flags.DEFINE_string("secondSequence", default=None,
                          help='Which column is the second sequence mapping to')
_app_flags.DEFINE_string("appendCols", default=None,
                          help='Which columns will be appended on the outputs')
_app_flags.DEFINE_string("outputSchema", default="pool_output,first_token_output,all_hidden_outputs",
                          help='The choices of output features')
_app_flags.DEFINE_integer("sequenceLength", default=128,
                          help='Maximum overall sequence length.')
_app_flags.DEFINE_string("modelName", default='',
                          help='Name of pretrained model')
_app_flags.DEFINE_integer("batchSize", default=32,
                          help='Maximum overall sequence length.')
_APP_FLAGS = _app_flags.FLAGS

class GPTConfig(object):
    def __init__(self):
        """ Configuration adapter for `ez_bert_feat`
            It adapts user command args to configuration protocol of `ez_transfer` engine
        """
        input_table = FLAGS.tables
        output_table = FLAGS.outputs

        all_input_col_names = get_all_columns_name(input_table)
        
        first_sequence = _APP_FLAGS.firstSequence
        assert first_sequence in all_input_col_names, "The first sequence should be in input schema"
        second_sequence = _APP_FLAGS.secondSequence
        if second_sequence not in all_input_col_names:
            second_sequence = ""
        append_columns = [t for t in _APP_FLAGS.appendCols.split(",") if t and t in all_input_col_names] \
                          if _APP_FLAGS.appendCols else []
        tf.logging.info(input_table)
        
        selected_cols_set = [first_sequence]
        if second_sequence:
            selected_cols_set.append(second_sequence)
        selected_cols_set.extend(append_columns)
        selected_cols_set = set(selected_cols_set)
        input_schema = get_selected_columns_schema(input_table, selected_cols_set)
        
        output_schema = _APP_FLAGS.outputSchema
        for column_name in append_columns:
            output_schema += "," + column_name

        config_json = {
            "preprocess_config": {
                "input_schema": input_schema,
                "output_schema": output_schema,
                "first_sequence": first_sequence,
                "second_sequence": second_sequence,
                'sequence_length': _APP_FLAGS.sequenceLength,
            },
            "model_config": {
                "my_vocab_path": "oss://alg-misc/BERT/bert_pretrain/open_domain/gpt/mega_clue_vocab/clue-vocab.txt",
            },
            "predict_config": {
                "predict_input_fp": None,
                "predict_batch_size": 1,
                "predict_output_fp": None
            }
        }
        config_json["worker_hosts"] = FLAGS.worker_hosts
        config_json["task_index"] = FLAGS.task_index
        config_json["job_name"] = FLAGS.job_name
        config_json["num_gpus"] = FLAGS.workerGPU
        config_json["num_workers"] = FLAGS.workerCount

        self.worker_hosts = str(config_json["worker_hosts"])
        self.task_index = int(config_json["task_index"])
        self.job_name = str(config_json["job_name"])
        self.num_gpus = int(config_json["num_gpus"])
        self.num_workers = int(config_json["num_workers"])

        self.input_schema = config_json['preprocess_config']['input_schema']
        self.label_name = config_json['preprocess_config'].get('label_name', None)
        self.label_enumerate_values = config_json['preprocess_config'].get('label_enumerate_values', None)
        self.output_schema = config_json['preprocess_config'].get('output_schema', None)
        self.sequence_length = config_json['preprocess_config']['sequence_length']
        self.first_sequence = config_json['preprocess_config']['first_sequence']
        self.second_sequence = config_json['preprocess_config']['second_sequence']

        self.vocab_file_path = config_json['model_config']['my_vocab_path']

        self.predict_input_fp = config_json['predict_config']['predict_input_fp']
        self.predict_output_fp = config_json['predict_config'].get('predict_output_fp', None)
        self.predict_batch_size = config_json['predict_config']['predict_batch_size']

        self.news_config = GroverConfig.from_json_file('oss://alg-misc/BERT/bert_pretrain/open_domain/gpt/mega_clue_vocab/mega.json')
        self.ckpt_fn = "oss://alg-misc/BERT/bert_pretrain/open_domain/gpt/mega_clue_vocab/model.ckpt-220000"

app_config = GPTConfig()

predictor = AppPredictor(app_config, 
                 thread_num=1, 
                 queue_size=256,
                 job_name="app_predictor")

preprocessor = GPTPreprocessor(app_config,
                              thread_num=predictor.thread_num,
                              input_queue=queue.Queue(),
                              output_queue=queue.Queue())
postprocessor = GPTPostprocessor(app_config,
                                prediction_colname="predictions",
                                thread_num=predictor.thread_num,
                                input_queue=queue.Queue(),
                                output_queue=queue.Queue())

predictor.run_predict(reader=None,
                      preprocessor=preprocessor,
                      postprocessor=postprocessor,
                      writer=None)

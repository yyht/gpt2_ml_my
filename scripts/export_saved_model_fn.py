import sys
import os
import argparse
import json
import re
import time
import tensorflow.compat.v1 as tf
import numpy as np

from train.modeling import GroverModel, GroverConfig, sample
from tokenization import tokenization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.DEBUG)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

parser = argparse.ArgumentParser(description='Contextual generation (aka given some metadata we will generate articles')
parser.add_argument(
    '-metadata_fn',
    dest='metadata_fn',
    type=str,
    help='Path to a JSONL containing metadata',
)
parser.add_argument(
    '-input_path',
    dest='input_path',
    type=str,
    help='Out jsonl, which will contain the completed jsons',
)
parser.add_argument(
    '-output_path',
    dest='output_path',
    type=str,
    help='Out jsonl, which will contain the completed jsons',
)
parser.add_argument(
    '-out_fn',
    dest='out_fn',
    type=str,
    help='Out jsonl, which will contain the completed jsons',
)
parser.add_argument(
    '-input',
    dest='input',
    type=str,
    help='Text to complete',
)
parser.add_argument(
    '-config_fn',
    dest='config_fn',
    default='configs/mega.json',
    type=str,
    help='Configuration JSON for the model',
)
parser.add_argument(
    '-ckpt_fn',
    dest='ckpt_fn',
    default='../models/mega/model.ckpt',
    type=str,
    help='checkpoint file for the model',
)
parser.add_argument(
    '-export_dir',
    dest='export_dir',
    type=str,
    help='export_dir checkpoint file for the model',
)
parser.add_argument(
    '-target',
    dest='target',
    default='article',
    type=str,
    help='What to generate for each item in metadata_fn. can be article (body), title, etc.',
)
parser.add_argument(
    '-batch_size',
    dest='batch_size',
    default=1,
    type=int,
    help='How many things to generate per context. will split into chunks if need be',
)
parser.add_argument(
    '-num_folds',
    dest='num_folds',
    default=1,
    type=int,
    help='Number of folds. useful if we want to split up a big file into multiple jobs.',
)
parser.add_argument(
    '-fold',
    dest='fold',
    default=0,
    type=int,
    help='which fold we are on. useful if we want to split up a big file into multiple jobs.'
)
parser.add_argument(
    '-max_batch_size',
    dest='max_batch_size',
    default=None,
    type=int,
    help='max batch size. You can leave this out and we will infer one based on the number of hidden layers',
)
parser.add_argument(
    '-top_p',
    dest='top_p',
    default=0.95,
    type=float,
    help='p to use for top p sampling. if this isn\'t none, use this for everthing'
)
parser.add_argument(
    '-min_len',
    dest='min_len',
    default=1024,
    type=int,
    help='min length of sample',
)
parser.add_argument(
    '-eos_token',
    dest='eos_token',
    default=102,
    type=int,
    help='eos token id',
)
parser.add_argument(
    '-samples',
    dest='samples',
    default=5,
    type=int,
    help='num_samples',
)

args = parser.parse_args()
proj_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
vocab_file_path = os.path.join(proj_root_path, "tokenization/clue-vocab.txt")
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path , do_lower_case=True)
news_config = GroverConfig.from_json_file(args.config_fn)

# We might have to split the batch into multiple chunks if the batch size is too large
default_mbs = {12: 32, 24: 16, 48: 3}
max_batch_size = args.max_batch_size if args.max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

# factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
num_chunks = int(np.ceil(args.batch_size / max_batch_size))
batch_size_per_chunk = int(np.ceil(args.batch_size / num_chunks))

# This controls the top p for each generation.
top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * args.top_p

graph = tf.Graph()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3,
                                    allow_growth=False)
session_conf = tf.ConfigProto(
              intra_op_parallelism_threads=8,
              inter_op_parallelism_threads=8,
              allow_soft_placement=True,
              gpu_options=gpu_options)

receiver_features = {
    "initial_context":tf.placeholder(tf.int32, [1, None], name='initial_context'),
    "p_for_topp":tf.placeholder(tf.int32, [1], name='p_for_topp'),
    "eos_token":tf.placeholder(tf.int32, [], name='eos_token'),
    "min_len":tf.placeholder(tf.int32, [], name='min_len'),
    "max_len":tf.placeholder(tf.int32, [], name='max_len'),
    "k_for_topk":tf.placeholder(tf.int32, [], name='k_for_topk')
}

from train.modeling import export_model_fn_builder

def serving_input_receiver_fn():
    print(receiver_features, "==input receiver_features==")
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(receiver_features)()
    return input_fn

model_fn = export_model_fn_builder(news_config, args.ckpt_fn)

estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=args.ckpt_fn)

export_dir = estimator.export_savedmodel(args.export_dir, 
                                serving_input_receiver_fn,
                                checkpoint_path=args.ckpt_fn)
print("===Succeeded in exporting saved model==={}".format(args.export_dir))

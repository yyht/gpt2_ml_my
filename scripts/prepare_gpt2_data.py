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


##### ignore tf deprecated warning temporarily
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

import re
from urlextract import URLExtract
url_extract_api = URLExtract()

def clean(text):
    text = re.sub("""(<[=0-9a-zA-Z\/&"":_\\.]+>;?)+""", "", text)
    text = re.sub("""((&|#|$)+[0-9a-zA-Z]+;?)+""", "", text)
    text = re.sub("""[★☆\u3000]+""", "", text)
    try:
        urls = url_extract_api.find_urls(text)
        for url in urls:
            text = text.replace(url, "")
        return text
    except:
        return text

args = parser.parse_args()
proj_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
vocab_file_path = os.path.join(proj_root_path, "tokenization/bert-base-chinese-vocab.txt")
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path , do_lower_case=True)

def get_file_path(root_path, file_list, dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)

if os.path.isdir(args.input_path):
    file_list, dir_list = [], []
    get_file_path(args.input_path, file_list, dir_list)
    print("==total file==", len(file_list))
else:
    file_list = [args.input_path]

def process(document):
    init_len = 0
    index = 0
    document = "".join(document)
    sentences = re.split(r"([。!！?？；;])", document)
    document = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]

    context = "".join(document)
    clean_original, fake_samples, fake_probs, bert_tokens = generate_text(context, 0.8)


for input_file in file_list:
    document_len = 0
    with tf.gfile.GFile(input_file, "r") as reader:
        while True:
            line = reader.readline()
            
            if not line:
                process(all_documents[-1])
                document_len = 0
                break
            line = line.strip()
            line = "".join(line.split(" "))

            # Empty lines are used as document delimiters
            if not line or len(line) < 1:
                # all_documents.append([])
                process(all_documents[-1])
                all_documents.append([])
                document_len = 0
                continue
            if len(line) + document_len < 508:
                document_len += len(line)
                all_documents[-1].append(line)
            else:
                process(all_documents[-1])
                all_documents.append([])
                document_len = 0
                continue
    
fwobj.close()
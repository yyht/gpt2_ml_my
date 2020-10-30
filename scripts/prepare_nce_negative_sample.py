import sys
import os
import argparse
import json
import re

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
#####

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

def extract_generated_target(output_tokens, tokenizer):
	"""
	Given some tokens that were generated, extract the target
	:param output_tokens: [num_tokens] thing that was generated
	:param encoder: how they were encoded
	:param target: the piece of metadata we wanted to generate!
	:return:
	"""
	# Filter out first instance of start token
	assert output_tokens.ndim == 1

	start_ind = 0
	end_ind = output_tokens.shape[0]

	return {
		'extraction': tokenization.printable_text(''.join(tokenizer.convert_ids_to_tokens(output_tokens))),
		'start_ind': start_ind,
		'end_ind': end_ind,
	}

import re
def clean(text):
	text = re.sub("(<p>)+", "", text)
	text = re.sub("(</p>)+", "", text)
	text = re.sub("""(<imgsrc="23857019472"/>)+""", "", text)
	text = re.sub("(&nbsp)+", "", text)
	return text

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

tf_config = tf.ConfigProto(allow_soft_placement=True)
graph = tf.Graph()
with graph.as_default():
	sess = tf.Session(config=tf_config)
	initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
	p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
	eos_token = tf.placeholder(tf.int32, [])
	min_len = tf.placeholder(tf.int32, [])
	max_len = tf.placeholder(tf.int32, [])

	tokens, probs = sample(news_config=news_config, initial_context=initial_context,
							   eos_token=eos_token, min_len=min_len, 
							   max_len=max_len,
							   ignore_ids=None, p_for_topp=p_for_topp,
							   do_topk=True)
	saver = tf.train.Saver()
	saver.restore(sess, args.ckpt_fn)
	print(u'🍺Model loaded. \nInput something please:⬇️')

def generate_text(text, ratio=0.8):

	output_lst = []
	prob_lst = []
	with graph.as_default():

		line = tokenization.convert_to_unicode(text)
		line = clean(line)
		print(line)
		bert_tokens = tokenizer.tokenize(line)
		encoded = tokenizer.convert_tokens_to_ids(bert_tokens)
		encoded_prefix = encoded[0:int(len(encoded)*ratio)]
		print("=encoded length== ", len(encoded), '==context length==', len(encoded_prefix))
		context_formatted = []
		context_formatted.extend(encoded_prefix)

		for i in range(args.samples):
			print("Sample,", i + 1, " of ", args.samples)
			# Format context end
			gens = []
			gens_raw = []
			gen_probs = []
			for chunk_i in range(num_chunks):
				tokens_out, probs_out = sess.run([tokens, probs],
												 feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
															eos_token: args.eos_token, 
															min_len: args.min_len,
															max_len: len(encoded),
															p_for_topp: top_p[chunk_i]})

				for t_i, p_i in zip(tokens_out, probs_out):
					extraction = extract_generated_target(output_tokens=t_i, tokenizer=tokenizer)
					gens.append(extraction['extraction'])
					gen_probs.append(p_i)

			l = re.findall('.{1,70}', gens[0].replace('[UNK]', '').replace('##', ''))
			output_lst.append(l)
			prob_lst.append(gen_probs)
	return line, output_lst, prob_lst, bert_tokens

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

all_documents = [[]]
fwobj = tf.gfile.GFile(args.output_path+"_with_nce_output.txt", "w")

def process(document):
	init_len = 0
	index = 0
	document = "".join(document)
	sentences = re.split(r"([。!！?？；;])", document)
	document = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]

	context = "".join(document)

	clean_original, fake_samples, fake_probs, bert_tokens = generate_text(context, 0.8)

	for fake_sample, prob in zip(fake_samples, fake_probs):
		output_dict = {
			"clean_original":"".join(clean_original),
			"gpt_generated":fake_sample,
			"probs":(prob[0]).tolist(),
			"ppl":np.exp(np.log(prob[0]+1e-10).mean()).tolist()
		}
		fwobj.write(json.dumps(output_dict, ensure_ascii=False)+"\n")

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





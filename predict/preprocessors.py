from predict.distribution import Process
from tokenization import tokenization

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

def convert_to_single_example(tokenizer, text, max_seq_length):
    text = clean(text)
    tokens = tokenizer.tokenize(text)
    token_length = len(tokens) 
    initial_context = tokenizer.convert_tokens_to_ids(tokens[0:int(token_length*0.8)])
    p_for_topp = [0.95]
    eos_token = 102
    min_len = int(len(tokens)*0.95)
    max_len = len(tokens)
    k_for_topk = 1000

    return [initial_context, p_for_topp, eos_token, 
            min_len, max_len, k_for_topk]

class GPTPreprocessor(Process):
    def __init__(self,
                 config,
                 thread_num=None,
                 input_queue=None,
                 output_queue=None,
                 job_name='GPTPreprocessor'):
        super(MyPreprocessor, self).__init__(job_name, thread_num, input_queue, output_queue)
        self.first_sequence = config.first_sequence
        self.sequence_length = config.sequence_length
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config.vocab_file_path , do_lower_case=True)

    def process(self, inputs):
        seq_text = tokenization.convert_to_unicode(inputs[self.first_sequence]) # Corresponding to ODPS Table column
        [input_ids, p_for_topp, eos_token, 
            min_len, max_len, k_for_topk] = convert_to_single_example(
            tokenizer=self.tokenizer, text=seq_text, max_seq_length=self.sequence_length)
        ret = {key: np.array([val]) for key, val in inputs.items()}
        ret["initial_context"] = np.array([input_ids]) # Shape of [1, seq_len]
        ret["p_for_topp"] = np.array([p_for_topp])
        ret["eos_token"] = np.array(eos_token)
        ret["min_len"] = np.array(min_len)
        ret["max_len"] = np.array(max_len)
        ret["k_for_topk"] = np.array(k_for_topk)
        return ret
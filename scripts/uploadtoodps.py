import re
import tensorflow as tf
from urlextract import URLExtract
import os, sys
url_extract_api = URLExtract()

fwobj = tf.gfile.GFile("/data/albert/my_chinese_pretrain.txt", "w")

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

def process(document):
    init_len = 0
    index = 0
    document = "".join(document)
    sentences = re.split(r"([。!！?？；;])", document)
    document = ["".join(i) for i in zip(sentences[0::2],sentences[1::2])]

    context = "".join(document)
    if len(context) >= 5:
        fwobj.write(context+"\n")
        
def get_file_path(root_path, file_list, dir_list):
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        dir_file_path = os.path.join(root_path, dir_file)
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            get_file_path(dir_file_path, file_list, dir_list)
        else:
            file_list.append(dir_file_path)

file_list, dir_list = [], []
get_file_path("/data/albert/corpus", file_list, dir_list)
print("==total file==", len(file_list))

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
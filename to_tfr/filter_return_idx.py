"""
This will tokenize all entries in jsonl.zst files, check the token length, 
and return the label number of those that are less than the given number (max_token).

It will generate a text file under the path specified by `output` (is a directory).
"""



from transformers import AutoTokenizer
import tensorflow as tf
import lm_dataformat as lmd
from absl import flags
from absl import app
import os
from pathlib import Path
from random import random
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import pickle


FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'The input folder for jsonl.zst.')
flags.DEFINE_string('output', None, 'The output directory.')
flags.DEFINE_string('tokenizer_path', None, 'tokenizer path.')
flags.DEFINE_integer('threads', 1, 'number of threads.')
flags.DEFINE_integer('max_token', 2048, 'throw away samples with more token numbers.')
flags.DEFINE_string('name', 'data', 'file prefix.')



def tokenize(i, lst, tokenizer, offset):
    result = tokenizer(lst, return_tensors='np', max_length=int(1e8), truncation=True)
    is_short = np.vectorize(lambda token_list: len(token_list) < 2048)

    keep_idx = is_short(result['input_ids'])

    print(f"Chunk {i} finished.")
    return (np.where(keep_idx)[0] + offset).tolist()


def filter_token_len(docs):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_path)
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    NUM_PROCESS = FLAGS.threads
    LEN = len(docs) // NUM_PROCESS

    parallel_args = [(i, docs[i*LEN:(i+1)*LEN], tokenizer, i*LEN) for i in range(NUM_PROCESS)]
    with mp.Pool(processes=NUM_PROCESS) as pool:
        result = list(pool.starmap(tokenize, parallel_args))

    print("All processes finished.")
    
    file_path = FLAGS.output + f"/{FLAGS.name}.txt"
    print(f"Writing to text file at {file_path}")

    with open(file_path, "w") as f:
        f.write(str(result))


def get_files(input_path):
    supported_file_types = ['jsonl.zst']
    
    if isinstance(input_path, str):
        input_path = Path(input_path)
    
    if input_path.is_dir():
        # get all files with supported file types
        files = [list(Path(input_path).glob(f"*{ft}")) for ft in supported_file_types]
        # flatten list
        files = [f for sublist in files for f in sublist]
        assert files, f"No files with supported types found in directory: {input_path}"
    elif input_path.is_file():
        assert any(
            str(input_path).endswith(f_type) for f_type in supported_file_types
        ), f"Input file type must be one of: {supported_file_types}"
        files = [input_path]
    else:
        raise FileNotFoundError(f"No such file or directory: {input_path}")

    return [str(f) for f in files]


def read_from_file(input_path):
    reader = lmd.Reader(input_path)
    lst = list(reader.stream_data())
    print(f"Reading {input_path} completed.")
    return lst


def main(argv):
    if FLAGS.output:
        os.makedirs(FLAGS.output, exist_ok=True)
    
    # Read the files in the folder
    files = get_files(FLAGS.input)
    with mp.Pool(processes=FLAGS.threads) as pool:
        results = pool.map(read_from_file, files)
    
    # Flatten the list
    docs = []
    for doc in results:
        docs.extend(doc)
    print(f"Reading finished. Tokenizing.")

    filter_token_len(docs)

if __name__=="__main__":
    app.run(main)


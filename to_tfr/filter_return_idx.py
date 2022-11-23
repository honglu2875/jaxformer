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

    keep_idx = is_short(result[i]['input_ids'])

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



def main(argv):
    if FLAGS.output:
        os.makedirs(FLAGS.output, exist_ok=True)
    
    # Read the folder as a whole
    reader = lmd.Reader(FLAGS.input)
    # Make data a contiguous list
    docs = list(reader.stream_data())


    
    print(f"Tokenizing.")
    filter_token_len(docs)

if __name__=="__main__":
    app.run(main)


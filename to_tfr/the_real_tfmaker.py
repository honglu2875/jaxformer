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


FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'The input folder for jsonl.zst.')
flags.DEFINE_string('output', None, 'The output directory.')
flags.DEFINE_string('format', 'tfrecord', '[tfrecord]')
flags.DEFINE_string('tokenizer_path', None, 'tokenizer path.')
flags.DEFINE_integer('threads', 1, 'number of threads.')
flags.DEFINE_integer('max_token', 2048, 'throw away samples with more token numbers.')
flags.DEFINE_string('name', 'data', 'file prefix.')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_to_file(writer, data):
    feature = {
            "text": _int64_feature(data)
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(tf_example.SerializeToString())



def write_tfrecord(sequences, file_path):
    with tf.io.TFRecordWriter(file_path) as writer:
        for seq in sequences:
            write_to_file(writer, seq)


def create_tfrecords(docs):
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_path)
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    NUM_PROCESS = FLAGS.threads
    LEN = len(docs) // NUM_PROCESS

    def tokenize(id, lst, start_idx, length, result):
        result[id] = tokenizer(lst[start_idx:start_idx + length], return_tensors='np', max_length=int(1e8), truncation=True)


    with mp.Manager() as manager:
        result = manager.dict()
        processes = []
        for i in range(NUM_PROCESS):
            processes.append(mp.Process(target=tokenize, args=(i, docs, i*LEN, LEN, result)))
            processes[-1].start()
                                                        
        for i in range(NUM_PROCESS):
            processes[i].join()
        result = dict(result)


    # Filtering is faster outside the multiprocessing
    all_seq = []
    for i in range(NUM_PROCESS): 
        keep_idx = np.vectorize(lambda token_list: len(token_list) < 2048)(result[i]['input_ids'])
        all_seq.extend(result[i]['input_ids'][keep_idx].tolist()) 
    
    total_len = len(all_seq)
    print(f"Total length of filtered data: {total_len}")

    write_tfrecord(all_seq, FLAGS.output + f"/{FLAGS.name}_{total_len}.tfrecords")



def main(argv):
    if FLAGS.output:
        os.makedirs(FLAGS.output, exist_ok=True)
    
    # Read the folder as a whole
    reader = lmd.Reader(FLAGS.input)
    # Make data a contiguous list
    docs = list(reader.stream_data())


    
    if FLAGS.format=='tfrecord':
        print(f"Creating TFRecords.")
        results = create_tfrecords(docs)
    else:
        raise ValueError('unsupported format.')

if __name__=="__main__":
    app.run(main)


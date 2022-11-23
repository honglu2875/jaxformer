"""
This script will be applied to three dirs: the input dir, the index dir and the output dir.
It will search through the index dir for the *SAME FILENAME* as the files in the input dir. If found, it will parse the index file into a list,
  and use them as indices to pick samples. In the input dir, the file opened is assumed to be jsonl.zst format. The filtered samples will be put
  in the output dir.
It supports multiprecessing (better if you let the number of threads to be less than cpu count).
"""

import lm_dataformat as lmd
from absl import flags, app
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import ast
import os


FLAGS = flags.FLAGS
flags.DEFINE_string('input', None, 'The input dir.')
flags.DEFINE_string('index_dir', None, 'The index dir.')
flags.DEFINE_string('output', None, 'The output dir.')
flags.DEFINE_integer('threads', 1, 'The number of threads.')
flags.DEFINE_string('target_format', 'lm', 'The target format (lm, json)')


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

def unravel(lst):
	if isinstance(lst, list):
		result = []
		for elem in lst:
			result.extend(unravel(elem))
		return result
	elif isinstance(lst, int):
		return [lst]
	else:
		return []

def apply_filter(file_path):
    file_name = file_path.split("/")[-1]
    index_path = FLAGS.index_dir + f"/{file_name}"
    output_path = FLAGS.output + f"/{file_name}"
    reader = lmd.Reader(file_path)

    if not Path(index_path).exists():
        print(f"{index_path} does not exist. Abort.")
        return

    with open(index_path, 'r') as f:
        indices = ast.literal_eval(f.read())

    indices = unravel(indices)
    print(f"{index_path}: {len(indices)} unravelled indices")

    ind_dict = dict()
    for ind in indices:
        ind_dict[ind] = True

    result = []
    for i, doc in enumerate(reader.stream_data()):
        if i in ind_dict:
            result.append(doc)

    if FLAGS.target_format == "lm":
        ar = lmd.Archive(output_path)
        for doc in result:
            ar.add_data(doc, meta={})
        ar.commit()
    elif FLAGS.target_format == "json":
        from datasets import Dataset
        dataset = Dataset.from_list([{'text': s} for s in result])
        dataset.to_json(output_path)

    print(f"New data file written in {output_path}")



def main(argv):
    if FLAGS.output:
        os.makedirs(FLAGS.output, exist_ok=True)

    # Read the files in the folder
    files = get_files(FLAGS.input)

    print(files)
    
    with mp.Pool(processes=FLAGS.threads) as pool:
        pool.map(apply_filter, files)

if __name__=='__main__':
    app.run(main)

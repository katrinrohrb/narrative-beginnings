import json
import stanza 
import pathlib
from pathlib import Path
import os 
import pandas as pd
import numpy as np

def results_from_json(filenames, results_dir):

  if isinstance(filenames, str):
    filename = filenames
    json_file_path = Path(results_dir, Path(filename).stem + '-result.json')

    if json_file_path.exists():
        with open(json_file_path, 'r', encoding="utf-8") as f:
            json_results = json.load(f)
        return json_results
    else:
        return None
  elif isinstance(filenames, list):
    return {filename : results_from_json(filename, results_dir) for filename in filenames}
  

def chunker_alt(seq, size):
    return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

def chunk_lengths(total_sents, num_chunks):
  base_chunk_size, remainder = divmod(total_sents, num_chunks)
  return [base_chunk_size + 1] * remainder + [base_chunk_size] * (num_chunks - remainder)

def chunker(seq, size = None, num_chunks = None, start_at = None, stop_before = None, ind = None, return_ind = False):
  if num_chunks == None and size == None:
    raise RuntimeError("One of num_chunks and size must be given!")

  if (num_chunks and (len(seq) < num_chunks)) or (size and (len(seq) < size)):
     raise("sequence is too short to be chunked in this way")
  #make chunks of fixed size if possible 
  if (size != None) and (num_chunks == None):
    all_chunks = [seq[pos:pos + size] for pos in range(0, len(seq), size)]
  
  #guarantee certain number of chunks
  elif (size == None) and (num_chunks != None):
    lengths = chunk_lengths(len(seq), num_chunks)
    sums = [sum(lengths[0:i]) for i in range(0, num_chunks + 1)]
    chunk_borders = zip(sums[:-1], sums[1:])
    all_chunks = [seq[start:end] for start, end in chunk_borders]
  else:
    raise RuntimeError("chunker should not be called with both size and num_chunks args")
  
  # If only part requested...
  if start_at == None:
    start_at = 0
  if stop_before == None:
    stop_before = len(all_chunks)
  if ind == None:
     ind = range(start_at, stop_before)
  else:
     ind = ind[start_at:stop_before]
     
  chunks = [all_chunks[i] for i in ind if i < len(all_chunks)]  
  #Finally, return
  if return_ind:
    return chunks, ind
  else:
    return chunks
  
def non_empty_labels_from_iter(series):
  while True:
    next_element = next(series, "")
    if next_element == "":
      break
    else:
      yield next_element
  


class Results(): 
    
    @classmethod
    def convert_json_to_parquet(cls, json_fns, json_dir, labels_pqt_fn, scores_pqt_fn, drive_dir):
      #from .core import drive_dir
      if type(json_dir) != list: 
        results = results_from_json(json_fns, json_dir)
      else:
        json_fns_list = json_fns
        json_dir_list = json_dir 
        results = {}
        for json_fns, json_dir in zip(json_fns_list, json_dir_list):
           results.update(results_from_json(json_fns, json_dir))

      max_len = max([len(results[filename]) for filename in results.keys() if results[filename] != None])
       
      labels_padded = [pd.Series(data = \
      [result["label"] for result in results[filename]] + [""] * (max_len - len(results[filename])), \
                                name = filename) for filename in results.keys() if results[filename] != None]

      labels_df = pd.DataFrame(labels_padded)
      labels_df = labels_df.T

      fp_labels_df = Path(drive_dir, labels_pqt_fn)
      labels_df.to_parquet(fp_labels_df)


      scores_padded = [pd.Series(data = \
      [result["score"] for result in results[filename]] + [np.NaN] * (max_len - len(results[filename])), \
                                name = filename) for filename in results.keys() if results[filename] != None]

      scores_df = pd.DataFrame(scores_padded)

      scores_df = scores_df.T

      fp_scores_df = Path(drive_dir, scores_pqt_fn)
      scores_df.to_parquet(fp_scores_df)
    
    def __init__(self):
       pass
    
    @classmethod
    def convert_json_to_parquet_ext_canon(cls, drive_dir):
       import json

       data_dir = Path(drive_dir, "Datasets/")
       project_dir =  Path(drive_dir, "Colab Notebooks/katspace-project/")

       with open(Path(project_dir, "old_canon_fns.json")) as user_file:
          file_contents = user_file.read()
          old_fns = json.loads(file_contents)

       canon_df = pd.read_excel(Path(data_dir, "20240818_canon_master_merged.xlsx", index = 0))
       new_fns = [filename for filename in canon_df["filename"] if filename not in old_fns]
       json_fns_list = [old_fns, new_fns]
       json_dir_list = [Path(drive_dir, "results/predict-286-gutenberg"), Path(drive_dir, "results/predict-286-canon") ]
       
       labels_pqt_fn = "results/2025_02_05_labels_df_canon.parquet"
       scores_pqt_fn = "results/2025_02_05_scores_df_canon.parquet"

       Results.convert_json_to_parquet(json_fns_list, json_dir_list, labels_pqt_fn, scores_pqt_fn, drive_dir)
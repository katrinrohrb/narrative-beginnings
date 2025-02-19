import matplotlib.pyplot as plt
from katspace.data import chunk_lengths, chunker
from katspace.core import space_types_pos, space_types_ext, space_types, space_types_pos_ext
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns

#this function exists two times, once in the notebooks (sb_data_vis_canon, compare_canon_non_compare)
def calculate_ratios(df, insert_values = False, set_index = True, suffix = "_rt"): 
    
    if insert_values == False:
        _df = df.copy()
    else: _df = df

    _df['all_space']=_df[space_types_pos].sum(axis=1)

    for space_type in space_types_ext:
        _df[space_type + suffix] = _df[space_type]/_df["total"]

    if insert_values == False:
        _df = _df[["year"] + [space_type + suffix for space_type in space_types_ext]]
    
    if set_index:
        _df.set_index("year", inplace = True)
    
    return _df

def calculate_ratios2(df, insert_values=False, set_index=True, suffix="_rt"): 
    if not insert_values:
        _df = df.copy()
    else:
        _df = df

    _df['all_space'] = _df[space_types_pos].sum(axis=1)

    for space_type in space_types_ext:
        if space_type in df:
            _df[space_type + suffix] = _df[space_type] / _df["total"]


    if not insert_values:
        _df = _df[["year", "author_last", "title"] + [space_type + suffix for space_type in space_types_ext]]
    
    if set_index:
        _df.set_index("year", inplace=True)
    
    return _df


def smooth_df(df, half_window_size = 5, set_index = True, trim_window = True):

    df = df.reset_index()
    min_year, max_year = df["year"].min(), df["year"].max()

    #df_list = [(df.copy(), i) for i in range(-half_window_size+1, half_window_size)]
    df_list = []
    for i in range(-half_window_size+1, half_window_size): 
        _df = df.copy()
        _df["year"] = _df["year"].apply(lambda x : x + 1) 
        df_list.append(_df)

    for i, df in enumerate(df_list):
        df_list[i]["year"] =  df_list[i]["year"] + i

    smooth_df = pd.concat(df_list)
    if trim_window == True: 
        smooth_df = smooth_df[smooth_df["year"].isin(range(min_year + 2 * half_window_size, max_year))]
    else: 
        smooth_df = smooth_df[smooth_df["year"].isin(range(min_year, max_year))]
    if set_index: 
        smooth_df.set_index("year", inplace = True)

    return smooth_df


def hist_heatmap(df, chunk_size_target = 5, vert_num_chunks = 20, space_types = ["action_space", "perceived_space", "visual_space", "descriptive_space", "all_space", "no_space"]):
    cols = list(space_types) + ["total", "year"]
    df = df[cols].set_index("year")
    df.loc[:,'books'] = 1 

    years = df.index.sort_values().unique()

    chunks = chunker(years, size = chunk_size_target)
    chunk_sizes = list(map(len, chunks))
    num_chunks = len(chunk_sizes)

    def grouper_f(idx):
        for c, chunk in enumerate(chunks):
            if idx in chunk: 
                break
        return c 

    df["chunk"] = df.index.map(grouper_f)
    num_books = df[["books", "chunk"]].groupby("chunk").sum()
    num_books = num_books["books"].values

    color = {"perceived_space": "Blues",
                    "action_space": "Greens",
                    "visual_space": "Reds",
                    "descriptive_space": "GnBu",
                    "no_space": "Greys",
                    "all_space": 'Purples'
                    }
    fig, axs = plt.subplots(ncols=1, nrows=len(space_types), figsize=(3, len(space_types) * 3),
                    layout="constrained") 

    for i, space_type in enumerate(space_types):
        title = space_type 
        y = []
        x = []

        for idx, book in df.iterrows():
            this_chunk = book["chunk"]
            nr_books_in_chunk = num_books[this_chunk]
            if (book.total == 0):
                continue

            norm_count = book[space_type] / book["total"]

            y += [norm_count]
            year = chunks[this_chunk][0]
            x += [year]
        
        axs[i].set_title(title)
        axs[i].hist2d(x, y, bins = [num_chunks, vert_num_chunks], cmap = color[space_type])

def plot_p_values_heatmap(res, genres, space_type):

    values = np.round(res[space_type].pvalue, 3)

    fig, ax = plt.subplots(figsize = (4,6))
    im = ax.imshow(values)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(genres)), labels=genres)
    ax.set_yticks(np.arange(len(genres)), labels=genres)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(genres)):
        for j in range(len(genres)):
            if i >= j: 
                text = ax.text(j, i, values[i, j],
                            ha="center", va="center", color="w")
            else: 
                value = res[space_type].statistic[j,i]
                value = np.round(100 * value, 1)
                ax.text(j, i, value, 
                            ha="center", va="center", color="w")

    ax.set_title(space_type)
    #fig.tight_layout()
    plt.show()

#sum_sel defines all_space!
def plot_preprocess(results, num_chunks=None, chunk_size = None, ind = None, 
                    start_at = None, stop_before = None,
                    sum_sel=space_types_pos, results_format = "parquet", return_max=False):
    #ignore files without results
    results = {filename: result for filename, result  in results.items() if result != None}

    #ignore files with fewer than num_chunks sentences
    if num_chunks != None:
        results = {filename: result for filename, result  in results.items() if len(result) >= num_chunks}
        ignored_files = len([filename for filename, result  in results.items() if len(result) < num_chunks])
    elif chunk_size != None:
        results = {filename: result for filename, result  in results.items() if len(result) >= chunk_size}
        ignored_files = len([filename for filename, result  in results.items() if len(result) < chunk_size])
    else:
        raise RuntimeError("One of num_chunks or chunk_size must be passed!")
    if ignored_files > 0: 
        print(f"plot_preprocess: Ignoring {ignored_files} files because they have too few sents")

    if results_format != "parquet":
      labels = {filename: [result["label"] for result in results[filename]] for filename in results.keys()}
    else:
      labels = results

    chunked_labels = {}
    maxs = []
    for filename in labels.keys():
        chunks = chunker(labels[filename], num_chunks=num_chunks, size = chunk_size, start_at=start_at, stop_before=stop_before)
        chunked_labels[filename] = chunks

    chunk_sizes = {filename: [len(chunk) for chunk in chunked_labels[filename]] for filename in labels.keys()}
    chunk_size_zero = [filename for filename in labels.keys() if chunk_sizes[filename] == 0]
    if len(chunk_size_zero) > 0:
      print(f"The following files have chunks of size 0: \n {chunk_size_zero}")

    counters = {filename: [Counter(label_chunk) for label_chunk in chunked_labels[filename]] for filename in chunked_labels.keys()}

    def total_sum(counter, sum_sel):
        num = [counter[space_type] for space_type in sum_sel]
        return sum(num)

    # change order of indexing to have space type as outermost index
    count_dict = {space_type: {filename: [counter[space_type] for counter in counters[filename]] for filename in counters.keys()} for space_type in space_types}

    count_dict["all_space"] = {}
    for filename in counters.keys():
      count_dict["all_space"][filename] = [total_sum(counter, sum_sel) for counter in counters[filename]]

    return count_dict, chunk_sizes
    

def plot_hist_preprocess(count_dict, chunk_sizes, debug_msg = "", 
                         start_at_one = False, start_at = None):

    y = []
    x = []

    for filename in count_dict.keys():
        for c, size in enumerate(chunk_sizes[filename]):
          if size == 0:
            print(f"chunk nr {c} has size zero: {filename}")
            #zero_chunks.append((filename, c))
        norm_counts_np = np.array(count_dict[filename]) / np.array(chunk_sizes[filename])
        y += list(norm_counts_np)

        
        next_x = range(len(chunk_sizes[filename]))

        if start_at != None and start_at > 0:
            next_x = map(lambda x:x+start_at, next_x)

        if start_at_one:
            next_x = map(lambda x:x+1, next_x)

        x += next_x

        if len(x) != len(y):
          print(f"x and y have different lengths: {filename}, {debug_msg} \n ++++++ x: {x}, \n ++++++ y: {y}")
          break
    return x,y

def beginning_history_preprocess(results, years, num_chunks = None, sum_sel = space_types_pos, return_df = False, chunk_size = None): 
    #ignore files without results
    results = {filename: result for filename, result  in results.items() if (result != None) and (filename in years.keys())}

    #ignore files with fewer than num_chunks sentences
    if num_chunks != None: 
        min_len = num_chunks
    elif chunk_size != None: 
        min_len = chunk_size
    else: 
        raise("One of num_chunks or chunk_size is mandatory!")
    
    results = {filename: result for filename, result  in results.items() if len(result) >= min_len}
    ignored_files = len([filename for filename, result  in results.items() if len(result) < min_len])
    if ignored_files > 0: 
        print(f"plot_preprocess: Ignoring {ignored_files} files because they have too few sents")

    labels = results

    chunked_labels = {filename: chunker(labels[filename], num_chunks=num_chunks, size = chunk_size, stop_before = 1) for filename in labels.keys()}
    chunk_sizes = {filename: [len(chunk) for chunk in chunked_labels[filename]] for filename in labels.keys()}

    chunk_size_zero = [filename for filename in labels.keys() if chunk_sizes[filename] == 0]
    if len(chunk_size_zero) > 0:
      print(f"The following files have chunks of size 0: \n {chunk_size_zero}")

    counters = {filename: [Counter(label_chunk) for label_chunk in chunked_labels[filename]] for filename in chunked_labels.keys()}

    def total_sum(counter, sum_sel):
        num = [counter[space_type] for space_type in sum_sel]
        return sum(num)

    if not return_df: 
        # change indexing 
        count_dict = { 
        space_type: 
            { filename : {
                            "num_sents" : [counter[space_type] for counter in counters[filename]], 
                            "year" : years[filename]
                        }
            for filename in counters.keys()
            } 
        for space_type in space_types
        }

        count_dict["all_space"] = {}
        for filename in counters.keys():
            count_dict["all_space"][filename] = {
                                "num_sents" : [total_sum(counter, sum_sel) for counter in counters[filename]], 
                                "year" : years[filename]
                            }

        return count_dict, chunk_sizes
    
    if return_df: 
        dict = {
            "filename" : [], 
            "year" : [],
            "total" : []
        }
        for space_type in space_types_pos_ext: 
            dict[space_type] = []

        for filename in counters.keys():
            dict["filename"].append(filename)
            dict["year"].append(years[filename])

            for space_type in space_types_pos:
                dict[space_type].append(counters[filename][0][space_type])
            dict["all_space"].append(total_sum(counters[filename][0], sum_sel))
            dict["total"].append(chunk_sizes[filename][0])
            

        return pd.DataFrame(dict)
    

def prep_plot_compare_new(dfs, labels, space_type_d, insert_values = False): 

    _dfs = []

    for space_type in space_type_d:
        plot_configs = space_type_d[space_type]

        for df, label, plot_config in zip(dfs, labels, plot_configs):
            _df = df.copy()
            _df.loc[:, "label"] = label
        
            df_part = _df.reset_index()[["year"] + [space_type] + ["label"]].copy()
            df_part = df_part.rename({space_type : "ratio"}, axis = 1)

            df_part.loc[:,"space_type"] = space_type

            for key in plot_config: 
                df_part.loc[:, key] = plot_config[key]
            
            _dfs.append(df_part)
    
    return pd.concat(_dfs, ignore_index=True)

def plot_compare(dfs, palettes, space_types = None):

    sns.set_context("paper")
    sns.set_style("whitegrid")

    if space_types == None: 
        space_types = dfs["space_type"].unique()

    fig, axs = plt.subplots(ncols=1, nrows=len(space_types), figsize=(8, len(space_types) * 6),
                        layout="constrained")
    
    if len(space_types) == 1: 
        axs = [axs]
        
    for ax, space_type in zip(axs, space_types):
        ax.set_title(space_type)
        data = dfs.query("space_type == @space_type")
        sns.lineplot(data = data, x = "year", y = "ratio", hue = "label", ax = ax, style = "label", 
                        palette = palettes[space_type])
        ax.set_xmargin(0)

def get_years_dict(df):
    temp_df = df.set_index("filename")["year"]
    years = temp_df.to_dict()
    return years

    
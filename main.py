from mpi4py import MPI
import os
from datetime import datetime
from itertools import combinations, islice
import LCS_cython
import csv
from typing import List, Dict

# Default variable initialization for MPI4PY
# If you want to learn more, I recommend checking out:
# https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


# ----------------------------------------------------------------------------
# Below are generic reusable functions supporting the main structure of
# parallelization on the Cambridge HPC.
# ----------------------------------------------------------------------------


def create_result_directory(name: str, includeTimeStamp: bool = True) -> str:
    """
    Create a directory to buffer multiple files storing the results of your
    calculations. This will be executed on each worker, hence the try-except block 
    in the implementation.

    Args:
        name (str): Root name of the directory storing results in the form of multiple files.
        includeTimeStamp (bool, optional): Include timestamp in directory name.
        There might be cases where more than one results directory will be created
        due to delays in starting multiple processes. To prevent overwriting, set
        includeTimeStamp to False. Defaults to True.

    Returns:
        str: Name of the directory.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M") if includeTimeStamp else ""
    directory_name = f"{name}_{timestamp}"

    # Due to the distributed nature of the mpi4py execution in slurm,
    # we wrap the result directory creation in a try-except block.
    try:
        os.makedirs(directory_name)
        print(f"Directory {directory_name} created successfully!")
    except:
        print(f"Directory {directory_name} already exists.")

    return directory_name



def generatorSlice(iterable: combinations, start: int, count: int) -> islice:
    """
    Slice the generator based on the starting index and the number of elements in the iterable.

    Args:
        iterable (combinations): Generator of combinations.
        start (int): Starting index.
        count (int): Number of elements in the iterable.

    Returns:
        islice: Slice of the generator.
    """
    return islice(iterable, start, start + count, 1)


def chunk_combinations(nprocs: int) -> List[islice]:
    """
    Generate a list of chunks of generators. You can think about it as a similar process to batching.
    We slice the generator yelding the list of all combinations to partition it according to the number 
    of available workers.

    Args:
        nprocs (int): number of workers it's defined by MPI4PY see default variables at the top.

    Returns:
        List[islice]: list of genrators which union covers all combinations of indexes
    """
    
    # Load the list of indexes (keys) that uniquely identify a record in the input data file.
    indexes: List[int] = loadIndexes()
    # Initialize the generator returning 2's combinations of all indexes.
    data_gen: combinations = combinations(indexes, 2)
    # Length of the list of indexes.
    index_len: int = len(indexes) 
    # we have to manually type the number of our combinations
    # unfortunatelly we cannot just consume the generator and 
    # check to size of the list of all combinations. It will occupy
    # too much memory (e.g. for index_len = 100k, the combinations of 2's 
    # is around 5*10^9. Assuming that indexes are integers such a list will
    # occupy: 56B * 5*10^9 = 280*10^9B = 280 GB. It is stored in the main memory, 
    # causing unavaidable out of memory error). 
    # Therefore, the size must be hardcoded; otherwise, we will have to the consume 
    # the gnerator what takes time. In our case it's very easy as it's the binomail theorem. 
    # For all combinations of 3's just multiply the current comb_len by: 
    # (index_len - 2)/3. You can use scipy.special.binom() to achieve the same result. 
    # Nevertheless, for the sake of readability we decided to hardcode this.  
    comb_len: int = index_len * (index_len - 1) / 2
    # the lines below slise the generator according to the avaluable number of workers.
    # We never actually touch our list of combinations as we just modify where the begning 
    # and the end of the generator is. We do it by creating the list of starting points and 
    # number of elements after the starting index to consume.
    ave, res = divmod(comb_len, nprocs)
    counts = [int(ave + 1 if p < res else ave) for p in range(nprocs)]
    starts = [int(sum(counts[:p])) for p in range(nprocs)]

    # finally we slice the generator according to starting index and size of elements. 
    # We achieve the list of generators that can be freely "send" scatered to multiple workers.
    return [generatorSlice(data_gen, starts[p], counts[p]) for p in range(nprocs)]

def loadIndexes(fileName: str = "inputData.csv") -> List[int]:
    """Load list of indexes (keys) that uniqualy point to records in input file.

    Args:
        fileName (str, optional): Name of the input file. Defaults to "inputData.csv".

    Returns:
        List[int]: list of indexes
    """
    res = []
    with open(fileName, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for row in spamreader:
            res.append(int(row[0].split(",")[0]))
    return res

def loadIndexDataMap(fileName: str = "inputData.csv") -> Dict[int, str]:
    """
    Load a map of indexes pointing to data from a record. !!!YOU MUST MODIFY THIS FUNCTION 
    ACCORDING TO YOUR NEEDS!!!. In our case, we only needed a string as our data. 

    Args:
        fileName (str, optional): Name of the input file. Defaults to "inputData.csv".

    Returns:
        Dict[int, str]: Map of index to data being tested.
    """
    

    res = {}
    with open(fileName, newline="") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=" ", quotechar="|")
        for row in spamreader:
            tmp = row[0].split(",")
            res[int(tmp[0])] = tmp[1]
    return res


# ----------------------------------------------------------------------------
# Helper functions specific for the LCS calculations
# ----------------------------------------------------------------------------

def process_and_encode_string(input_string):
    transformed = input_string.replace("(", "").replace(")", "") \
                              .replace("-XXX-", "-").replace("XXX-", "") \
                              .replace("-XXX", "").split("-")
    return [s.encode('utf-8') for s in transformed]

# ----------------------------------------------------------------------------
# Computations start here!!!
# Consider the code below as running on each worker in parallel.
# Additionally, we can use rank and nprocs to distinguish between workers.
# ----------------------------------------------------------------------------


output_dir_name = create_result_directory("results")

data_slices = chunk_combinations(nprocs) if rank == 0 else None

# scatter generators accross workers
data = comm.scatter(data_slices, root=0)

# Example of print that will be saved in log files.
print(f"rank: {rank}, numprocess: {nprocs}")

# we save results in results directory using independent files for each worker. 
# It provides distributed output to file hance reducing the risk of collecting results 
# by the coordinator (worker with rank 0) that might cause out of memory issues.
output_file_path = os.path.join(output_dir_name, f"output_{rank}.csv")

# write headers to output csv file
# you can freely modify it to your needs
with open(output_file_path, "w") as f:
    f.write("1,2,3,4,5,6,7,8\n")

# load dictionary pointing from index to data record.
# Using this map we pass data to our compute funciotn. In our case it's 
# PW_cython.compare_sequence(). It's cythonized python package that massively 
# speeds up the computation. We highelly recomend using cython indsted of python 
# to boost performance. You can also try codon: 
# https://github.com/exaloop/codon
# to compile native python to optimised executable
dataMap = loadIndexDataMap()

# iterat over combinations of indexed from generator and compute. 
# In our case we run pair-wise sequence alignment.
# conditionally safe results to our output file.
for ix, iy in data:
    score = LCS_cython.cluster_ZNFs(process_and_encode_string(dataMap[ix]), process_and_encode_string(dataMap[iy]))
    if score:
        with open(output_file_path, "a") as f:
            # Flatten the list, round the elements to two decimal places, and join without spaces
            score_str = ','.join([f"{round(s, 2):.2f}" for s in score[0]])
            f.write(f"{ix},{iy},{score_str}\n")


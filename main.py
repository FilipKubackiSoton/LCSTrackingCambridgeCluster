from mpi4py import MPI
import os
from datetime import datetime
from itertools import combinations, islice
#import PW_cython
import csv
from typing import List, Dict
import numpy as np
from collections import defaultdict
import json  # For saving the counter dictionary to a file easily
import os
import LCS_cython
# Default variable initialization for MPI4PY
# If you want to learn more, I recommend checking out:
# https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()
status = MPI.Status()   # get MPI status object
workers_update_filename = "workers_update.json"


# ----------------------------------------------------------------------------
# Below are generic reusable functions supporting the main structure of
# parallelization on the Cambridge HPC.
# ----------------------------------------------------------------------------


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
        
    indexes: List[int] = loadIndexes() # loadIndexes()
    # Initialize the generator returning 2's combinations of all indexes.
    data_gen: combinations = combinations(indexes, 2)
    # Length of the list of indexes.
    index_len: int = len(indexes) 
    nn = nprocs - 1
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
    ave, res = divmod(comb_len, nn)
    counts = [int(ave + 1 if p < res else ave) for p in range(nn)]
    starts = [int(sum(counts[:p])) for p in range(nn)]

    # finally we slice the generator according to starting index and size of elements. 
    # We achieve the list of generators that can be freely "send" scatered to multiple workers.
    return [generatorSlice(data_gen, starts[p], counts[p]) for p in range(nn)]

# ----------------------------------------------------------------------------
# Computations start here!!!
# Consider the code below as running on each worker in parallel.
# Additionally, we can use rank and nprocs to distinguish between workers.
# ----------------------------------------------------------------------------

class IncrementalMeanStdWelford:
    def __init__(self):
        self.n = 0
        self.mean = 0
        self.s = 0

    def add_element(self, x):
        self.n += 1
        old_mean = self.mean
        self.mean += (x - old_mean) / self.n
        self.s += (x - self.mean) * (x - old_mean)

    def get_current_mean(self):
        return self.mean

    def get_current_s(self):
        return self.s

    def __str__(self):
        return str(self.n) + "," + str("{:.2f}".format(self.get_current_mean())) + "," + str("{:.2f}".format(self.get_current_s())) + "\n"

def save_to_file(data, filename):
    """Save the given data to a file in JSON format."""
    with open(filename, 'w') as f:
        json.dump(data, f)

def coordinator_process():
    completed_workers = 0

    while completed_workers < nprocs - 1:
        # Receive data from any worker
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        # counter[status.Get_source()] +=1
        # totalcombinations += 1
        # if totalcombinations%frequecySave == 0:
        #     substepSaves += 1
        #     totalcombinations = 0
        #     save_to_file(counter, workers_update_filename)
        #     index_len = len(loadIndexes())
        #     comb_len: int = index_len * (index_len - 1) / 2
        #     print(f"Finished: {substepSaves*frequecySave/comb_len*100} %")

        # Check for a sentinel value indicating a worker has finished sending data
        if data is None:
            completed_workers += 1
            print(f"Compted worker rank: {completed_workers}")
            continue
        
        # Process the data (for example, update statistics)
        # ix, score = data
        ##############################################################
        # keep in mind that the score should ne just a single number
        # consider adding the extra functionality to covert more values from
        # score to single number
        ##############################################################
        #results[ix].add_element(score[2]) 

    # # After all data is processed, perform further analysis or save the results
    # with open("final_results.csv", "w") as f:
    #     for index, values in results.items():
    #         f.write(str(index) + "," + str(values))


def process_and_encode_string(input_string):
    transformed = input_string.replace("(", "").replace(")", "") \
                            .replace("-XXX-", "-").replace("XXX-", "") \
                            .replace("-XXX", "").split("-")
    return [s.encode('utf-8') for s in transformed]

# Worker process function
def worker_process(data, rank: int, processed_samples_num: int) -> None:

    dataMap = loadIndexDataMap()
    dataMapClean = {k: process_and_encode_string(v) for k, v in dataMap.items()}
    try: 
        datagen = islice(data, processed_samples_num, None)

        results = LCS_cython.cluster_ZNFs_test(list(datagen), dataMapClean)
        file_path = f"output/output_{rank}.csv"
        # Writing to the CSV file
        with open(file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Writing each list (row) into the CSV file
            for row in results:     
                # score_str = ','.join([f"{round(s, 2):.2f}" for s in row])
                writer.writerow(row)
        # for ix, iy in islice(data, processed_samples_num, None)  :
        #     progress_counter += 1
        #     progress_path = f"output/progress_{rank}.txt"
        #     with open(progress_path, "w") as p:
        #         p.write(str(progress_counter))
        #     seq1 = dataMap[ix]
        #     seq2 = dataMap[iy]
        #     score = LCS_cython.cluster_ZNFs(process_and_encode_string(seq1), process_and_encode_string(seq2))
        #     comm.send((ix, score), dest=0, tag=0)  # Sending to rank 0
        #     comm.send((iy, score), dest=0, tag=0)  # Sending to rank 0
        #     if score:
        #         print(score)
        #         with open(f"output/output_{rank}.csv", "a") as f:
        #             score_str = ','.join([f"{round(s, 2):.2f}" for s in score[0]])
        #             f.write(f"{ix},{iy},{score_str}\n")
    
    except Exception as e:
        print(f"Error writing to file: {e}")             

    # Send a sentinel value to indicate completion
    comm.send(None, dest=0, tag=1)


# Example of print that will be saved in log files.
print(f"rank: {rank}, numprocess: {nprocs}")

data_slices = chunk_combinations(nprocs) if rank == 0 else None
print(data_slices)

def load_from_file(filename):
    """Load data from a JSON file into a dictionary."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

counter = load_from_file(workers_update_filename) if os.path.exists(workers_update_filename) else defaultdict(lambda : 0)

if rank == 0:
    for i in range(1, nprocs):
        file_path = f"output/output_{i}.csv"
        if(not os.path.exists(file_path)):
            print(f"Creating output dir: {i}")
            with open(file_path, "w") as f:
                f.write("SourceIndex,OutputIndex,1,2,3,4,5,6,7,8\n")
        comm.send(data_slices[i-1], dest=i)
    coordinator_process()
else:
    counter = 0
    progress_path = f"output/progress_{rank}.txt"
    try:
        with open(progress_path, "r") as p:
            saved_counter = p.read().strip()  # Read and strip whitespace
            counter = int(saved_counter)  # Attempt conversion
    except Exception:
        print(f"NO progress for rank: {rank}")
        # Handle the case where conversion fails
        counter = 0  # Or another appropriate default value

    print(f"Worker: {rank} - counter: {counter}")
    data = comm.recv(source=0)
    worker_process(data, rank, counter)

# data = comm.scatter(data_slices, root=0)
    
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
    _, _, sequences = zs.loadDistances()
    
    indexes: List[int] = np.arange(len(sequences)) # loadIndexes()
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
    results = defaultdict(IncrementalMeanStdWelford)
    
    totalcombinations = 0
    frequecySave = 1000
    completed_workers = 0

    while completed_workers < nprocs - 1:
        # Receive data from any worker
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        counter[status.Get_source()] +=1
        totalcombinations += 1
        if totalcombinations%frequecySave == 0:
            totalcombinations = 0
            save_to_file(counter, workers_update_filename)
            #save(results)

        # Check for a sentinel value indicating a worker has finished sending data
        if data is None:
            completed_workers += 1
            print(completed_workers)
            continue
        
        # Process the data (for example, update statistics)
        ix, score = data
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



# Worker process function
def worker_process(data, rank):

    def process_and_encode_string(input_string):
        transformed = input_string.replace("(", "").replace(")", "") \
                                .replace("-XXX-", "-").replace("XXX-", "") \
                                .replace("-XXX", "").split("-")
        return [s.encode('utf-8') for s in transformed]

    distances, ZNF_seq, sequences = zs.loadDistances()
    new_blosum62_tuple, (df_new_blosum62, new_blosum_alpha, new_blosum_array) = zs.getMatrixPipeline()

    with open(f"output_{rank}.csv", "w") as f:
        f.write("1,2,3,4,5,6,7,8\n")
        # f.write("Index1,Index2,Score")
        for ix, iy in data:
            seq1 = sequences[ix]
            seq2 = sequences[iy]
            # dist1 = distances[ix]
            # dist2 = distances[iy]
            # seq1_conv = zs.convert_sequence(seq1.replace("-", ""), new_blosum_alpha)
            # seq2_conv = zs.convert_sequence(seq2.replace("-", ""), new_blosum_alpha)
            # score = zsp.testCompute(seq1_conv, seq2_conv, dist1, dist2, new_blosum_array)

            score = LCS_cython.cluster_ZNFs(process_and_encode_string(seq1), process_and_encode_string(seq2))

            comm.send((ix, score), dest=0, tag=0)  # Sending to rank 0
            comm.send((iy, score), dest=0, tag=0)  # Sending to rank 0
            if score:
                f.write(','.join([f"{round(s, 2):.2f}" for s in score[0]]))
                #f.write(str(ix) + "," + str(iy) + "," + str("{:.2f}".format(score)) + "\n")

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

counter = load_from_file(workers_update_filename) if os.path.exists(workers_update_filename) else defaultdict(0)

if rank == 0:
    for i in range(1, nprocs):
        comm.send(islice(data_slices[i-1], counter[i], None) , dest=i)
    coordinator_process()
else:
    data = comm.recv(source=0)
    worker_process(data, rank)

# data = comm.scatter(data_slices, root=0)
    
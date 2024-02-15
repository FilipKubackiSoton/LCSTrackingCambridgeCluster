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
import time
import math
from mpi4py.futures import MPIPoolExecutor
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

def process_and_encode_string(input_string):
    transformed = input_string.replace("(", "").replace(")", "") \
                            .replace("-XXX-", "-").replace("XXX-", "") \
                            .replace("-XXX", "").split("-")
    return [s.encode('utf-8') for s in transformed]

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
            res[int(tmp[0])] = process_and_encode_string(tmp[1])
    return res

def chunk_combinations(dataMap: Dict[int, any], nprocs: int) -> List[islice]:
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
        
    indexes: List[int] =list(dataMap.keys())
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
    return [islice(data_gen, starts[p],  starts[p] + counts[p], 1) for p in range(nn)]
    #return [generatorSlice(data_gen, starts[p], counts[p]) for p in range(nn)]

# ----------------------------------------------------------------------------
# Computations start here!!!
# Consider the code below as running on each worker in parallel.
# Additionally, we can use rank and nprocs to distinguish between workers.
# ----------------------------------------------------------------------------

def coordinator_process() -> None:
    completed_workers = 0
    while completed_workers < nprocs - 1:
        # Receive data from any worker
        data = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        # Check for a sentinel value indicating a worker has finished sending data
        if data is None:
            completed_workers += 1
            print(f"Rank calculations finished: {completed_workers} out of {nprocs - 1}")

def GetPartitionIndexes(sampleSize: int, partitionNum: int) -> [int]:
    res = [0]
    x = 0
    a = sampleSize*sampleSize/2/partitionNum
    for _ in range(partitionNum-1):
        res.append(sampleSize - math.sqrt((sampleSize - x)**2 - 2*a))
        x = res[-1]
    return [int(z) for z in res] + [sampleSize]

# # Worker process function
# def worker_process(partitionStart: int, partitionEnd: int, rank: int) -> None:
#     def getCombinationsSparseRanges(n: int, p: int):
#         chunksize = n//p
#         res = [chunksize*i for i in range(p)]
#         res.append(n)
#         return res
    
#     dataMap = loadIndexDataMap()
#     dataMapLen = len(dataMap.keys())

#     sizePartition = partitionEnd - partitionStart
#     progressFlags = getCombinationsSparseRanges(sizePartition, 10)
#     i = 0
#     startTime = time.time()
#     for indexIn in range(partitionStart, partitionEnd, 1):
#         LCS_cython.cluster_ZNFs_parition2(indexIn, dataMapLen, dataMap, rank)      
#         if((progressFlags[i] + partitionStart)==indexIn):
#             elapsedTime = time.time() - startTime
#             startTime = time.time()
#             remainingTime = (10 - i) * elapsedTime
#             print(f'rank: {rank} - parition completed: {i} out of 10, in time: {time.strftime("%H:%M:%S", time.gmtime(elapsedTime))}, time remaining: {time.strftime("%H:%M:%S", time.gmtime(remainingTime))}')
#             i += 1
        
#     # Send a sentinel value to indicate completion
#     comm.send(None, dest=0, tag=1)

def worker_process(dataSlice: islice, rank: int) -> None:
    dataMap = loadIndexDataMap()
    with MPIPoolExecutor() as executor:
        score = executor.map(LCS_cython.cluster_ZNF, ((dataMap[ix], dataMap[iy]) for ix, iy in dataSlice))
        for x in score: 
            if x > 6.0:
                print(f"rank: {rank}, score: {x}")

    # size = 0 
    # for ix, iy in dataSlice:
    #     score = LCS_cython.cluster_ZNF(dataMap[ix], dataMap[iy]) 
    #     if score > 6.0: 
    #         print(f"IndexIN: {ix}, IndexOUT: {iy}, Score: {score}")
    #     size += 1
    #print(f"Worker at rank {rank} received the count of indexes: {size}")
    comm.send(None, dest=0, tag=1)

if __name__ == "__main__":

    dataMap = loadIndexDataMap() if rank == 0 else None
    data_slices = chunk_combinations(dataMap, nprocs) if rank == 0 else None
    paritions = GetPartitionIndexes(len(dataMap.keys()),  nprocs-1) if rank == 0 else None

    if rank == 0:
        print("---------------------------------------------------")
        print("PARTITIONS")
        print("---------------------------------------------------")
        print(paritions)
        print("---------------------------------------------------")

        for i in range(1, nprocs):
            # Creat output file for individual worker
            file_path = f"output/output_{i}.csv"
            if(not os.path.exists(file_path)):
                print(f"Creating output dir: {i}")
                with open(file_path, "w") as f:
                    f.write("SourceIndex,OutputIndex,1,2,3,4,5,6,7,8\n")
            # send data generator slice to the worker
            comm.send(data_slices[i-1], dest=i)
        # after sending all the data request start the coordinator process to which workers can comunicate
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
        worker_process(data, rank)
        #worker_process(rank,dataMapClean, dataMapLen, paritions[rank-1], paritions[rank])
        #worker_process(paritions[rank-1], paritions[rank], rank)


    # data = comm.scatter(data_slices, root=0)
        
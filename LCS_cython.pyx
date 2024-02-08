from copy import deepcopy
import numpy as np

cimport cython
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list as c_list
from libcpp.map cimport map
from libcpp cimport bool
from libcpp.pair cimport pair
from cython.operator cimport dereference as deref
from cython.parallel import parallel, prange
from libc.stdlib cimport malloc, free
from Bio.Align import substitution_matrices

cdef inline int int_max(int a, int b): return a if a >= b else b

#from Bio.SubsMat.MatrixInfo import blosum62, ident
blosum62 = substitution_matrices.load("BLOSUM62")
ident = blosum62

cimport openmp

def sanitize_matrix(original_matrix):
    matrix = deepcopy(original_matrix)
    matrix.update(((b,a),val) for (a,b),val in matrix.items())
    matrix.update(((b,'X'),0) for (a,b),val in matrix.items())
    matrix.update((('X',a),0) for (a,b),val in matrix.items())
    
    for key,val in matrix.items():
        if key[0] == key[1]:
            matrix[key] = 1
        elif val > 0:
            matrix[key] = 0.5
        else:
            matrix[key] = 0
    
    matrix.update((('X','X'),0) for (a,b),val in matrix.items())
    
        
    return matrix

def sanitize_matrix2(original_matrix, equal=False):
    matrix = deepcopy(original_matrix)
    
    matrix.update(((b,a),val) for (a,b),val in matrix.items())
    matrix.update(((b,'X'),0.0) for (a,b),val in matrix.items())
    matrix.update((('X',a),0.0) for (a,b),val in matrix.items())
    
    for key,val in matrix.items():
        if key[0] == key[1]:
            matrix[key] = 1
        elif val > 0.0:
            matrix[key] = 0.5
        elif (equal is True) and (val == 0.0):
            matrix[key] = 0.5
        else:
            matrix[key] = 0.0
    
    matrix.update((('X','X'),0) for (a,b),val in matrix.items())
    
    matrix2 = dict()        
    for key,val in matrix.items():
        a = key[0]
        b = key[1]
        matrix2[(ord(a),ord(b))] = val
    
    return matrix2

@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void longest_common_subsequence(const vector[string] &xx, const vector[string] &yy, vector[vector[pair[int,int]]] &C, int m, int n, const map[pair[char, char], float] &matrix) nogil:
    cdef int i, j
    cdef int left, over
    cdef pair[int, int] cell
    cdef pair[char, char] entry
    #0 is match, 1 is left, 2 is up
    for j in xrange(1, n+1):
        for i in xrange(1, m+1):
            #if xx[i-1] == yy[j-1]:
            if calculate_sim(xx[i-1], yy[j-1], matrix, entry) >= 2.5:
                cell.first = C[j-1][i-1].first + 1
                cell.second = 0
            else:
                left = C[j][i-1].first
                over = C[j-1][i].first
                if left < over:
                    cell.first = over
                    cell.second = 2
                else:
                    cell.first = left
                    cell.second = 1
            C[j][i] = cell

@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[float] calculate_lcs(const vector[string] &xx, const vector[string] &yy, const map[pair[char, char], float] &matrix) nogil:
    cdef int i, j
    cdef float len_a, len_b
    cdef c_list[string] ret
    cdef int l
    
    cdef vector[vector[pair[int, int]]] C
    cdef int max_length
    cdef float count = 0
    cdef vector[vector[pair[int, int]]] temp
    cdef vector[pair[int, int]] temp_vector
    
    cdef pair[int, int] default_pair
    default_pair.first = 0
    default_pair.second = -1
    cdef float A_gap = 0
    cdef float B_gap = 0
    cdef float A_gap_temp = 0
    cdef float B_gap_temp = 0
    cdef float A_gap_extend = 0
    cdef float B_gap_extend = 0
    cdef float A_gap_extend_temp = 0
    cdef float B_gap_extend_temp = 0
    
    cdef float A_score = 0
    cdef float B_score = 0
    cdef float A_score_gap = 0
    cdef float B_score_gap = 0
    
    cdef vector[float] results
    results.resize(2, 0)
    
    cdef float min_ratio = 0
    cdef float min_ratio_gap = 0
   
    i = xx.size()
    j = yy.size()
    
    len_a = <float>i
    len_b = <float>j
    
    if i > 0 and j > 0:
        max_length = max(i, j) 
        C.resize(j+1, vector[pair[int, int]] (i+1, default_pair))
        longest_common_subsequence(xx, yy, C, i, j, matrix)
        while i > 0 or j > 0:
            l = C[j][i].second
            if l == 0:
                count = count + 1
                if count > 1:
                    A_gap += A_gap_temp
                    B_gap += B_gap_temp
                    A_gap_extend += A_gap_extend_temp
                    B_gap_extend += B_gap_extend_temp
                A_gap_temp = 0
                B_gap_temp = 0
                A_gap_extend_temp = 0
                B_gap_extend_temp = 0
                i -= 1
                j -= 1
            elif l == 1:
                i -= 1
                if A_gap_temp == 0:
                    A_gap_temp = 1
                else:
                    A_gap_extend_temp += 1
            elif l == 2:
                j -= 1
                if B_gap_temp == 0:
                    B_gap_temp = 1
                else:
                    B_gap_extend_temp += 1
            else:
                break

        A_score = count / len_a
        B_score = count / len_b
       
        A_score_gap = (count - ((A_gap * 0.5) + (A_gap_extend * 0.1))) / len_a
        B_score_gap = (count - ((B_gap * 0.5) + (B_gap_extend * 0.1))) / len_b
        
        if A_score_gap < 0:
            A_score_gap = 0
        if B_score_gap < 0:
            B_score_gap = 0
            
        min_ratio = min(A_score, B_score) * 10.0
        min_ratio_gap = min(A_score_gap, B_score_gap) * 10.0
        
        results[0] = min_ratio
        results[1] = min_ratio_gap
        
    return results


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
cdef float calculate_sim(const string &string1, const string &string2, const map[pair[char, char], float] &matrix, pair[char, char] &entry) nogil:
    cdef int i
    cdef int m, n
    cdef float score_temp = 0
    cdef float score = 0
    for i in xrange(3):
        entry.first = string1[i]
        entry.second = string2[i]
        score += deref(matrix.find(entry)).second
    return score

###
"""
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[float] calculate_lcs_ratio(vector[string] &ZNF1, vector[string] &ZNF2,
                              float ZNF1_len, float ZNF2_len, vector[vector[float]] &results,
                              const map[pair[char, char], float] &blosum, const map[pair[char, char], float] &identity,
                              const map[pair[char, char], float] &blosum_mismatch, const map[pair[char, char], float] &identity_mismatch) nogil:
    cdef float min_ratio_ident, min_ratio_gap_ident
    cdef float min_ratio_blosum, min_ratio_gap_blosum
    cdef float min_ratio_ident_mismatch, min_ratio_gap_ident_mismatch
    cdef float min_ratio_blosum_mismatch, min_ratio_gap_blosum_mismatch
    cdef vector[float] temp_results
    cdef vector[float] lcs_results
    
    
    lcs_results = calculate_lcs(ZNF1, ZNF2, blosum_mismatch)
    min_ratio_blosum_mismatch = lcs_results[0]
    min_ratio_gap_blosum_mismatch = lcs_results[1]
    
    if min_ratio_blosum_mismatch >= 6.0:
        lcs_results = calculate_lcs(ZNF1, ZNF2, blosum)
        min_ratio_blosum = lcs_results[0]
        min_ratio_gap_blosum = lcs_results[1]
    
        lcs_results = calculate_lcs(ZNF1, ZNF2, identity)
        min_ratio_ident = lcs_results[0]
        min_ratio_gap_ident = lcs_results[1]
        
        lcs_results = calculate_lcs(ZNF1, ZNF2, identity_mismatch)
        min_ratio_ident_mismatch = lcs_results[0]
        min_ratio_gap_ident_mismatch = lcs_results[1]
        
        temp_results.resize(8)
        temp_results[0] = min_ratio_ident
        temp_results[1] = min_ratio_gap_ident
        temp_results[2] = min_ratio_blosum
        temp_results[3] = min_ratio_gap_blosum
        temp_results[4] = min_ratio_ident_mismatch
        temp_results[5] = min_ratio_gap_ident_mismatch
        temp_results[6] = min_ratio_blosum_mismatch
        temp_results[7] = min_ratio_gap_blosum_mismatch

    return temp_results

        
from threading import Lock
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cluster_ZNFs(const vector[string] &ZNF1, const vector[string] &ZNF2):
   
    cdef map[pair[char, char], float] blosum
    cdef map[pair[char, char], float] identity
    cdef map[pair[char, char], float] blosum_mismatch
    cdef map[pair[char, char], float] identity_mismatch
    
    blosum = sanitize_matrix2(blosum62)
    identity = sanitize_matrix2(ident)
    blosum_mismatch = sanitize_matrix2(blosum62, equal=True)
    identity_mismatch = sanitize_matrix2(ident, equal=True)
    
    cdef float ZNF1_len = ZNF1.size()
    cdef float ZNF2_len = ZNF2.size()
    cdef vector[vector[float]] results
    
    temp_result = calculate_lcs_ratio(ZNF1, ZNF2, ZNF1_len, ZNF2_len, results, blosum, identity, blosum_mismatch, identity_mismatch)
    
    if not temp_result.empty():
        results.push_back(temp_result)
    
    return results


"""
###


@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void calculate_lcs_ratio(int i, int j, vector[string] &ZNF1, vector[string] &ZNF2,
                              float ZNF1_len, float ZNF2_len, vector[vector[float]] &results,
                              const map[pair[char, char], float] &blosum, const map[pair[char, char], float] &identity,
                              const map[pair[char, char], float] &blosum_mismatch, const map[pair[char, char], float] &identity_mismatch,
                              openmp.omp_lock_t &mylock) nogil:
    cdef float min_ratio_ident, min_ratio_gap_ident
    cdef float min_ratio_blosum, min_ratio_gap_blosum
    cdef float min_ratio_ident_mismatch, min_ratio_gap_ident_mismatch
    cdef float min_ratio_blosum_mismatch, min_ratio_gap_blosum_mismatch
    cdef vector[float] temp_results
    cdef vector[float] lcs_results
    
    
    
    lcs_results = calculate_lcs(ZNF1, ZNF2, blosum_mismatch)
    min_ratio_blosum_mismatch = lcs_results[0]
    min_ratio_gap_blosum_mismatch = lcs_results[1]
    #with gil:
    #    print("-----------------")
    #    print(ZNF1)
    #    print(ZNF2)
    #    print(lcs_results)

    if min_ratio_blosum_mismatch >= 6.0:
        lcs_results = calculate_lcs(ZNF1, ZNF2, blosum)
        min_ratio_blosum = lcs_results[0]
        min_ratio_gap_blosum = lcs_results[1]
    
        lcs_results = calculate_lcs(ZNF1, ZNF2, identity)
        min_ratio_ident = lcs_results[0]
        min_ratio_gap_ident = lcs_results[1]
        
        lcs_results = calculate_lcs(ZNF1, ZNF2, identity_mismatch)
        min_ratio_ident_mismatch = lcs_results[0]
        min_ratio_gap_ident_mismatch = lcs_results[1]
        
        #if min_ratio_ident >= 3.33:
        temp_results.resize(10)
        temp_results[0] = <float>i
        temp_results[1] = <float>j
        temp_results[2] = min_ratio_ident
        temp_results[3] = min_ratio_gap_ident
        temp_results[4] = min_ratio_blosum
        temp_results[5] = min_ratio_gap_blosum
        temp_results[6] = min_ratio_ident_mismatch
        temp_results[7] = min_ratio_gap_ident_mismatch
        temp_results[8] = min_ratio_blosum_mismatch
        temp_results[9] = min_ratio_gap_blosum_mismatch
        
        #with gil:
        #    print("-----------------")
        #    print(lcs_results)

        openmp.omp_set_lock(&mylock)
        results.push_back(temp_results)
        openmp.omp_unset_lock(&mylock)


"""
from threading import Lock
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cluster_ZNFs(const vector[vector[string]] &ZNFs_list):
   
    cdef map[pair[char, char], float] blosum
    cdef map[pair[char, char], float] identity
    cdef map[pair[char, char], float] blosum_mismatch
    cdef map[pair[char, char], float] identity_mismatch
    
    blosum = sanitize_matrix2(blosum62)
    identity = sanitize_matrix2(ident)
    blosum_mismatch = sanitize_matrix2(blosum62, equal=True)
    identity_mismatch = sanitize_matrix2(ident, equal=True)
    
    cdef int i, j
    cdef vector[string] ZNF1
    cdef vector[string] ZNF2
    cdef float ZNF1_len
    cdef float ZNF2_len
    cdef int ZNFs_length
    cdef float lcs_len
    cdef int index
    cdef openmp.omp_lock_t mylock
    openmp.omp_init_lock(&mylock)
    ZNFs_length = ZNFs_list.size()
    cdef vector[vector[float]] results
    with nogil, parallel():
        for i in prange(ZNFs_length, schedule="dynamic"):
            ZNF1_len = ZNFs_list[i].size()
            for j in xrange(i+1, ZNFs_length):
                ZNF2_len = ZNFs_list[j].size()
                calculate_lcs_ratio(i, j, ZNFs_list[i], ZNFs_list[j], ZNF1_len, ZNF2_len, results, blosum, identity, blosum_mismatch, identity_mismatch, mylock)
    openmp.omp_destroy_lock(&mylock)
    return results
"""

cdef vector[string] convert_to_vector_string(list byte_strings):
    cdef vector[string] result
    for byte_string in byte_strings:
        if not isinstance(byte_string, bytes):
            raise ValueError("All items in the list must be bytes.")
        # Convert Python bytes to std::string and add to the vector
        result.push_back(byte_string)
    return result


cdef vector[vector[string]] convert_to_cpp_vector(dict dataMap):
    cdef vector[vector[string]] ZNFs_list
    cdef vector[string] ZNF_group
    for key, sequences in dataMap.items():
        ZNFs_list.push_back(convert_to_vector_string(sequences))
        # ZNF_group = vector[string]()
        # for sequence in sequences:
        #     ZNF_group.push_back(sequence.encode('utf-8'))  # Assuming Python 3 strings (unicode), need to encode to bytes
        # ZNFs_list.push_back(ZNF_group)
    return ZNFs_list

cdef vector[vector[int]] convert_list_of_tuples_to_cpp_vector(list tuples_list):
    cdef vector[vector[int]] cpp_vector
    cdef vector[int] temp_vector
    for tuple_ in tuples_list:
        # Ensure the tuple has exactly 2 integers
        if len(tuple_) == 2 and all(isinstance(x, int) for x in tuple_):
            temp_vector = vector[int]()
            # Manually add each integer from the tuple to the temp_vector
            for item in tuple_:
                temp_vector.push_back(item)
            cpp_vector.push_back(temp_vector)
        else:
            raise ValueError("All items in the list must be tuples of 2 integers")
    return cpp_vector


from threading import Lock
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cluster_ZNFs_test(generator, dataMapDict):
    
    # Assuming `generator` is a Python generator that yields tuples of ints (i, j)
    
    # Convert generator to a list of tuples for processing
    tuples_list = generator # list(generator)
    tuples_length = len(tuples_list)
    numpy_array_of_tuples = np.array(tuples_list, dtype=np.int32)
    tuples_array = np.array(list(generator), dtype=np.int32)

    cdef map[pair[char, char], float] blosum
    cdef map[pair[char, char], float] identity
    cdef map[pair[char, char], float] blosum_mismatch
    cdef map[pair[char, char], float] identity_mismatch
    
    blosum = sanitize_matrix2(blosum62)
    identity = sanitize_matrix2(ident)
    blosum_mismatch = sanitize_matrix2(blosum62, equal=True)
    identity_mismatch = sanitize_matrix2(ident, equal=True)
    
    cdef int i, j
    cdef vector[string] ZNF1
    cdef vector[string] ZNF2
    cdef float ZNF1_len
    cdef float ZNF2_len
    cdef int ZNFs_length
    cdef float lcs_len
    cdef int index
    cdef openmp.omp_lock_t mylock
    cdef int num_tuples = tuples_array.shape[0]
    cdef vector[vector[string]] dataMap
    cdef vector[vector[int]] indexesTuple

    dataMap = convert_to_cpp_vector(dataMapDict)
    indexesTuple = convert_list_of_tuples_to_cpp_vector(tuples_list)
    openmp.omp_init_lock(&mylock)
    print(f"Number of num_tuples: {num_tuples}")
    cdef vector[vector[float]] results
    with nogil, parallel():
        for index in prange(num_tuples, schedule="dynamic"):
            # i, j = tuples_list[index]  # This line is not valid Cython syntax because you cannot directly assign to i, j in nogil
            i = indexesTuple[index][0]
            j = indexesTuple[index][1]

            ZNF1 = dataMap[i]
            ZNF2 = dataMap[j]
            ZNF1_len = ZNF1.size()
            ZNF2_len = ZNF2.size()
            calculate_lcs_ratio(i, j, ZNF1, ZNF2, ZNF1_len, ZNF2_len, results, blosum, identity, blosum_mismatch, identity_mismatch, mylock)
    openmp.omp_destroy_lock(&mylock)
    return results
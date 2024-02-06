cimport cython
import numpy as np
cimport numpy as cnp
from libcpp.string cimport string
from libcpp.vector cimport vector

from libcpp.list cimport list as c_list
from libc.math cimport floor, ceil, fmax
from libcpp.map cimport map
from libcpp.pair cimport pair
from cython.parallel import parallel, prange
cdef map[pair[int, int], float] blosum62
cdef pair[int, int] entry

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef (float) sequence_compare_cython(const vector[int] &seq1, const vector[int] &seq2, const double[:,:] &matrix) nogil:
    cdef int i, j, k
    cdef float score, total_alignment_score

    cdef int gap_open_penalty = 12
    cdef int gap_extension_penalty = 1
    cdef int endpenalty = 3
    
    cdef int len_seq1
    cdef int len_seq2
    len_seq1 = seq1.size() // 3
    len_seq2 = seq2.size() // 3
    cdef vector[float] pairscore
    pairscore.resize((len_seq1+1) * (len_seq2+1), 0)
    cdef vector[float] gaplocation
    gaplocation.resize((len_seq1+1) * (len_seq2+1), 0)
    cdef vector[float] cumul_score
    cumul_score.resize((len_seq1+1) * (len_seq2+1), 0)

    # Initialize the cumul_score for gap penalties at the beginning of the sequences.
    for i in range(len_seq1+1):
        cumul_score[i*(len_seq2+1)] = -i*endpenalty
    for j in range(len_seq2+1):
        cumul_score[j] = -j*endpenalty
        
    for i in range(1, len_seq1+1):
        for j in range(1, len_seq2+1):
            score = 0
            for k in range(0, 3):
                score += matrix[seq1[3*(i-1)+k], seq2[3*(j-1)+k]]
                
            pairscore[i*(len_seq2+1) + j] = score
    
    cdef vector[float] gap_open_seq1
    cdef vector[float] gap_open_seq2
    gap_open_seq1.resize((len_seq1+1) * (len_seq2+1), 0)
    gap_open_seq2.resize((len_seq1+1) * (len_seq2+1), 0)
    
    # Align and score the middle of the sequences and assign affine gap penalties
    cdef float localmax, temp
    for i in range(1, len_seq1+1):
        for j in range(1, len_seq2+1):
            localmax = cumul_score[((i-1)*(len_seq2+1)) + (j-1)] + pairscore[i*(len_seq2+1) + j]
            temp = cumul_score[((i-1)*(len_seq2+1)) + j] - gap_open_penalty
            if temp > localmax:
                localmax = temp
            temp = cumul_score[(i*(len_seq2+1)) + (j-1)] - gap_open_penalty
            if temp > localmax:
                localmax = temp
            temp = gap_open_seq1[((i-1)*(len_seq2+1)) + j] - gap_extension_penalty
            if temp > localmax:
                localmax = temp
            temp = gap_open_seq2[(i*(len_seq2+1)) + (j-1)] - gap_extension_penalty
            if temp > localmax:
                localmax = temp
            
            if cumul_score[((i-1)*(len_seq2+1)) + (j-1)] + pairscore[i*(len_seq2+1) + j] == localmax:
                gaplocation[i*(len_seq2+1) + j] = -1
            elif cumul_score[((i-1)*(len_seq2+1)) + j] - gap_open_penalty == localmax:
                gaplocation[i*(len_seq2+1) + j] = 1
            elif cumul_score[(i*(len_seq2+1)) + (j-1)] - gap_open_penalty == localmax:
                gaplocation[i*(len_seq2+1) + j] = 2
            elif gap_open_seq1[((i-1)*(len_seq2+1)) + j] - gap_extension_penalty == localmax:
                gaplocation[i*(len_seq2+1) + j] = 3
            else:
                gaplocation[i*(len_seq2+1) + j] = 4
                
            cumul_score[i*(len_seq2+1) + j] = localmax
            gap_open_seq1[i*(len_seq2+1) + j] = max(
                gap_open_seq1[((i-1)*(len_seq2+1)) + j] - gap_extension_penalty,
                cumul_score[((i-1)*(len_seq2+1)) + j] - gap_open_penalty
            )
            gap_open_seq2[i*(len_seq2+1) + j] = max(
                gap_open_seq2[(i*(len_seq2+1)) + (j-1)] - gap_extension_penalty,
                cumul_score[(i*(len_seq2+1)) + (j-1)] - gap_open_penalty
            )

    # Align the end of the sequences
    cdef float end_gap_1_max, end_gap_2_max
    end_gap_1_max = -9999
    for i in range(0, len_seq2+1):
        temp = cumul_score[len_seq1*(len_seq2+1) + len_seq2-i] - i*endpenalty
        if temp > end_gap_1_max:
            end_gap_1_max = temp
    
    end_gap_2_max = -9999
    for i in range(0, len_seq1+1):
        temp = cumul_score[(len_seq1-i)*(len_seq2+1) + len_seq2] - i*endpenalty
        if temp > end_gap_2_max:
            end_gap_2_max = temp
    
    if end_gap_1_max >= end_gap_2_max:
        total_alignment_score = end_gap_1_max
    else:
        total_alignment_score = end_gap_2_max
    
    return total_alignment_score


##################################

from Bio.Align import substitution_matrices
import numpy as np
import itertools

def compare_sequences(outputPath, sequences_df):
    blosum62 = np.asarray(substitution_matrices.load("BLOSUM62"))
    blosum_alpha = substitution_matrices.load("BLOSUM62").alphabet

    def convert_sequence(seq):
        seq_conv = np.zeros(len(seq), dtype=np.int32)
        for i in range(len(seq)):
            seq_conv[i] = blosum_alpha.index(seq[i])
        return seq_conv

    for config in sequences_df:
        source_index, source_seq, target_index, target_seq = config["source_index"], config["source"], config["target_index"], config["target"]

        seq1_conv = convert_sequence(source_seq.replace("-", ""))
        seq2_conv = convert_sequence(target_seq.replace("-", ""))
        
        raw_score = sequence_compare_cython(seq1_conv, seq2_conv, blosum62)
        longer_length = max(len(seq1_conv), len(seq2_conv))
        adjusted_score = raw_score / (longer_length/3)

        if adjusted_score > 0:
            with open(outputPath, 'a') as f:
                f.write(str(source_index)+","+str(target_index)+","+str(raw_score)+","+str(adjusted_score)+"\n")
            
        
def compare_sequence( source_seq, target_seq):
    blosum62 = np.asarray(substitution_matrices.load("BLOSUM62"))
    blosum_alpha = substitution_matrices.load("BLOSUM62").alphabet

    def convert_sequence(seq):
        seq_conv = np.zeros(len(seq), dtype=np.int32)
        for i in range(len(seq)):
            seq_conv[i] = blosum_alpha.index(seq[i])
        return seq_conv

    seq1_conv = convert_sequence(source_seq.replace("-", ""))
    seq2_conv = convert_sequence(target_seq.replace("-", ""))
    
    raw_score = sequence_compare_cython(seq1_conv, seq2_conv, blosum62)
    longer_length = max(len(seq1_conv), len(seq2_conv))
    adjusted_score = raw_score / (longer_length/3)
    return raw_score, adjusted_score
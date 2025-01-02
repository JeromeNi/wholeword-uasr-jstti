import os
import pickle
import numpy as np
import torch


## the output word boundaries txt file from `eval_w2b.sh`
word_boundaries_txt = f'/path/to/all_word_boundaries.txt'

## where the original frame-level hubert features are stored. We need the number of frames for each utterance
data_lengths_fn = f'/path/to/hubert_feats/feat_hubert_large_L21/valid_0_1.len'

## You can compare the stats with the original word boundaries upon which wav2boundaries is trained on. Modify to the directory where your word segmentation (say, from GradSeg) is stored. 
topk = 2048
manifest_path = f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/gradseg_unsup_segmentation/'

# point to where you stored the downloaded file lists. These are used to filter out bad files.
file_list_fn= "/path/to/valid_file_list.txt" 

## from eval_segmentation.py at vqwordseg
def boundaries_to_intervals(boundaries):
    intervals = []
    j_prev = 0
    for j in np.where(boundaries)[0]:
        intervals.append((j_prev, j + 1))
        j_prev = j + 1
    return intervals


def intervals_to_boundaries(intervals):
    boundaries = np.zeros(intervals[-1][1], dtype=bool)
    boundaries[[i[1] - 1 for i in intervals]] = True
    return boundaries

def score_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the segmentation boundaries.
    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.
    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_boundaries_ref = 0
    n_boundaries_seg = 0
    n_boundaries_correct = 0
    for i_boundary, boundary_ref in enumerate(ref):
        boundary_seg = seg[i_boundary]
        assert boundary_ref[-1]  # check if last boundary is True
        assert boundary_seg[-1]
        
        # If lengths are the same, disregard last True reference boundary
        if len(boundary_ref) == len(boundary_seg):
            boundary_ref = boundary_ref[:-1]
            # boundary_seg = boundary_seg[:-1]

        boundary_seg = seg[i_boundary][:-1]  # last boundary is always True,
                                             # don't want to count this

        # If reference is longer, truncate
        if len(boundary_ref) > len(boundary_seg):
            boundary_ref = boundary_ref[:len(boundary_seg)]
        
        boundary_ref = list(np.nonzero(boundary_ref)[0])
        boundary_seg = list(np.nonzero(boundary_seg)[0])
        n_boundaries_ref += len(boundary_ref)
        n_boundaries_seg += len(boundary_seg)

        for i_seg in boundary_seg:
            for i, i_ref in enumerate(boundary_ref):
                if abs(i_seg - i_ref) <= tolerance:
                    n_boundaries_correct += 1
                    boundary_ref.pop(i)
                    break

    # Temp
#     print("n_boundaries_correct", n_boundaries_correct)
#     print("n_boundaries_seg", n_boundaries_seg)
#     print("n_boundaries_ref", n_boundaries_ref)

    precision = float(n_boundaries_correct)/n_boundaries_seg
    recall = float(n_boundaries_correct)/n_boundaries_ref
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf

    return precision, recall, f




def score_word_token_boundaries(ref, seg, tolerance=0):
    """
    Calculate precision, recall, F-score for the word token boundaries.
    Parameters
    ----------
    ref : list of vector of bool
        The ground truth reference.
    seg : list of vector of bool
        The segmentation hypothesis.
    tolerance : int
        The number of slices with which a boundary might differ but still be
        regarded as correct.
    Return
    ------
    output : (float, float, float)
        Precision, recall, F-score.
    """
    n_tokens_ref = 0
    n_tokens_seg = 0
    n_tokens_correct = 0
    for i_boundary, boundary_ref in enumerate(ref):
        boundary_seg = seg[i_boundary]
        assert boundary_ref[-1]  # check if last boundary is True
        assert boundary_seg[-1]
        
        # The code below shouldn't be done for token scores
        # # If lengths are the same, disregard last True reference boundary
        # if len(boundary_ref) == len(boundary_seg):
        #     boundary_ref = boundary_ref[:-1]
        # boundary_seg = seg[i_boundary][:-1]  # last boundary is always True,
                                             # don't want to count this

        # If reference is longer, truncate
        if len(boundary_ref) > len(boundary_seg):
            boundary_ref = boundary_ref[:len(boundary_seg)]
            boundary_ref[-1] = True

        # Build list of ((word_start_lower, word_start_upper), (word_end_lower,
        # word_end_upper))
        word_bound_intervals = []
        for word_start, word_end in boundaries_to_intervals(boundary_ref):
            word_bound_intervals.append((
                (max(0, word_start - tolerance), word_start + tolerance),
                (word_end - tolerance, word_end + tolerance)
                ))
        seg_intervals = boundaries_to_intervals(boundary_seg)

        n_tokens_ref += len(word_bound_intervals)
        n_tokens_seg += len(seg_intervals)

        # Score word token boundaries
        for seg_start, seg_end in seg_intervals:
            # print seg_start, seg_end
            for i_gt_word, (word_start_interval,
                    word_end_interval) in enumerate(word_bound_intervals):
                word_start_lower, word_start_upper = word_start_interval
                word_end_lower, word_end_upper = word_end_interval

                if (word_start_lower <= seg_start <= word_start_upper and
                        word_end_lower <= seg_end <= word_end_upper):
                    n_tokens_correct += 1
                    word_bound_intervals.pop(i_gt_word)  # can't re-use token
                    # print "correct"
                    break

    # # Temp
    # print("n_tokens_correct", n_tokens_correct)
    # print("n_tokens_seg", n_tokens_seg)
    # print("n_tokens_ref", n_tokens_ref)

    precision = float(n_tokens_correct)/n_tokens_seg
    recall = float(n_tokens_correct)/n_tokens_ref
    if precision + recall != 0:
        f = 2*precision*recall / (precision + recall)
    else:
        f = -np.inf

    return precision, recall, f


def get_os(precision, recall):
    """Calculate over segmentation score."""
    if precision == 0:
        return -np.inf
    else:
        return recall/precision - 1


def get_rvalue(precision, recall):
    """Calculate the R-value."""
    os = get_os(precision, recall)
    r1 = np.sqrt((1 - recall)**2 + os**2)
    r2 = (-os + recall - 1)/np.sqrt(2)
    rvalue = 1 - (np.abs(r1) + np.abs(r2))/2
    return rvalue

length_dict = {}


with open(os.path.join(file_list_fn), 'r') as f:
    file_lines = f.readlines()
    
with open(os.path.join(data_lengths_fn), 'r') as f:
    length_lines = f.readlines()
    
for file_line, length_line in zip(file_lines, length_lines):
    length_dict[file_line.strip()] = int(length_line.strip())


with open(word_boundaries_txt, 'r') as f:
    word_boundary_lines = f.readlines()

filenames = []
predicted_boundaries = []
for word_boundary_line in word_boundary_lines:
    word_boundary_splitted = word_boundary_line.strip().split()
    filename = 'train-clean' + word_boundary_splitted[0].split('train-clean')[1]
    filenames.append(filename)
    boundary_arr = np.zeros(length_dict[filename])
    for word_boundary_pairs in word_boundary_splitted[2:]:
        start, end = word_boundary_pairs.split(',')
        start = round(float(start) * 50)
        end = round(float(end) * 50)
#         print(start, end)
        
        if start < length_dict[filename]:
            boundary_arr[start] = 1
        elif start > length_dict[filename] + 1:
            print(start, length_dict[filename])

        if end < length_dict[filename]:
            boundary_arr[end] = 1
        elif end > length_dict[filename] + 1:
            print(end, length_dict[filename])
    boundary_arr[-1] = 1
    boundary_arr[0] = 1
    predicted_boundaries.append(boundary_arr)


with open(os.path.join(manifest_path, 'valid_utts.pkl'), 'rb') as fb:
    boundary_dict = pickle.load(fb)

ref_boundaries = []
seg_boundaries = []


for idx, filename in enumerate(filenames):
    boundary_arr = np.zeros(len(predicted_boundaries[idx])).astype('int32')
    boundaries = boundary_dict[filename]
    
#     print(len(boundary_arr))

    boundary_locs = np.where(boundaries['ref_bound'] == 1)[0]
    while (boundary_locs//2)[-1]  >= len(boundary_arr):
        boundary_locs = boundary_locs[:-1]
    #print(boundary_locs //2, len(boundary_arr))
    boundary_arr[boundary_locs // 2] = 1
    boundary_arr[0] = 1
    boundary_arr[-1] = 1
    ref_boundaries.append(boundary_arr)

    boundary_arr = np.zeros(len(predicted_boundaries[idx])).astype('int32')
    boundaries = boundary_dict[filename]

    boundary_locs = np.where(boundaries['seg_bound'] == 1)[0]
    #print(boundary_locs //2, len(boundary_arr))
    while (boundary_locs//2)[-1]  >= len(boundary_arr):
        boundary_locs = boundary_locs[:-1]
    #print(boundary_locs //2, len(boundary_arr))
    boundary_arr[boundary_locs // 2] = 1
    boundary_arr[-1] = 1
    boundary_arr[0] = 1
    seg_boundaries.append(boundary_arr)

# print(boundary_dict[filenames[bad_idx[0]]]['ref_bound'])
bt_p, bt_r, bt_f = score_boundaries(ref_boundaries, predicted_boundaries, tolerance = 1)
print(bt_p, bt_r, bt_f)

bt_p, bt_r, bt_f = score_boundaries(ref_boundaries, seg_boundaries, tolerance = 1)
print(bt_p, bt_r, bt_f)


bt_p, bt_r, bt_f = score_word_token_boundaries(ref_boundaries, predicted_boundaries, tolerance = 1)
print(bt_p, bt_r, bt_f)

bt_p, bt_r, bt_f = score_word_token_boundaries(ref_boundaries, seg_boundaries, tolerance = 1)
print(bt_p, bt_r, bt_f)


############### END OF EVALULATION ###############
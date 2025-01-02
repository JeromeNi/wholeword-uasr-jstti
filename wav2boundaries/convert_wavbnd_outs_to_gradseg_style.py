import os
import pickle
import numpy as np

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

flag="prom_xx_height_xx"
topk=2048
audio_dir=f'/path/to/LibriSpeech_clean_pruned_top{topk}_no_sil/'
for which_set in ['train', 'valid', 'valid_true']:
    length_dict = {}

    ## boundaries location from eval script of wav2boundaries
    word_boundaries_txt = f'/path/to/saved_boundary_outputs/{which_set}_{flag}/all_word_boundaries.txt'

    ## the HuBERT features directory containing frame length for each utterance
    length_dir = f'/path/to/hubert_feats/feat_hubert_large_L21/'

    ## the directory containing train/valid/valid_true_file_list.txt
    file_list_txt_dir = "/path/to/file_lists" 
    
    ## in case prediction error on some utterances in wav2bnd outputs, back off into the corresponding segmentation of this set of unsupervised word boundaries instead; usually, this is the teacher labels prepared for wav2bnd training
    manifest_path=f'{audio_dir}/your_dir_to_utts_pkl_for_wav2bnd_teachers/'
    
    ## where to save the new boundaries
    new_manifest_path = f'{audio_dir}/your_dir_to_utts_pkl_after_applying_wav2bnd_to_teachers/'

    with open(os.path.join(file_list_txt_dir, f'{which_set}_file_list.txt'), 'r') as f:
        file_lines = f.readlines()

    with open(os.path.join(length_dir, f'{which_set}_0_1.len'), 'r') as f:
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

            if start < length_dict[filename]:
                boundary_arr[start] = 1
            elif start > length_dict[filename] + 1:
                print(start, length_dict[filename], filename)

            if end < length_dict[filename]:
                boundary_arr[end] = 1
            elif end > length_dict[filename] + 1:
                print(end, length_dict[filename], filename)
        # boundary_arr[-1] = 1
        boundary_arr[0] = 1
        predicted_boundaries.append(boundary_arr)


    predicted_boundaries_dict = {}

    for fn, pred_bnd in zip(filenames, predicted_boundaries):
        predicted_boundaries_dict[fn] = pred_bnd

    os.makedirs(new_manifest_path, exist_ok = True)

    
    with open(os.path.join(manifest_path, f'{which_set}_utts.pkl'), 'rb') as fb:
        boundary_dict = pickle.load(fb)


    all_available_filenames = list(boundary_dict.keys())


    new_boundary_dict = {}

    ref_boundaries = []
    seg_boundaries = []


    for fidx, filename in enumerate(all_available_filenames):
        if len(boundary_dict[filename]['ref_bound']) != len(boundary_dict[filename]['seg_bound']):
            print(filename, 'wtf')


        if fidx % 1000 == 0:
            print(fidx)
        new_boundary_dict[filename] = {}

        # curr_pred_bnd = predicted_boundaries_dict[filename]

        if filename in predicted_boundaries_dict:
            curr_pred_bnd = predicted_boundaries_dict[filename]
        #     boundary_arr = np.zeros(len(curr_pred_bnd) * 2).astype('int32')
        # else:
        boundary_arr = np.zeros(len(boundary_dict[filename]['ref_bound'])).astype('int32')
        boundary_arr = boundary_dict[filename]['ref_bound']
        boundary_arr[0] = 1
        boundaries = boundary_dict[filename]

#         ref_boundary_locs = np.where(boundaries['ref_bound'] == 1)[0]

#         for ridx in range(len(ref_boundary_locs)):
#             if ref_boundary_locs[ridx] >= len(boundary_arr):
#                 print(ref_boundary_locs[ridx], len(boundary_arr))
#                 ref_boundary_locs[ridx] = len(boundary_arr) -1

#         boundary_arr[ref_boundary_locs] = 1
#         boundary_arr[0] = 1

        new_boundary_dict[filename]['ref_bound'] =  boundary_arr


        # boundary_arr = np.zeros(len(curr_pred_bnd) * 2).astype('int32')
        
        boundaries = boundary_dict[filename]
        boundary_arr = np.zeros(len(boundary_dict[filename]['ref_bound'])).astype('int32')
        

        if filename not in predicted_boundaries_dict:
#             seg_boundary_locs = np.where(boundaries['seg_bound'] == 1)[0]

#             for ridx in range(len(seg_boundary_locs)):
#                 if seg_boundary_locs[ridx] >= len(boundary_arr):
#                     print(seg_boundary_locs[ridx], len(boundary_arr), filename)
#                     seg_boundary_locs[ridx] = len(boundary_arr) -1

#             boundary_arr[seg_boundary_locs] = 1
#             boundary_arr[0] = 1
            boundary_arr = boundaries['seg_bound']
            boundary_arr[0] = 1
        else:
            new_pred_bnds = predicted_boundaries_dict[filename]
            seg_boundary_locs = np.where(new_pred_bnds == 1)[0] * 2
            boundary_arr[seg_boundary_locs] = 1
            boundary_arr[0] = 1


        new_boundary_dict[filename]['seg_bound'] = boundary_arr


    with open(os.path.join(new_manifest_path, f'{which_set}_utts.pkl'), 'wb') as fb:
        pickle.dump(new_boundary_dict, fb)
    
    

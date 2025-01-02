import os
import pickle
import numpy as np
import torch

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
    
    

which_set = 'valid_true'


bad_fns = []
bad_fns_txt = f'/nobackup/users/junruin2/data/hubert/libri-train-clean/feat_hubert_large_l21_ori_feats_no_sil_wav2bnd/top_1024/km_dir_1024/discrete_speech_{which_set}_clsattnbndkmeans/bnd_errors.txt'
with open(bad_fns_txt, 'r') as f:
    bad_lines = f.readlines()
    
for bad_line in bad_lines:
    bad_fns.append(bad_line.strip().split()[0])

bad_fns = set(bad_fns)

# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top4096_no_sil/vghubert_unsup_segmentation/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top2048_no_sil/wav2boundaries_gradseg_unsup_0.419tokenf1_e2e_refinement_increase_pen/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top1024_no_sil/wav2boubdaries_gradseg_unsup_segmentation_updated'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top1024_no_sil/wav2boundaries_gradseg_unsup_unpaired_text_train_other_500_e2e_refinement_no_codebook_prob/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top1024_no_sil/wav2boundaries_gradseg_unsup_unpaired_text_train_other_500_e2e_refinement_no_codebook_prob_w2b/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top1024_no_sil/wav2boundaries_gradseg_unsup_unpaired_text_train_other_500_e2e_refinement_no_codebook_prob_init_from_1e-3_model_2e-4lr/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top4096_no_sil/wav2boundaries_gradseg_unsup_tokenf10.351/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top4096_no_sil/wav2boundaries_gradseg_unsup_from_1024_e2e'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top4096_no_sil/wav2boundaries_gradseg_unsup_from_2048_e2e'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top4096_no_sil/wav2boundaries_gradseg_unsup_from_1024_e2e_w2b_large_no_opt/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top4096_no_sil/wav2boundaries_gradseg_unsup_from_2048_e2e_w2b_large_no_opt/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top4096_no_sil/wav2boundaries_gradseg_unsup_from_1024_e2e_w2b_large_no_opt_balanced_f1/'


new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top2048_no_sil/wav2boundaries_gradseg_unsup_0.419tokenf1_e2e_refinement/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top2048_no_sil/wav2boundaries_gradseg_unsup_0.419tokenf1/'
# new_manifest_path = '/nobackup/users/junruin2/LibriSpeech_clean_pruned_top2048_no_sil/wav2boundaries_gradseg_unsup_diff_bnd_large_batch_from_init_codebook_0/'
test_manifest_path = new_manifest_path
# print(test_manifest_path)
with open(os.path.join(test_manifest_path, f'{which_set}_utts.pkl'), 'rb') as fb:
    boundary_dict = pickle.load(fb)
    print(len(boundary_dict))

ref_boundaries = []
seg_boundaries = []


for idx, filename in enumerate(list(boundary_dict.keys())):
    if filename not in bad_fns:
        boundaries = boundary_dict[filename]
        
        boundary_arr = np.zeros(len(boundaries['ref_bound']) // 2)
        boundary_arr[np.where(boundaries['ref_bound'] == 1)[0] // 2] = 1
        boundary_arr[0] = 1
        boundary_arr[-1] = 1
        ref_boundaries.append(boundary_arr)

        boundary_arr = np.zeros(len(boundaries['seg_bound']) // 2)
        boundary_arr[np.where(boundaries['seg_bound'] == 1)[0] // 2] = 1
        boundary_arr[-1] = 1
        boundary_arr[0] = 1
        seg_boundaries.append(boundary_arr)
print(len(ref_boundaries))
bt_p, bt_r, bt_f = score_boundaries(ref_boundaries, seg_boundaries, tolerance = 1)
print(bt_p, bt_r, bt_f)


bt_p, bt_r, bt_f = score_word_token_boundaries(ref_boundaries, seg_boundaries, tolerance = 1)
print(bt_p, bt_r, bt_f)
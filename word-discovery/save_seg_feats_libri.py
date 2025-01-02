import argparse
import torch
import os
import pickle
import json
import soundfile as sf
import tqdm
import time
import numpy as np
from models import audio_encoder
import tqdm
import numpy as np
from itertools import groupby
from operator import itemgetter
import copy
from npy_append_array import NpyAppendArray


def cls_attn_seg_feats(feats, cls_attn_weights, threshold, pool, spf, level2,no_cls):
    # return a list of features that are segmented by cls attn weights
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    cls_attn_weights_sum = cls_attn_weights.sum(0)
    important_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].cpu().numpy()
    boundaries = []
    boundaries_all = []
    boundaries_ex1 = []
    for k, g in groupby(enumerate(important_idx), lambda ix : ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1]+1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            boundaries_all.append([t_s, t_e])
            boundaries_ex1.append([t_s, t_e])
        else:
            boundaries_all.append([t_s, t_e])
    
    if level2 or len(boundaries_ex1) == 0:
        boundaries = boundaries_all
    else:
        boundaries = boundaries_ex1
    # boundaries = boundaries_all
    seg_feats = []
    locations = []
    boundaries_in_sec = []
    boundaries_in_frame = []
    for t_s, t_e in boundaries:
        locations.append(spf*(t_s+t_e)/2.) # in seconds
        boundaries_in_sec.append([t_s*spf, t_e*spf]) # in seconds
        boundaries_in_frame.append([t_s, t_e]) # in frames
        if pool == "mean":
            seg_feats.append(feats[t_s:t_e].mean(0).cpu())
        elif pool == "max":
            max_id = torch.argmax(cls_attn_weights_sum[t_s:t_e])
            seg_feats.append(feats[t_s+max_id].cpu())
        elif pool == "median":
            seg_feats.append(feats[int((t_s+t_e)/2)].cpu())
        elif pool == "weightedmean":
            seg_feats.append((feats[t_s:t_e]*(cls_attn_weights_sum[t_s:t_e]/cls_attn_weights_sum[t_s:t_e].sum()).unsqueeze(1)).sum(0).cpu())
    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries_in_sec, "boundaries_frame": boundaries_in_frame}


def cls_attn_bnd_seg_feats(feats, cls_attn_weights, threshold, pool, spf, level2, no_cls, audio_length):
    # return a list of features that are segmented by cls attn weights
    threshold_value = torch.quantile(cls_attn_weights, threshold, dim=-1, keepdim=True) # [n_h, T]
    cls_attn_weights_sum = cls_attn_weights.sum(0)
    important_idx = torch.where((cls_attn_weights >= threshold_value).float().sum(0) > 0)[0].cpu().numpy()
    boundaries = []
    boundaries_all = []
    boundaries_ex1 = []
    for k, g in groupby(enumerate(important_idx), lambda ix : ix[0] - ix[1]):
        seg = list(map(itemgetter(1), g))
        t_s, t_e = seg[0], min(seg[-1]+1, cls_attn_weights.shape[-1])
        if len(seg) > 1:
            boundaries_all.append([t_s, t_e])
            boundaries_ex1.append([t_s, t_e])
        else:
            boundaries_all.append([t_s, t_e])
    
    if level2 or len(boundaries_ex1) == 0:
        boundaries = boundaries_all
    else:
        boundaries = boundaries_ex1
    
    word_boundaries_intervals = []
    word_boundaries_list = [boundaries[0][0]/2] # fist boundary
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        word_boundaries_list.append((left[1]+right[0])/2.)
    word_boundaries_list.append((boundaries[-1][1]+audio_length)/2) # last boundary
    for i in range(len(word_boundaries_list)-1):
        word_boundaries_intervals.append([word_boundaries_list[i], word_boundaries_list[i+1]])
    
    boundaries = copy.deepcopy(word_boundaries_intervals)  

    # boundaries = boundaries_all
    seg_feats = []
    locations = []
    boundaries_in_sec = []
    boundaries_in_frame = []
    for t_s, t_e in boundaries:
        locations.append(spf*(t_s+t_e)/2.) # in seconds
        boundaries_in_sec.append([t_s*spf, t_e*spf]) # in seconds
        boundaries_in_frame.append([t_s, t_e]) # in frames
        if pool == "mean":
            seg_feats.append(feats[int(t_s):int(t_e)].mean(0).cpu())
        elif pool == "max":
            seg_feats.append(feats[int(t_s):int(t_e)].max(0)[0].cpu())
        elif pool == "median":
            seg_feats.append(feats[int((t_s+t_e)/2)].cpu())
    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries_in_sec, "boundaries_frame": boundaries_in_frame}


def force_align_seg_feats(feats, text_alignment, fps, pool):
    seg_feats = []
    locations = []
    boundaries = []
    meta_toks = text_alignment.split(" ")
    for meta_tok in meta_toks:
        toks = meta_tok.split('__')
        if len(toks) == 3:
            s = float(toks[0])
            e = float(toks[2])
            boundaries.append([s,e])
            locations.append((s+e)/2.)
            if int(s*fps) == int(e*fps):
                seg_feats.append(feats[int(s*fps)].cpu())
            else:
                if pool == "mean":
                    seg_feats.append(feats[int(s*fps):int(e*fps)].mean(0).cpu())
                elif pool == "max":
                    print(int(s*fps), int(e*fps), len(feats))
                    seg_feats.append(feats[int(s*fps):int(e*fps)].max(0)[0].cpu())
                elif pool == "median":
                    seg_feats.append(feats[int((s*fps+e*fps)/2)].cpu())
    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries}


def sample_pool_feats(feats,  fps, pool,factor = 13):
    seg_feats = []
    locations = []
    boundaries = []
    seg_len = int(len(feats) / factor)
    for f in range(seg_len):
        boundaries.append([f * factor / fps, (f+1) * factor / fps])
        locations.append((f+0.5) * factor / fps)
        if pool == "mean":
            seg_feats.append(feats[f * factor: (f+1) * factor].mean(0).cpu())
        elif pool == "max":
            seg_feats.append(feats[f * factor: (f+1) * factor].max(0)[0].cpu())
        elif pool == "median":
            seg_feats.append(feats[f * factor: (f+1) * factor].cpu())

    return {"seg_feats": seg_feats, "locations": locations, "boundaries": boundaries}

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp_dir", type=str)
parser.add_argument("--dataset", type=str, default='spokencoco')
parser.add_argument("--data_json", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO/SpokenCOCO_val_unrolled_karpathy_with_alignments.json")
parser.add_argument("--audio_base_path", type=str, default="/data1/scratch/coco_pyp/SpokenCOCO")
parser.add_argument("--save_root", type=str, default="/data2/scratch/pyp/discovery/word_unit_discovery/")
parser.add_argument("--percentage", type=int, default=None, help="if None, the feats_type is the original name, otherwise, it's feats_type_percentage")
parser.add_argument("--threshold", type=float, default=0.90)
parser.add_argument("--reduce_method", type=str, default="mean", choices=['mean', 'max', 'median', 'weightedmean'])
parser.add_argument("--tgt_layer_for_attn", type=int, default=7, help="where attn weights are coming from, as for features, if feats_type==preFeats, and feature comes from previous layer of tgt_layer_for_attn, otherwise, feature comes from the same layer")
parser.add_argument("--level2", action="store_true", default=False, help="if True, use feats and atten weights from level2 (not avaliable for models that only has one level of w2v2)")
parser.add_argument("--segment_method", type=str, choices=['clsAttn', 'clsAttnBnd','forceAlign', 'samplePool', 'perturbed'], default=None, help="if use cls attn segmentation or use force alignment segmentation. If use, need model_args.use_audio_cls_token to be True")
parser.add_argument("--ori_feats", action="store_true", default=False)
args = parser.parse_args()

save_root = os.path.join(args.save_root, args.exp_dir.split("/")[-1])
feats_type = args.dataset + "_" + args.reduce_method + "_" + str(args.threshold) + "_" + str(args.tgt_layer_for_attn) + "_" + args.segment_method

if args.percentage is not None:
    feats_type = feats_type + "_" + str(args.percentage)
save_root = os.path.join(save_root, feats_type)
print("data save at: ", save_root)
os.makedirs(save_root, exist_ok=True)
print(args)
if not os.path.isdir(args.exp_dir):
    raise RuntimeError(f"{args.exp_dir} does not exist!!")

########################## setup model ##########################
with open(os.path.join(args.exp_dir, "args.pkl"), "rb") as f:
    model_args = pickle.load(f)
model = audio_encoder.AudioEncoder(model_args)
bundle = torch.load(os.path.join(args.exp_dir, "best_bundle.pth"))
if "dual_encoder" in bundle:
    model.carefully_load_state_dict(bundle['dual_encoder'], load_all=True)
else:
    model.carefully_load_state_dict(bundle['model'], load_all=True)
model.eval()
model = model.cuda()
########################## setup model ##########################


data_start_time = time.time()

with open(args.data_json, "r") as f:
    data_json = json.load(f)['data']

locF_temp = []
j = 0

data_dict = {}
missing_ali = 0

level2 = False
tgt_layer = args.tgt_layer_for_attn
wav_fn_list = []

if args.ori_feats:
    npaa = NpyAppendArray(os.path.join(save_root, "data.npy"))
    npaa_len = []

for item in tqdm.tqdm(data_json):
    if "librispeech" in args.dataset:
        wav_fn = item['caption']['wav']
        key = os.path.join(args.audio_base_path, item['caption']['wav'])
    elif args.dataset == "timit":
        wav_fn = item['wav']
        key = item['wav']
    audio, sr = sf.read(key, dtype = 'float32')
    assert sr == 16000
    with torch.no_grad():
        w2v2_out = model(torch.from_numpy(audio).unsqueeze(0).cuda(), padding_mask=None, mask=False, need_attention_weights=True, tgt_layer=tgt_layer)
    
    if args.segment_method == "clsAttn": # use cls attn for segmentation
        if not (model_args.use_audio_cls_token and model_args.cls_coarse_matching_weight > 0.):
            no_cls = True
        else:
            no_cls = False
        feats = w2v2_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
        spf = len(audio)/sr/feats.shape[-2]
        attn_weights = w2v2_out['attn_weights'].squeeze(0) # [1, num_heads, tgt_len, src_len] -> [num_heads, tgt_len, src_len]
        if no_cls:
            cls_attn_weights = (attn_weights.sum(1) - attn_weights[:,range(attn_weights.shape[2]),range(attn_weights.shape[2])]).squeeze().cpu()
        else:
            cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
        out = cls_attn_seg_feats(feats, cls_attn_weights, args.threshold, args.reduce_method, spf, level2, no_cls)
        
    if args.segment_method == "clsAttnBnd": # use cls attn for segmentation
        if not (model_args.use_audio_cls_token and model_args.cls_coarse_matching_weight > 0.):
            no_cls = True
        else:
            no_cls = False
        feats = w2v2_out['features'].squeeze(0)[1:] # [1, T+1, D] -> [T, D]
        spf = len(audio)/sr/feats.shape[-2]
        attn_weights = w2v2_out['attn_weights'].squeeze(0) # [1, num_heads, tgt_len, src_len] -> [num_heads, tgt_len, src_len]
        
        if no_cls:
            cls_attn_weights = (attn_weights.sum(1) - attn_weights[:,range(attn_weights.shape[2]),range(attn_weights.shape[2])]).squeeze().cpu()
        else:
            cls_attn_weights = attn_weights[:, 0, 1:] # [num_heads, tgt_len, src_len] -> [n_h, T]
        out = cls_attn_bnd_seg_feats(feats, cls_attn_weights, args.threshold, args.reduce_method, spf, level2, no_cls, len(feats))
    elif args.segment_method == "forceAlign":
        if model_args.use_audio_cls_token:
            feats = w2v2_out['features'].squeeze(0)[1:]
        else:
            feats = w2v2_out['features'].squeeze(0)
        spf = len(audio)/sr/feats.shape[-2]
        if 'text_alignment' in item:
            out = force_align_seg_feats(feats, item['text_alignment'], fps=1./spf, pool=args.reduce_method)
        else:
            missing_ali += 1
            out = {'seg_feats': [feats.mean(0).cpu() if args.reduce_method == "mean" else feats.max(0)[0].cpu()], "locations": [feats.shape[0]//2], 'boundaries': [[0, len(audio)/sr]]}


    elif args.segment_method == "perturbed":
        if model_args.use_audio_cls_token:
            feats = w2v2_out['features'].squeeze(0)[1:]
        else:
            feats = w2v2_out['features'].squeeze(0)
        spf = len(audio)/sr/feats.shape[-2]
        if 'perturbed_text_alignment' in item:
            out = force_align_seg_feats(feats, item['perturbed_text_alignment'], fps=1./spf, pool=args.reduce_method)
        else:
            print('Error!')
            missing_ali += 1
            out = {'seg_feats': [feats.mean(0).cpu() if args.reduce_method == "mean" else feats.max(0)[0].cpu()], "locations": [feats.shape[0]//2], 'boundaries': [[0, len(audio)/sr]]}


    elif args.segment_method == "samplePool":
        if model_args.use_audio_cls_token:
            feats = w2v2_out['features'].squeeze(0)[1:]
        else:
            feats = w2v2_out['features'].squeeze(0)
        spf = len(audio)/sr/feats.shape[-2]
        if len(feats) >= 15:
            out = sample_pool_feats(feats,  fps=1./spf, pool=args.reduce_method)
        else:
            missing_ali += 1
            out = {'seg_feats': [feats.mean(0).cpu() if args.reduce_method == "mean" else feats.max(0)[0].cpu()], "locations": [feats.shape[0]//2], 'boundaries': [[0, len(audio)/sr]]}


    seg_feats = out['seg_feats']
    seg_feats = torch.stack(seg_feats).cpu()
    
    
    wav_fn_list.append(wav_fn)
    
    if args.ori_feats:
        feat_arr = feats.cpu().numpy()
        npaa.append(feat_arr)
        npaa_len.append(len(feat_arr))
        data_dict[wav_fn] = {"seg_feats": seg_feats, "locations": torch.tensor(out['locations']), "boundaries": torch.tensor(out['boundaries']), "boundaries_frame": torch.tensor(out['boundaries_frame']), "spf": spf}
    else:
        data_dict[wav_fn] = {"seg_feats": seg_feats, "locations": torch.tensor(out['locations']), "boundaries": torch.tensor(out['boundaries']), "boundaries_frame": torch.tensor(out['boundaries_frame']), "spf":spf}   
if args.segment_method == "forceAlign":
    print(f"missing alignments: {missing_ali}")

with open(os.path.join(save_root, 'data_dict.pkl'), "wb") as f:
    pickle.dump(data_dict, f)
    
if args.ori_feats: 
    with open(os.path.join(save_root, 'data.lengths'), 'w') as f:
        for length in npaa_len:
            f.write(str(length) + '\n')
with open(os.path.join(save_root, 'file_list.txt'), 'w') as f:
    for filename in wav_fn_list:
        f.write(filename + '\n')

print(f"save pickle data at {os.path.join(save_root, 'data_dict.pkl')}")



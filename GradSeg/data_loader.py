from boltons import fileutils
import numpy as np
import torch
import torchaudio
import json
import os


# def get_data_buckeye(path, max_files):

# 	wavs = list(fileutils.iter_find_files(path, "*.wav"))
# 	all_wavs = []
# 	all_bounds = []
# 	all_pseudo_bounds = []			
# 	rp = np.random.permutation(len(wavs))
# 	wavs = [wavs[i] for i in rp]
# 	if max_files != -1:
# 		wav_iter = wavs[:max_files]
# 	else:
# 		wav_iter = wavs
# 	for wav in wav_iter:
# 		word_fn = wav.replace("wav", "word")
# 		words = open(word_fn, 'r').readlines()
# 		words = [w.strip().split() for w in words]
# 		bounds = [(int(w[0]), int(w[1])) for w in words]
     
# 		pseudo_word_fn = wav.replace("wav", "vghubert.word")
# 		pseudo_words = open(pseudo_word_fn, 'r').readlines()
# 		pseudo_words = [w.strip().split() for w in pseudo_words ]
# 		pseudo_bounds = [(int(w[0]), int(w[1])) for w in pseudo_words]

# 		waveform, sample_rate = torchaudio.load(wav)
# 		if len(bounds) > 0:
# 			all_wavs.append(waveform)
# 			all_bounds.append(bounds)
# 			all_pseudo_bounds.append(pseudo_bounds)
# 	return all_wavs, all_bounds, all_pseudo_bounds

def get_data_buckeye(file_tsv, file_boundary_json):

	with open(file_tsv, 'r') as f:
		tsv_lines = f.readlines()
		parent_dir = tsv_lines[0].strip()
		file_lines = tsv_lines[1:]
  
	with open(file_boundary_json, 'r') as f:
		boundary_dict = json.load(f)
	
	wavs = [os.path.join(parent_dir, file_line.strip().split()[0]) for file_line in file_lines]
	wavs_fns = [file_line.strip().split()[0] for file_line in file_lines]
	all_wavs = []
	all_bounds = []
	all_fns = []
	rp = np.random.permutation(len(wavs))
	wavs = [wavs[i] for i in rp]
	wavs_fns = [wavs_fns[i] for i in rp]
	for idx, wav in enumerate(wavs):
		words = boundary_dict[wavs_fns[idx]]
		bounds = [(int(w[0]), int(w[1])) for w in words]

		waveform, sample_rate = torchaudio.load(wav)
		# if len(bounds) > 0:
		all_wavs.append(waveform)
		all_bounds.append(bounds)
		all_fns.append(wavs_fns[idx])
	return all_wavs, all_bounds, all_fns


def get_emb(wavs, model, layer=-1, feat_idx=-1):
	es = []
	for waveform in wavs:
		e = embed(waveform, model, layer, feat_idx)
		es.append(e)
	return es

def get_single_emb(wavform, model, layer=-1, feat_idx=-1):
	return embed(wavform, model, layer, feat_idx)

def embed(y, model, extract_layer=-1, feat_idx = -1):
	with torch.no_grad():
		model.eval()
		y = torch.Tensor(y).cuda()
		x, _ = model.extract_features(y)
		x = x[extract_layer]
		if not feat_idx == -1:
			x = x[:,:,feat_idx]
	return x.data.cpu().numpy()[0]


def get_model(arc):
	if arc == 'BASE': #12 output layers
					bundle = torchaudio.pipelines.WAV2VEC2_BASE
					my_dim = 768
	elif arc == 'LARGE': #24 output layers
					bundle = torchaudio.pipelines.WAV2VEC2_LARGE
					my_dim = 1024
	elif arc == 'LARGE_LV60K': #24 output layers
					bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
					my_dim = 1024
	elif arc == 'XLSR53': #24 output layers
					bundle = torchaudio.pipelines.WAV2VEC2_XLSR53
					my_dim = 1024
	elif arc == 'HUBERT_BASE':
					bundle = torchaudio.pipelines.HUBERT_BASE
					my_dim = 768
	elif arc == 'HUBERT_LARGE':
					bundle = torchaudio.pipelines.HUBERT_LARGE
					my_dim = 1024
	elif arc == 'HUBERT_XLARGE':
					bundle = torchaudio.pipelines.HUBERT_XLARGE
					my_dim = 1280 #?
	else:   
					bundle = torchaudio.pipelines.WAV2VEC2_BASE #WAV2VEC2_BASE WAV2VEC2_LARGE WAV2VEC2_LARGE_LV60K WAV2VEC2_XLSR53
					my_dim = 768
	return bundle.get_model(), my_dim


def get_bounds(boundaries):
	l = [0]
	for i in range(len(boundaries)-1):
					l.append((boundaries[i][1] + boundaries[i+1][0])//2)
	l.append(boundaries[-1][1])
	return l

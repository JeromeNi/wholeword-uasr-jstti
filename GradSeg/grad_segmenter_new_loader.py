import torch
import argparse
import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier, LogisticRegression
import eval_segmentation
import data_loader
import pickle
import os


parser = argparse.ArgumentParser(description='GradSeg word segmentation')

parser.add_argument('--train_tsv', type=str, default='/cs/labs/yedid/yedid/wordcluster/buckeye_processed/train', help='tsv file for training data')
parser.add_argument('--val_tsv', type=str, default='/cs/labs/yedid/yedid/wordcluster/buckeye_processed/val', help='dir of .wav, .wrd files for val data')

parser.add_argument('--train_boundary_json', type=str, default=200, help='file boundary json for training data')
parser.add_argument('--val_boundary_json', type=str, default=200, help='file boundary json for evalutation data')


parser.add_argument('--layer', type=int, default=-6, help='layer index (output)')
parser.add_argument('--offset', type=int, default=0, help='offset to window center')
parser.add_argument('--arc', type=str, default='BASE', help='model architecture options: BASE, LARGE, LARGE_LV60K, XLSR53, HUBERT_BASE, HUBERT_LARGE, HUBERT_XLARGE')

parser.add_argument('--min_separation', type=int, default=4, help='min separation between words')
parser.add_argument('--target_perc', type=int, default=40, help='target quantization percentile')

parser.add_argument('--frames_per_word', type=int, default=10, help='5 words in a second')
parser.add_argument('--loss', type=str, default='ridge', help='ridge || logres')
parser.add_argument('--C', type=float, default=1.0, help='logistic regression parameter')
parser.add_argument('--reg', type=float, default=1e4, help='ridge regularization')

parser.add_argument('--save_bounds_dir', type=str, default="", help='where to save the boundaries')
parser.add_argument('--save_name', type=str, default="", help='where to save the boundaries')
parser.add_argument('--eval_metrics', default=True, action=argparse.BooleanOptionalAction)

args = parser.parse_args()
print(args)


def get_grad_mag(e):
    e = np.pad(e,1, mode='reflect')
    e = e[2:] - e[:-2]
    mag = e ** 2
    return mag.mean(1)

def get_seg(d, num_words, min_separation):
  idx = np.argsort(d)
  selected = []
  for i in idx[::-1]:
    if len(selected) >= num_words:
      break
    if len(selected) == 0 or (np.abs(np.array(selected) - i)).min() > min_separation:
      selected.append(i)
  return np.sort(np.array(selected))


frames_per_embedding = 160 #(not 320), instead of multiplying by 2 later

#init seeds 
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Model init
model, dim = data_loader.get_model(args.arc)
model = model.cuda()





train_wavs, train_bounds, train_fns = data_loader.get_data_buckeye(args.train_tsv, args.train_boundary_json)
val_wavs, val_bounds, val_fns = data_loader.get_data_buckeye(args.val_tsv, args.val_boundary_json)

train_e = data_loader.get_emb(train_wavs, model, args.layer)
# val_e = data_loader.get_emb(val_wavs, model, args.layer)

print("frame duration (s): %f" % (frames_per_embedding/16000))


ds = []
for idx in range(len(train_e)):
  d = get_grad_mag(train_e[idx])
  ds.append(d)

ds = np.concatenate(ds)
th = np.percentile(ds, args.target_perc)
targets = ds > th
if args.loss == 'ridge':
  clf = Ridge(alpha=args.reg)
else:
  clf = LogisticRegression(C = args.C, max_iter = 1000)
train_e_np = np.concatenate(train_e)
mu = train_e_np.mean(0)[None, :]
std =  train_e_np.std(0)[None, :]
clf.fit((train_e_np - mu)/std, targets)

ref_bounds = []
seg_bounds = []
save_bounds_dict = {}
for idx in range(len(val_wavs)):
        curr_embed = data_loader.get_single_emb(val_wavs[idx], model, args.layer)
        if args.loss == 'logres':
          d = clf.predict_proba((curr_embed - mu)/std)[:,1]
        else:
          d = clf.predict((curr_embed - mu)/std)
        num_words = int(len(curr_embed)/args.frames_per_word)
        p = get_seg(d, num_words, args.min_separation)


        p = p * 2 + args.offset
        p = np.minimum(p, 2*(len(d)-1))
        p = p.astype('int')
        
        if len(val_bounds[idx]) > 0:
            boundaries = np.array(data_loader.get_bounds(val_bounds[idx])) // frames_per_embedding
            boundaries = np.minimum(boundaries[1:-1], (len(d)-1)*2)
            
            ref_bound = np.zeros(len(d)*2)
            seg_bound = np.zeros(len(d)*2)
            ref_bound[boundaries] = 1
            seg_bound[p] = 1
            ref_bound[-1] = 1
            seg_bound[-1] = 1
            ref_bounds.append(ref_bound)
            seg_bounds.append(seg_bound)
            
            save_bounds_dict[val_fns[idx]] = {"ref_bound": ref_bound, "seg_bound": seg_bound}
        else:
            seg_bound = np.zeros(len(d)*2)
            seg_bound[p] = 1
            seg_bound[-1] = 1
            seg_bounds.append(seg_bound)
            
            save_bounds_dict[val_fns[idx]] = {"ref_bound": None, "seg_bound": seg_bound}            
        
        
if args.save_bounds_dir != "":
    save_fn = os.path.join(args.save_bounds_dir, args.save_name + '.pkl')
    with open(save_fn, 'wb') as f:
        pickle.dump(save_bounds_dict, f)

if args.eval_metrics:
    precision, recall, f = eval_segmentation.score_boundaries(ref_bounds, seg_bounds, 2)
    os = eval_segmentation.get_os(precision, recall)*100
    r_val = eval_segmentation.get_rvalue(precision, recall)*100
    print("Final result:", precision*100, recall*100, f*100, os, r_val)

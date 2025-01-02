import os
import sys
import numpy as np
import torchaudio
import torch
import fairseq
from scipy.signal import find_peaks
from sklearn.metrics import f1_score
import subprocess
import tqdm
import torch.multiprocessing as mp
from absl import app, flags  

mp.set_start_method('spawn',force=True)

def main(argv):
    vad_path=argv[2] # a list of vads
    model_path=argv[3]
    output_dir=argv[4]
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    allvads={}
    max_vad_length=20 
    min_vad_length=0.2
    with open(vad_path) as buf:
        for line in buf:
            path,sid,s,e= line.rstrip().split(' ')
            vad=[float(s),float(e)]
            dur=vad[1]-vad[0]
            assert dur>0,dur
            assert os.path.isfile(path),path
            if sid not in allvads:
                allvads[sid]={}
            if path not in allvads[sid]:
                allvads[sid][path]=[]
            if len(allvads[sid][path])>0:
                prevvs,prevve=allvads[sid][path][-1][1]
                assert prevve<=vad[0],(prevvs,prevve,vad)
            if dur>max_vad_length-2:
                print('splitting long vad into 18s chunks',line)
                nb_vads=int(dur/(max_vad_length-2)+1)
                for i in range(nb_vads):
                    if vad[1]-vad[0]<0.3:
                        break
                    start=vad[0]
                    end=min(vad[0]+max_vad_length-2,vad[1])
                    allvads[sid][path].append((end-start,[start,end]))
                    vad=[end,vad[1]]
            else: 
                allvads[sid][path].append((dur,vad))
    # creating queue
    queue = mp.Queue()
    processes = [] 
    for rank in range(FLAGS.num_processes):
        p = mp.Process(target=infer, args=(rank, queue))
        p.start()
        processes.append(p)
    for key in allvads:
        queue.put((model_path,key,output_dir,max_vad_length,allvads[key]))
    for _ in range(FLAGS.num_processes):
        queue.put(None)  # sentinel value to signal subprocesses to exit
    for p in processes:
        p.join()  # wait for all subprocesses to finish

def infer(rank,queue):

    device = torch.device(f"cuda:{rank}")
    feat_sr=50 # the frame rate of your model
    batch_size=1 # does not work for more than 1
    model=None
    while True:
        tokens=[]
        x = queue.get()
        if x is None:
            break
        model_path,sid,output_dir,max_vad_length,allvads=x
        print('inference on speaker:',sid,'nb of files:',len(allvads),'on device:',device)
        output_file=os.path.join(output_dir,sid)
        if os.path.isfile(output_file):
            print(output_file,'exists')
            continue
        if model is None:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
            model=model[0].eval().float().to(device)
        for path in allvads:
            boundaries=[]
            index=0
            source,sr=torchaudio.load(path)
            assert sr==16000,(sr,path,'is not at 16khz')
            source=source.flatten() 
            vads=allvads[path]
            actual_end=float(len(source)/sr)
            while index<len(vads):
                batch,batch_padding,batch_vad=[],[],[]
                for _ in range(batch_size):
                    if index>len(vads)-1:
                        break
                    vad, padded_clip, index,padding=getitem(vads,index,max_vad_length,feat_sr,sr,source,actual_end)
                    batch.append(padded_clip)
                    batch_vad.append(vad)
                    batch_padding.append(padding)
                
                batch_padding=torch.cat(batch_padding,0).to(device)
                indexes=torch.where(batch_padding.view(-1)==0)
                indexes = [cind.cpu() for cind in indexes]
                indexes = tuple(indexes)
                
                probs=model(batch,device=device)['x'].detach().cpu().view(-1)
                probs=probs[indexes]
                probs=probs.numpy()

                peaks = get_peak_preds(probs,prominence_candidate=0.4,height_candidate=0.33,distance_candidate=1)
                # peaks to boundaries
                if batch_size>1: 
                    start_ind=0
                    length=torch.sum(1-batch_padding,dim=1).int()
                    times=[]
                    for i in range(len(batch)):
                        end_ind=start_ind+length[i]
                        t=(torch.where(peaks[start_ind:end_ind]==1)[0]/feat_sr).numpy()
                        t+=batch_vad[i][0]
                        times+=t.tolist()
                        start_ind=end_ind 
                else:
                    times=(torch.where(peaks==1)[0]/feat_sr).numpy() 
                    times+=batch_vad[0][0]
                    use_vads_as_boundaries=True
                    if use_vads_as_boundaries and len(times)>1:
                        if times[0]-batch_vad[0][0]>0.15:
                            times=np.insert(times,0,batch_vad[0][0],0)
                        if batch_vad[0][1]-times[-1]>0.15:
                            times=np.append(times,batch_vad[0][1])
                times=np.around(times,2)
                if len(times)>1:
                    prevs,preve=0,0
                    for i in range(0,len(times)-1):
                        start,end=times[i],times[i+1]     
                        #print(sid,start,end,prevs,preve,vad)
                        assert preve<=start,(start,end,sid)
                        preve=end
                        boundaries.append([start,end])
            if len(boundaries)==0:
                print('no boundaries found in',path)
                continue
            #boundaries=np.sort(np.around(np.array(boundaries),2),axis=0)
            boundaries_txt=''
            prevs,preve=0,0
            for start,end in boundaries:
                assert preve<=(start+0.001),(preve<=start,prevs,preve,start,end,sid)
                boundaries_txt+=str(start)+','+str(end)+" "
                prevs,preve=start,end
            tokens.append(path+' '+sid+' '+boundaries_txt)
        # writing output files
        with open(output_file,'w') as buf:
            buf.write('\n'.join(tokens)+'\n')
    del model
    return

# This function predicts the peaks to calculate plain F1
def get_peak_preds(preds,prominence_candidate,height_candidate,distance_candidate):
    peak_ind = find_peaks(preds,prominence=prominence_candidate,height=height_candidate,distance=distance_candidate)[0]
    peaks = torch.zeros(preds.shape)
    peaks[peak_ind] = 1
    return peaks 


def getitem(vads,index,max_vad_length,feat_sr,sr,source,actual_end):
    
    dur,vad=vads[index]
    max_wav_length=int(80+max_vad_length*sr)
    max_frames_sentence=int(np.ceil(feat_sr*max_vad_length))
    concat_vad=vad.copy()
   
    use_padding=False
    if use_padding: 
        if index==0:
            pad_start=min(1,vad[0])
        else:
            prevdur,prevvad=vads[index-1]
            prevstart,prevend=prevvad
            pad_start=min(1,(concat_vad[0]-prevend)/2)
            
         
    while True:
        index+=1
        if index>=len(vads):
            break
        n_dur,n_vad=vads[index]
        sil_dur=n_vad[0]-concat_vad[1]
        assert sil_dur>=0
        if sil_dur>1:
        #if concat_vad[1]!=n_vad[0]:
            break
        if dur+n_dur+sil_dur>max_vad_length-2:
            break
        # there can be a silence between two vads
        assert concat_vad[-1]<=n_vad[0]
        concat_vad=[concat_vad[0],n_vad[1]] 
        dur+=n_dur+sil_dur
   
    if use_padding: 
        if index>=len(vads):
            pad_end=min(1,actual_end-concat_vad[1])
        else:
            nextdur,nextvad=vads[index]
            nextstart,nextend=nextvad
            pad_end=min(1,(nextstart-concat_vad[1])/2)
            #print(concat_vad,nextvad,pad_end)
        concat_vad[0]-=pad_start
        concat_vad[1]+=pad_end
        dur=dur+pad_end+pad_start
    # make labels from bounds
    nb_frames=min(int(dur*feat_sr)+1,max_frames_sentence) 
    
    padding=torch.zeros(max_frames_sentence) 
    padding[nb_frames:]=1
    padding=padding.reshape(1,-1)

    clip=torch.clone(source[int(concat_vad[0]*sr):int(concat_vad[1]*sr)])
    padded_clip=torch.zeros(int(max_wav_length))
    padded_clip[:len(clip)]=clip
    padded_clip=padded_clip.reshape(1,-1)
    
    return concat_vad,padded_clip,index,padding

if __name__=='__main__':
    ngpu=sys.argv[1] # number of gpus
    FLAGS = flags.FLAGS
    flags.DEFINE_integer("num_processes", ngpu, "Number of subprocesses to use")
    app.run(main)
    

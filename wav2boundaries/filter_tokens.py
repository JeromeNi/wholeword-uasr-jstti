import os
import sys
import numpy as np
import subprocess
import tqdm

def main(token_file,vad_file):
    bounds={}
    vadarray={}
    bounds_in_vad={}
    min_word_size=0.1
    max_word_size=2
    with open(vad_file) as buf:
        for line in buf:
            path,sid,start,end= line.rstrip().split(' ')
            fid=os.path.basename(path).split('.')[0]
            if fid not in vadarray:
                bounds[fid]=[]
                vadarray[fid]={'sid':sid,'path':path,'start':[],'end':[]}
            vadarray[fid]['start'].append(float(start))       
            vadarray[fid]['end'].append(float(end))       
    tokens=[]
    with open(token_file) as buf:
        for line in buf:
            data = line.rstrip().split(' ')
            if len(data[0])==0:
                continue
            path,sid=data[:2]
            fid=os.path.basename(path).split('.')[0]
            if fid not in vadarray:
                #print("discarding",fid)
                continue
            
            words=data[2:]
            if ',' in words[0]:
                for word in words:
                    s,e=word.split(',')
                    start,end=float(s),float(e)
                    bounds[fid].append(float(start))
                    bounds[fid].append(float(end))
            else:
                start,end=words
                bounds[fid].append(float(start))
                bounds[fid].append(float(end))
    
    # for each vad, get all boundaries that belong to it
    c=0
    outstyle='oneline' # oneline, tde, tokens
    for fid in vadarray:
        bounds[fid]=np.unique(np.array(bounds[fid]))   
        path=vadarray[fid]['path']
        sid=vadarray[fid]['sid']
        out=path+' '+sid
        for i in range(len(vadarray[fid]['start'])):
            vs=vadarray[fid]['start'][i]
            ve=vadarray[fid]['end'][i]
            ind=np.logical_and(bounds[fid]>=vs,bounds[fid]<=ve)
            new_bounds=bounds[fid][ind]
            if len(new_bounds)==0:
                continue
                #if sid not in ['M03_O','M01_N'] and ve-vs<max_word_size:
                #    new_bounds=[vs,ve]
                #else:
                #    continue
            else:
                if new_bounds[0]-vs>min_word_size and new_bounds[0]-vs<max_word_size:
                    new_bounds=np.insert(new_bounds,0,vs)
                if ve-new_bounds[-1]>min_word_size and ve-new_bounds[-1]<max_word_size:
                    new_bounds=np.append(new_bounds,ve)
            if len(new_bounds)==0:
                continue
            for i in range(0,len(new_bounds)-1):    
                start,end=np.around((new_bounds[i],new_bounds[i+1]),2)
                if outstyle=='oneline':
                    out+=' '+str(start)+','+str(end) 
                elif outstyle=='tokens':
                    print(fid,start,end)
                elif outstyle=='tde':
                    print('Class',c)
                    print(fid,start,end)
                    print('')
                    c+=1
        if outstyle=='oneline' and ',' in out:
            print(out)     



if __name__=='__main__':
    token_file=sys.argv[1]
    vad_file=sys.argv[2]
    main(token_file,vad_file)
    

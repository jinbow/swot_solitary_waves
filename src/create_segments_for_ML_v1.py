import glob 

def process_segments_pass(fn_in, out_dir):
#    import s3fs
    import matplotlib as mpl
    mpl.use('Agg')
    import xarray as xr
    import numpy as np
    import pylab as plt
    #import tempfile

    dd = xr.open_dataset(fn_in, engine='h5netcdf')
    ddd = dd['ssha_karin_2'].data
    qual=dd['ssha_karin_2_qual'].data==0
    surface=dd['ancillary_surface_classification_flag'].data==0
    msk=qual&surface
    
    ddd=np.where(msk,ddd,np.nan)

    nperbox=65
    nlines=ddd.shape[0]
    nsegments=nlines//nperbox

    
    fig = plt.figure(figsize=(1,1))
    ax = fig.add_subplot(111)
    
    
    file_name=fn_in.split('/')[-1][:-3]
#    sf_s3_out=s3fs.S3FileSystem(anon=False,
#                               key='',
#                               secret='')
    for i in range(nsegments):
        msk_tmp=msk[i*nperbox:(i+1)*nperbox,3:68]
        #print(i,msk_tmp.sum())
        if msk_tmp.sum()>nperbox*nperbox/2:
            file_out=file_name+"_seg_%04i.png"%i
            #print('save figure to '+out_dir+"/"+file_out)
            d0=ddd[i*nperbox:i*nperbox+nperbox,3:68]
            d0-=np.nanmean(d0,0,keepdims=True)
            d0-=np.nanmean(d0,1,keepdims=True)
            plt.axis('off')
            plt.subplots_adjust(left=0,right=1,bottom=0,top=1)
            ax.imshow(d0,cmap=plt.cm.binary_r,vmin=-0.2,vmax=0.2)
            plt.savefig(out_dir+"/"+file_out, dpi=65,bbox_inches='tight', pad_inches=0)
            ax.clear()  
    return 

folder_in='/mnt/flow/swot/KaRIn/SWOT_L2_LR_SSH_1.1/'
fns=glob.glob(folder_in+"*Expert*.nc")
out_dir='/mnt/flow/swot/KaRIn/segments4ML/KaRIn_SSH_1.1_65x65/'

import time
t0=time.time()
for i,fn in enumerate(fns):
    print(i)
    process_segments_pass(fn,out_dir)

print(time.time()-t0)

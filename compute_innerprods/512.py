import time
import numpy as np
import h5py

# each image is 1MB (in 8bit precision) so blocks of 1000*k take exactly k GB

def mean_image(ds, blocksize=1000):
    """Just compute the mean of a dataset along the first dimension"""
    N = ds.shape[0]
    s = np.zeros([1]+list(ds.shape[1:]), dtype=np.int64)
    nblocks = N//blocksize+1
    for (i,off) in enumerate(range(0, N, blocksize)):
        print(f"Block {i+1} of {nblocks}")
        end = min(N, off+blocksize)
        s += np.sum(ds[off:end,...], axis=0, keepdims=True, dtype=np.int64)
    return s.astype(np.float64)/N

def gram(dsleft, dsright, blocksize=1000, subim=None, g=None):
    """
    Given a possibly large dataset, compute the inner products efficiently.
    
    If given, subtract the image 'subim' from each vector before taking inner
    product.
    """
    Nleft = dsleft.shape[0]
    Nright = dsright.shape[0]
    n_blocks = np.ceil(Nleft/blocksize)*np.ceil(Nright/blocksize)
    n_done = 0
    perc = 1
    print("  0 %")
    if g is None:
        g = np.zeros((Nleft, Nright), dtype=np.float64)
    for offleft in range(0, Nleft, blocksize):
#        print("L")
        endleft = min(Nleft, offleft+blocksize)
        xl = np.asarray(dsleft[offleft:endleft,...], dtype=np.float64)
        if subim is not None:
            xl -= subim
        xl = xl.reshape((xl.shape[0],-1)) # vectorize
        for offright in range(0, Nright, blocksize):
#            print("R")
            endright = min(Nright, offright+blocksize)
            xr = np.asarray(dsright[offright:endright,...], dtype=np.float64)
            if subim is not None:
                xr -= subim
            xr = xr.reshape((xr.shape[0],-1)) # vectorize
            g[offleft:endleft, offright:endright] = xl.dot(xr.T)
            n_done += 1
            if 100*n_done/n_blocks >= perc:
                print("  "+str(np.round(100*n_done/n_blocks, 2))+" %")
                perc += 1
    return g

time_file = open("/raid/scratch/devanshu/innerprod_512_times.txt", "w")

#skip = 100
print("loading data")
with h5py.File('/raid/ChestXRay14/chestxray14_512.h5', 'r') as f:
    X_train = f['/images/train'] # [::skip]
    X_val = f['/images/val'] # [::skip]
    X_test = f['/images/test'] # [::skip]

    print("Computing mean image")
    time_0 = time.time()
    meantrain = mean_image(X_train)
    time_file.write("mean_train: "+str(np.round(time.time()-time_0, 2))+" s\n")
    with h5py.File('/raid/scratch/devanshu/innerprods_512.h5', 'w') as f:
        print("Computing train_train")
        time_0 = time.time()
        tt = f.create_dataset('train_train', shape=(X_train.shape[0], X_train.shape[0]), dtype=np.float64)
        gram(X_train, X_train, subim=meantrain, g=tt)
        time_file.write("train_train: "+str(np.round(time.time()-time_0, 2))+" s\n")
        print("Computing train_val")
        time_0 = time.time()
        tt = f.create_dataset('train_val', shape=(X_train.shape[0], X_val.shape[0]), dtype=np.float64)
        gram(X_train, X_val, subim=meantrain, g=tt)
        time_file.write("train_val: "+str(np.round(time.time()-time_0, 2))+" s\n")
        print("Computing train_test")
        time_0 = time.time()
        tt = f.create_dataset('train_test', shape=(X_train.shape[0], X_test.shape[0]), dtype=np.float64)
        gram(X_train, X_test, subim=meantrain, g=tt)
        time_file.write("train_time: "+str(np.round(time.time()-time_0, 2))+" s\n")
        print("Computing val_val")
        time_0 = time.time()
        tt = f.create_dataset('val_val', shape=(X_val.shape[0], X_val.shape[0]), dtype=np.float64)
        gram(X_val, X_val, subim=meantrain, g=tt)
        time_file.write("val_val: "+str(np.round(time.time()-time_0, 2))+" s\n")
        print("Computing val_test")
        time_0 = time.time()
        tt = f.create_dataset('val_test', shape=(X_val.shape[0], X_test.shape[0]), dtype=np.float64)
        gram(X_val, X_test, subim=meantrain, g=tt)
        time_file.write("val_test: "+str(np.round(time.time()-time_0, 2))+" s\n")
        print("Computing test_test")
        time_0 = time.time()
        tt = f.create_dataset('test_test', shape=(X_test.shape[0], X_test.shape[0]), dtype=np.float64)
        gram(X_test, X_test, subim=meantrain, g=tt)
        time_file.write("test_test: "+str(np.round(time.time()-time_0, 2))+" s\n")

time_file.close()

print("done!")

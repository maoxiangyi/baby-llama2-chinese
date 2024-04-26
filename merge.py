import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    import glob
    data_path_list = glob.glob('/mnt/pfs/data_team/maoxiangyi/data/*')
    data_path_list = sorted(data_path_list)
    data_lst=[]
    for data_path in tqdm(data_path_list):
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())

    result = open('./data/pretrain_data.bin','ab')
    for data_path in tqdm(data_path_list):
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            result.write(data.tobytes())
    result.flush()
    result.close()

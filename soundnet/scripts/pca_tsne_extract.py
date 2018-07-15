import numpy as np
from os import walk
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pdb
from tqdm import tqdm

def go_folder(path):
    print('\n\nData Loading ...')
    f_order = []
    ori_data = []
    for (dirpath, dirnames, filenames) in walk(path):
        for a_name in filenames:
            if len(a_name.split('.')) == 2:
                ori_data.append( np.load(dirpath+'/'+ a_name) )
                f_order.append(dirpath+ '/' + a_name.split('.')[0])

    return f_order, np.array(ori_data)


def trans_pca(data, dim):
    print('\n\nStart PCA computing ...\n\n')
    feat = PCA(n_components=dim, svd_solver='randomized').fit_transform(data)

    return feat

def trans_tsne(data, dim):
    print('\n\nStart TSNE computing ...\n\n')
    feat = TSNE(n_components=dim, method='exact').fit_transform(data)
    return feat

def save_feat(f_list, feat, feat_name , dim):
    print('\n\n{0} feats saving ..\n\n.'.format(feat_name))

    for idx in range(len(f_list)):
        f_path = f_list[idx] +'.' + feat_name+'.npy'
        #np.save(open(f_path,'wb'), feat[idx].reshape(-1,dim))
        np.save(open(f_path,'wb'), feat[idx].reshape(-1))
        print('{0} Saved !'.format(f_path))

if __name__== '__main__':

    path = '../../dataset/mp3_soundnet_feat'
    encode_dim = 150

    f_list, data = go_folder(path)

    pca_data = trans_pca(data, encode_dim)
    save_feat(f_list, pca_data, 'pca', encode_dim)
#    tsne_data = trans_tsne(data, encode_dim)
#    save_feat(f_list, tsne_data, 'tsne', encode_dim)



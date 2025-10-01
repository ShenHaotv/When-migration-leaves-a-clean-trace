import numpy as np
import gc
from copy import deepcopy
from sklearn.model_selection import KFold
from scipy.linalg import expm

def run_cv_epoch(
    st_digraph,
    epoch,
    lamb_grid,
    n_folds=None,
    factr=1e10,
    random_state_1=500,
    random_state_2=1000,
    outer_verbose=True,
    inner_verbose=False,
    node_train_idxs_1=None,
    node_train_idxs_2=None,
    ):
    """Run cross-validation."""
    o1=st_digraph.C[epoch].shape[0]
    o2=st_digraph.C[epoch+1].shape[0]

    # default is None i.e., leave-one-out CV
    if n_folds is None:
       n_folds = 10
        
    # setup cv indicies
    if node_train_idxs_1 is None:
       node_train_idxs_1=setup_k_fold_cv(o=o1,n_splits=n_folds,random_state=random_state_1)
       
    if node_train_idxs_2 is None:
       node_train_idxs_2=setup_k_fold_cv(o=o2,n_splits=n_folds,random_state=random_state_2)

    # CV error
    n_lamb = len(lamb_grid)

    errs=np.zeros((n_folds,n_lamb))
    
    for fold in range(n_folds):      
        if outer_verbose:
            print("\n fold=", fold+1)
        
        node_train_idx_1=node_train_idxs_1[fold].astype(int)
        node_train_idx_2=node_train_idxs_2[fold].astype(int)
             
        for i, lamb in enumerate(lamb_grid):                                    #Formal cv
            if outer_verbose:
               print("\riteration lambda={}/{}".format(i + 1, n_lamb),end="",)
               # fit on train set
            lamb= float(lamb)
            (errs[fold,i])=error_epoch(st_digraph,
                                       epoch,
                                       node_train_idx_1,
                                       node_train_idx_2,
                                       lamb=lamb,                                    
                                       factr=factr,
                                       verbose=inner_verbose)           

        sum_of_products= sum(len(idx1) * len(idx2) for idx1, idx2 in zip(node_train_idxs_1, node_train_idxs_2))
  
    num=fold*(o1*o2)-sum_of_products
    cv_err=np.sum(errs,axis=0)/(num)
     
    return (cv_err,node_train_idxs_1,node_train_idxs_2)

def setup_k_fold_cv(o,n_splits, random_state):
    """Setup cross-validation indicies.0

    Args:
        o:number of observed nodes
        n_splits (:obj:`int`): number of CV folds
        random_state (:obj:`int`): random seed

    Returns:
        node_train_idxs: the indexes of indexes of rows of h that are selected for
        training
    """     
    kf = KFold(
        n_splits=n_splits, random_state=random_state, shuffle=True
    )  # k-fold cv object
    
    node_train_idxs = [train_index for train_index,_ in kf.split(np.arange(o))]
    return  node_train_idxs


def copy_spatial_digraph_epoch(st_digraph, epoch,node_train_idx_1,node_train_idx_2):
    """Copy SpatialDiGraph object

    Args:
        sp_digraph (:obj:`SpatialDiGraph`): SpatialDiGraph class
        node_idx (:obj:`numpy.ndarray`): indexes of rows of h that are selected

    Returns:
        sp_digraph_copy (:obj:`SpatialGraph`): SpatialDiGraph class
    """
    st_digraph_copy=deepcopy(st_digraph)
    d=st_digraph_copy.d
    A1=np.zeros((d,d))
    A1[np.ix_(node_train_idx_1, node_train_idx_1)] = st_digraph_copy.C[epoch][np.ix_(node_train_idx_1, node_train_idx_1)]
    A2=np.zeros((d,d))
    A2[np.ix_(node_train_idx_2, node_train_idx_2)] = st_digraph_copy.C[epoch][np.ix_(node_train_idx_2, node_train_idx_2)]
    st_digraph_copy.C[epoch]=A1
    st_digraph_copy.C[epoch+1]=A2

    return st_digraph_copy


def error_epoch(st_digraph,
                epoch,
                node_train_idx_1,
                node_train_idx_2,
                lamb,
                factr,
                verbose,
                softplus_inv_m_init=None,
                ):
  
    i=epoch
    d=st_digraph.d
    st_digraph_train=copy_spatial_digraph_epoch(st_digraph,i, node_train_idx_1,node_train_idx_2)    
    st_digraph_train.fit_epoch(epoch=i,
                               lamb=lamb,
                               factr=factr,
                               lb=-np.Inf,
                               ub=np.Inf,
                               verbose=verbose,
                               softplus_inv_m_init=softplus_inv_m_init)
    
    A=st_digraph.data_estimated[i+1][st_digraph.num_finite_epochs]
    B=st_digraph.data[i][st_digraph.num_finite_epochs]-st_digraph.finite_epoch_lengths[i]*np.ones((d,d))
    C1=st_digraph.C[i]
    C2=st_digraph.C[i+1]
    Lt=st_digraph_train.L[i]*st_digraph.finite_epoch_lengths[i]
    Lt_array=Lt.toarray()
    res=(expm(-Lt_array)@A-B)/B
    res_raw=C1@res@C2.T
    C1_train=C1[node_train_idx_1,:]
    C2_train=C2[node_train_idx_2,:]
    res_train=C1_train@res@C2_train.T
    err=np.linalg.norm(res_raw, ord='fro')**2 - np.linalg.norm(res_train, ord='fro')**2
        
    gc.collect()
    return err


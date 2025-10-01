from __future__ import absolute_import, division, print_function
import networkx as nx
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from loss import loss_wrapper
import scipy.sparse as sp
from scipy.linalg import expm


def getlaplacian(m,adj):
    A=adj.copy()
    A.data=m                                                                   #Compute laplacian
    A_rowsum=np.array(A.sum(axis=1)).reshape(-1)
    D=sp.diags(A_rowsum).tocsr()
    L=D-A
    return (L)

class SpatialTemporalDiGraph(nx.DiGraph):
    def __init__(self, time_points, average_pairwise_total_branch_length,  node_pos, edges, C=None):
        """Represents the spatial network which the data is defined on and
        stores relevant matrices / performs linear algebra routines needed for
        the model and optimization. Inherits from the networkx Graph object.

        Args:
            time_points (:obj:'1 D numpy ayyar'): sample time points, stored as [t0,t1,...,tn]
            average_pairwise_total_branch_lengths (:obj:`numpy.ndarrays`): 
                average sample branch length matrices among demes from different epochs
            node_pos (:obj:`numpy.ndarrays`): spatial positions of nodes 
            edges (:obj:`list of numpy.ndarrays`):  edge arrays (in undirected format) 
            C (:obj:'List'): each entry represents a k*d assignment matrix of observed demes
        """
        # Check inputs
        assert len(time_points.shape) == 1, "epoch must be a 1D numpy array"
        assert time_points[0]==0, "start from time point 0"
    
        # Check we have at least 2 epoch points (so len(epoch)-1 > 0)
        assert len(time_points) >= 2, "Need at least 2 epoch time points"
    
        # Check matrix is square
        assert len(average_pairwise_total_branch_length.shape) == 2, "Branch length matrix must be 2D"
        assert average_pairwise_total_branch_length.shape[0] ==average_pairwise_total_branch_length.shape[1],"Branch length matrix must be square"
    
        # Get matrix size
        matrix_size = average_pairwise_total_branch_length.shape[0]
        self.num_epochs = len(time_points)
        self.num_finite_epochs=self.num_epochs-1
        self.node_pos=node_pos
        self.d=node_pos.shape[0]
        
        if C is None:
           self.C=[np.identity(self.d) for _ in range(self.num_epochs)]
        else:
            self.C=C
    
        # Check matrix size is multiple of epoch intervals
        assert matrix_size==self.num_epochs*self.d, f"Matrix size {matrix_size} must be  equal to number of epoch intervals times the deme number"
      
        # Inherits from networkx Graph object
        super(SpatialTemporalDiGraph, self).__init__()
        self._init_digraph(node_pos, edges)  # init graph

        #Sample time points
        self.time_points=time_points
        self.finite_epoch_lengths=[]
        for i in range(self.num_finite_epochs):
            self.finite_epoch_lengths.append(self.time_points[i+1]-self.time_points[i])
        
        # Data
        self.raw_data=average_pairwise_total_branch_length
        self.data=[[np.zeros((self.d, self.d)) for _ in range(self.num_epochs)] for _ in range(self.num_epochs)]
        for i in range(self.num_epochs):
            for j in range(self.num_epochs):
                self.data[i][j]=self.raw_data[i*self.d:(i+1)*self.d, j*self.d:(j+1)*self.d]
        
        self.data_estimated=self.data
                
        #Get the adjacency matrix
        self.adjacency_sparse=nx.adjacency_matrix(self)
        self.adjacency=self.adjacency_sparse.toarray()
       
        m0_template = np.ones(len(self.adjacency_sparse.data))
        L0_template = getlaplacian(m0_template, self.adjacency_sparse)
       
        # Initialize lists for each epoch
        self.m0 = []   # List of migration rate vectors
        self.L0 = []   # List of Laplacian matrices
       
        for i in range(self.num_finite_epochs):         
            self.m0.append(m0_template)
            self.L0.append(L0_template)
        
        self.m=self.m0.copy()
        self.L=self.L0
        self.train_loss=np.zeros(self.num_epochs-1)
        self.deg=np.array(list(self.degree()))[:,1]



    def _init_digraph(self, node_pos, edges):
        """Initialize the digraph and related digraph objects

        Args:
            node_pos (:obj:`numpy.ndarray`):  spatial positions of nodes
            edges (:obj:`numpy.ndarray`): edge array
        """
        self.add_nodes_from(np.arange(node_pos.shape[0]))
       
        # Create a list to hold both original and reversed edges
        all_edges = []
        for edge in edges:
            all_edges.append(tuple(edge))
            all_edges.append((edge[1], edge[0]))  # Add the reversed edge
       
        # Add edges to the graph
        self.add_edges_from(all_edges)

        # add spatial coordinates to node attributes
        for i in range(len(self)):
            self.nodes[i]["idx"] = i
            self.nodes[i]["pos"] = node_pos[i, :]
    
 

    # ------------------------- Optimizers -------------------------


    def fit_epoch(
        self,    
        epoch,
        lamb,
        maxls=50,
        factr=1e7,
        m=10,
        lb=-np.Inf,
        ub=np.Inf,
        maxiter=15000,
        verbose=True,
        softplus_inv_m_init=None,
        ):
           
        """Estimates the edge weights of the full model holding the residual
        variance fixed using a quasi-newton algorithm, specifically L-BFGS.

        Args:
            epoch (:'int'): The index of epoch to fit
            lamb (:'float'): Hyperparameter for cross validation
            factr (:self:`float`): tolerance for convergence
            m   #The degree of each node
               self.deg=np.array(list(self.degree()))[:,1]axls (:self:`int`): maximum number of line search steps
            m (:self:`int`): the maximum number of variable metric corrections
            lb (:self:`int`): lower bound of parameters
            ub (:self:`int`): upper bound of parameters
            maxiter (:self:`int`): maximum number of iterations to run L-BFGS
            verbose (:self:`Bool`): boolean to print summary of results
            softplus_inv_m_init(:self:`float`):initial value of softplus inverse edge weights"""
    
        if  softplus_inv_m_init is None:
            softplus_inv_m_init=np.log(np.exp(self.m0[0])-1)
        
        # check inputs
        assert type(epoch) == int, "epoch must be int"
        assert type(factr) == float, "factr must be float"
        assert maxls > 0, "maxls must be at least 1"
        assert type(maxls) == int, "maxls must be int"
        assert type(m) == int, "m must be int"
        assert type(lb) == float, "lb must be float"
        assert type(ub) == float, "ub must be float"
        assert lb < ub, "lb must be less than ub"
        assert type(maxiter) == int, "maxiter must be int"
        assert maxiter > 0, "maxiter be at least 1"
   
        # run l-bfgs
        
        i=epoch
        x0 =softplus_inv_m_init
        X=self.data_estimated[i+1][self.num_finite_epochs]
        
        assert not np.any(X == 0), "Matrix contains zero elements!"

        Y=self.data[i][self.num_finite_epochs]-self.finite_epoch_lengths[i]*np.ones((self.d,self.d))
        C1=self.C[i]
        C2=self.C[i+1]
        res = fmin_l_bfgs_b(
              func=loss_wrapper,
              x0=x0,
              args=[self.adjacency,X,Y,C1,C2,self.d, self.deg, lamb],
              factr=factr,              
              m=m,
              maxls=maxls,
              maxiter=maxiter,
              approx_grad=False,
              bounds=[(lb, ub) for _ in range(x0.shape[0])],)
        
        if maxiter >= 100:
           assert res[2]["warnflag"] == 0, "did not converge"

        mt=np.log(1+np.exp(res[0]))
        self.m[i]=mt/self.finite_epoch_lengths[i]
        Lt=getlaplacian(mt, self.adjacency_sparse)
        self.L[i]=Lt/self.finite_epoch_lengths[i]
        self.train_loss[i]=res[1]
        Lt_array=Lt.toarray()
        self.data_estimated[i][self.num_finite_epochs]=expm(-Lt_array)@self.data_estimated[i+1][self.num_finite_epochs]+self.finite_epoch_lengths[i]*np.ones((self.d,self.d))
 
        return()

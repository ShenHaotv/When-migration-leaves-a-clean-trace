import numpy as np
from scipy.sparse import csr_matrix
from sim import Sim
from spatial_temporal_digraph import getlaplacian, SpatialTemporalDiGraph

def run_sim_migration(time_points,
                      topology_list,
                      ne_list,
                      target_n_replicates,
                      n_print):
     
    n_samples_per_node=10
    N_rows=11
    N_columns=9
    
    Simulation = Sim(time_points=time_points,
                     n_rows=N_rows,
                     n_columns=N_columns,
                     n_samples_per_node=n_samples_per_node)
    
    Simulation.setup_topologies(topology_list)
    Simulation.setup_populations(ne_list)

    average_pairwise_total_branch_length= Simulation.simulate_average_pairwise_total_branch_length(sequence_length=1,
                                                                                                   target_n_replicates=target_n_replicates,
                                                                                                   n_print=n_print)
 
    node_pos= Simulation.node_pos.copy()
    edges = Simulation.edges.copy()
    st_digraph = Simulation.st_digraph.copy()
    d=Simulation.d
    size=d*Simulation.num_epochs
    fill_in=np.ones((size,size))
       
    ground_truth=SpatialTemporalDiGraph(time_points,fill_in, node_pos, edges)
    ground_truth.m=[]
    ground_truth.L=[]
    for i in range(Simulation.num_epochs):
        M=np.zeros((d,d))
        for u,v in Simulation.st_digraph.edges():
            M[u,v]=Simulation.st_digraph[u][v]['weight_list'][i]
       
        M = csr_matrix(M)
        m=M.data
        L=getlaplacian(m,Simulation.st_digraph.adjacency_sparse)
        ground_truth.m.append(m)
        ground_truth.L.append(L)
    
    st_digraph = SpatialTemporalDiGraph(time_points, average_pairwise_total_branch_length, node_pos, edges)
 
    return(ground_truth,st_digraph)
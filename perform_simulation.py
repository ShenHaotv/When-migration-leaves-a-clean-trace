from sim_main import run_sim_migration
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from visualization import Vis
import copy
from cross_validation import run_cv_epoch

"""Fitting Function"""
def fit_epoch_opt(st_digraph,
                  epoch,
                  lamb_grid=np.geomspace(1e-6, 1e0,13)[::-1],
                  n_folds=10,
                  factr=1e10,
                  factr_fine=1e7,
                  random_state_1=100,
                  random_state_2=500,
                  outer_verbose=True,
                  inner_verbose=False,):
    
   cv,node_train_idxs_1,node_train_idxs_2=run_cv_epoch(st_digraph,
                                                       epoch,
                                                       lamb_grid,
                                                       n_folds,
                                                       factr,
                                                       random_state_1,
                                                       random_state_2,
                                                       outer_verbose,
                                                       inner_verbose,)
   
   if np.argmin(cv)==0:
      lamb_grid_fine=np.geomspace(lamb_grid[0],lamb_grid[1],7)[::-1]

   elif np.argmin(cv)==12:
        lamb_grid_fine=np.geomspace(lamb_grid[11],lamb_grid[12], 7)[::-1]
        
   else:
       lamb_grid_fine=np.geomspace(lamb_grid[np.argmin(cv)-1],lamb_grid[np.argmin(cv)+1], 7)[::-1]

   cv_fine,node_train_idxs_fine_1,node_train_idxs_fine_2=run_cv_epoch(st_digraph,
                                                                      epoch,
                                                                      lamb_grid_fine,
                                                                      n_folds,
                                                                      factr,
                                                                      random_state_1,
                                                                      random_state_2,
                                                                      outer_verbose,
                                                                      inner_verbose,
                                                                      node_train_idxs_1=node_train_idxs_1,
                                                                      node_train_idxs_2=node_train_idxs_2)

   lamb_opt=lamb_grid_fine[np.argmin(cv_fine)]
   lamb_opt=float("{:.3g}".format(lamb_opt))
   print(lamb_opt)
   
   st_digraph.fit_epoch(epoch,
                        lamb=lamb_opt,
                        factr=factr_fine)

   return (st_digraph,lamb_grid_fine,cv,cv_fine)

"""Perform Simulation, take the following topology sequence for example"""
t0=1
time_points=np.array([0, t0,2*t0])
topology_list=['large_scale_spatially_converging_lineages','small_scale_patterns','large_scale_directionally_migrating_lineages']

n_print=500
target_n_replicate=10000
ne_epoch_0 = 10**(np.random.uniform(-1, 1, 99))
ne_epoch_1 = 10**(np.random.uniform(-1, 1, 99))
ne_epoch_2 = 10**(np.random.uniform(-1, 1, 99))
ne_list=[ne_epoch_0,ne_epoch_1,ne_epoch_2]
ground_truth,st_digraph=run_sim_migration(time_points,
                                          topology_list,
                                          ne_list,
                                          target_n_replicate,
                                          n_print)
   
ground_truth.ne_list=ne_list 
st_digraph.ne_list=ne_list

"""Fitting"""
epoch_list=[1,0]
st_digraph.lamb_grid_list=[[],[]]
st_digraph.lamb_grid_fine_list=[[],[]]
st_digraph.cv_list=[[],[]]
st_digraph.cv_fine_list=[[],[]]

for k in range(2):
    epoch=epoch_list[k]
    st_digraph,lamb_grid_fine,cv,cv_fine=fit_epoch_opt(st_digraph,
                                                       epoch,
                                                       lamb_grid=np.geomspace(1e-6, 1e0,13)[::-1],
                                                       n_folds=10,
                                                       factr=1e10,
                                                       factr_fine=1e7,
                                                       random_state_1=100,
                                                       random_state_2=200,
                                                       outer_verbose=True,
                                                       inner_verbose=False,)
        
    st_digraph.lamb_grid_list[epoch]=np.geomspace(1e-6, 1e0,13)[::-1]
    st_digraph.lamb_grid_fine_list[epoch]=lamb_grid_fine
    st_digraph.cv_list[epoch]=cv
    st_digraph.cv_fine_list[epoch]=cv_fine
        

"""Visualization of ground truth"""

sp_digraph_g = copy.deepcopy(ground_truth)
projection = ccrs.Mercator()
fig, axs = plt.subplots(1,3, figsize=(9,3), dpi=300, subplot_kw={'projection': projection})
 
for j in range(3):
    A = sp_digraph_g.L[j].toarray()
    np.fill_diagonal(A, 0)
    M = -A
    sp_digraph_g.M = M
    sp_digraph_g.ne = sp_digraph_g.ne_list[j]
    v = Vis(axs[0], sp_digraph_g, projection=projection, edge_width=0.5,
         edge_alpha=1, edge_zorder=100, cbar_font_size=8, cbar_ticklabelsize=8, cbar_width="25%",
         cbar_bbox_to_anchor=(0.4, -0.15, 1, 1), mutation_scale=6)
    v.draw_migration_rates(axs[j], mode='Full', draw_map=None, set_title=None)
    
plt.show()

"""Visualization of fitted results"""
fig, axs = plt.subplots(1,2, figsize=(6,3), dpi=300, subplot_kw={'projection': projection})
sp_digraph_s = copy.deepcopy(st_digraph)
abs_max_full_list=[0.9, 1.5, 0.8]                                                                        

for j in range(2):
    A = sp_digraph_s.L[j].toarray()
    np.fill_diagonal(A, 0)
    M = -A
    sp_digraph_g.M = M
    sp_digraph_g.ne = sp_digraph_g.ne_list[j]
    v = Vis(axs[0], sp_digraph_g, projection=projection, edge_width=0.5,
        edge_alpha=1, edge_zorder=100, cbar_font_size=8, cbar_ticklabelsize=8, cbar_width="25%",
        cbar_bbox_to_anchor=(0.4, -0.15, 1, 1),mutation_scale=6,abs_max_full=abs_max_full_list[j])
    v.draw_migration_rates(axs[j], mode='Full', draw_map=None, set_title=None)
   

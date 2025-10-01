import networkx as nx
import numpy as np
import msprime


class Sim(object):
     
      def __init__(self,
                   time_points,
                   n_rows=11,
                   n_columns=9,
                   n_samples_per_node=10,):
          
        """
        Initializes a new Simulation instance with specific lattice dimensions and sampling parameters.

        Parameters:
           time_points (:obj:'1 D numpy ayyar'): sample time points, stored as [t0,t1,...,tn]
           n_rows (int): Number of rows in the lattice, determining the vertical size of the grid.
           n_columns (int): Number of columns in the lattice, determining the horizontal size of the grid.
           n_samples_per_node (int): Total number of samples per node, specifying how many individuals are simulated in each node.
          
        Notes:
         - This method sets up the basic properties of the simulation object, including the structure of the spatial grid (lattice) and parameters for population sampling.
         
         """
        assert len(time_points.shape) == 1, "epoch must be a 1D numpy array"
        assert time_points[0]==0, "start from time point 0"
        
        self.d=n_rows*n_columns
        self.time_points=time_points
        self.num_epochs = len(time_points)
        self.num_finite_epochs=self.num_epochs-1
        self.finite_epoch_lengths=[]
        for i in range(self.num_finite_epochs):
            self.finite_epoch_lengths.append(self.time_points[i+1]-self.time_points[i+1])
        self.n_rows=n_rows
        self.n_columns=n_columns
        self.n_samples_per_node=n_samples_per_node
        
        #set up a triangular grid network
        graph = nx.generators.lattice.triangular_lattice_graph(
        self.n_rows - 1, 2 * self.n_columns - 2, with_positions=True)
   
        self.st_digraph=nx.DiGraph(graph)
        
        for node in self.st_digraph.nodes():
            self.st_digraph.nodes[node]['sample_size_list'] = [None] * self.num_epochs
            
        for u, v in self.st_digraph.edges():
            self.st_digraph[u][v]['weight_list'] = [None] * self.num_epochs
            
    
     
      def setup_each_epoch(self,
                          epoch_index,
                          m_base=0.1,
                          m_low=0.3,
                          m_high=3,
                          m_topo=3,
                          boundary=None,
                          directional=None,
                          converging=None,
                          diverging=None,
                          circle=None,
                          ):  

          """
          Sets up a directed graph (digraph) for simulating migration patterns in a spatial population model.

          Parameters:
             epoch_index (int): the index of the epoch
             m_base (float): Base migration rate.
             m_low (float): Multiplier for lower migration rate areas.
             m_high (float): Multiplier for higher migration rate areas. 
             m_topo (float): Mulyiplier for topological patterns
             boundary (list): Specifies regions with different migration rates.
             directional (list): Specifies directionally migrating lineages.
             converging (list): Specifies areas acting as zone of spatially converging lineages.
             diverging (list): Specifies areas acting as zone of spatially converging lineages.
             circle (list): Specifies nodes forming zone of cyclic rotating lineages.
           
           """       
          
          # Set default migration weights and modify them based on boundary conditions
          for (u, v) in self.st_digraph.edges():
              self.st_digraph[u][v]['weight_list'][epoch_index] =0.1
              if boundary!=None:
                 a=max(boundary)
                 b=min(boundary)
                 if u[1]>a or v[1]>a:
                    self.st_digraph[u][v]['weight_list'][epoch_index]*=m_low
                 if u[1]<b or v[1]<b:
                    self.st_digraph[u][v]['weight_list'][epoch_index]*=m_high
     
          # Apply high migration rates to directional flows
          if directional!=None:
             for (startpoint,length,direction) in directional:
                 x=startpoint                                                  #Start point of the directional pattern
                 l=length                                                      #length of the directional pattern 
                 if direction=='E':                                            #East direction                        
                    for i in range(x[0],x[0]+l) :                                                                                                                   
                        self.st_digraph[(i,x[1])][(i+1,x[1])]['weight_list'][epoch_index]*=m_topo 
                 elif direction=='W':                                          #West direction                                                    
                      for i in range(x[0]-l,x[0]) :                                                                                                                   
                          self.st_digraph[(i+1,x[1])][(i,x[1])]['weight_list'][epoch_index]*=m_topo      
                          
          # Set up migration sinks                     
          if converging!=None:
             for (si,r) in converging:                                    #center and radius         
                 a={si}
                 b={si}
                 for i in range(r):
                     c=set()
                     for x in a:
                         for y in list(self.st_digraph.neighbors(x)):
                             if y not in b:
                                c.add(y)
                                self.st_digraph[y][x]['weight_list'][epoch_index]*=m_topo
                     b=a.union(c)
                     a=c
            
          # Set up migration sources
          if diverging!=None:
             for (so,r) in diverging:
                 a={so} 
                 b={so}
                 for i in range(r):
                     c=set()
                     for x in a:
                         for y in list(self.st_digraph.neighbors(x)):
                             if y not in b:
                                c.add(y)
                                self.st_digraph[x][y]['weight_list'][epoch_index]*=m_topo
                     b=a.union(c)
                     a=c
                     
          # Define cyclic migration routes    
          if circle!=None:
             for ci in circle:
                 for i in range(len(ci)):
                     if i<len(ci)-1:
                        a=ci[i]
                        b=ci[i+1]
                        self.st_digraph[a][b]['weight_list'][epoch_index]*=m_topo
                     else:
                         a=ci[i]
                         b=ci[0]
                         self.st_digraph[a][b]['weight_list'][epoch_index]*=m_topo
        
          # Set node sample sizes based on the sampling model
          for x in self.st_digraph.nodes():
              self.st_digraph.nodes[x]['sample_size_list'][epoch_index]=self.n_samples_per_node     #The number of individuals sampled in each deme
            
   
          return ()
      
      def set_up_topology_each_epoch(self,
                                     topology,
                                     epoch_inex):
          N_rows=11
          N_columns=9
          
          if topology=='small_scale_patterns':
             boundary=[5]
          else:
              boundary=None
                   
          if topology=='large_scale_directionally_migrating_lineages':
               directional=[]
               for j in range(N_rows):
                   directional.append(((0, j), 8, 'E'))
                               
          elif topology=='large_scale_converging_directionally_migrating_lineages':
               directional=[]
               for j in range(N_rows):
                   directional.append(((0, j), 4, 'E'))
                   directional.append(((8, j), 4, 'W'))
          
          elif topology=='small_scale_patterns':
               directional=[]
               directional.append(((4, 1), 4, 'E'))
               directional.append(((4, 1), 4, 'W'))
               directional.append(((0, 9), 4, 'E'))
               directional.append(((8, 9), 4, 'W'))
               
          elif topology.startswith("start_"):
               directional=[]
               k = int(topology.split("_")[1])
               for j in range(N_rows):
                   directional.append(((k, j), 1, 'E'))
                   directional.append(((8-k, j), 1, 'W'))
          else:
              directional=None
                             
          if topology=='large_scale_spatially_converging_lineages':
             converging=[((4,5),3)]
          elif topology=='small_scale_patterns':
               converging=[((2,3),1)]
          else:
              converging=None
              
          if topology=='large_scale_spatially_diverging_lineages':
             diverging=[((4,5),3)]
          elif topology=='small_scale_patterns':
               diverging=[((6,3),1)]
          else:
              diverging=None

          if topology=='small_scale_patterns':
             circle=[((2,6),(3,6),(3,7),(3,8),(2,8),(1,7)),((5,6),(6,6),(7,6),(6,7),(6,8),(5,7))] 
          else:
              circle=None
              
          m_topo=10
               
          m_base=0.1
          m_low=0.3
          m_high=3
            
          self.setup_each_epoch(epoch_inex,
                                m_base=m_base,
                                m_low=m_low,
                                m_high=m_high,
                                m_topo=m_topo,
                                boundary=boundary,
                                directional=directional,
                                converging=converging,
                                diverging=diverging,
                                circle=circle) 
          
          return ()
      
      #set up all topologies and relabel the nodes and set up node_pos and edges
      def setup_topologies(self,topology_list):
          """
          set up the graph topology of migration rates in each epoch
          
          topology(:obj:'list'): each entry is a string, recording the topology in that epoch"""
     
          for i in range(self.num_epochs):
              topology=topology_list[i]
              self.set_up_topology_each_epoch(topology,i)
          
          
          self.st_digraph=nx.convert_node_labels_to_integers(self.st_digraph)
               
          graph = self.st_digraph.to_undirected()
          pos_dict = nx.get_node_attributes(self.st_digraph, 'pos')
    
          node_pos= np.array(list(pos_dict.values()))
          edges = np.array(graph.edges)

          self.node_pos=node_pos
          self.edges=edges
          self.st_digraph.adjacency_sparse=nx.adjacency_matrix(self.st_digraph)
          self.st_digraph.adjacency=self.st_digraph.adjacency_sparse.toarray()
    
          return ()
         
      def setup_populations(self,
                             ne_list=None):  
           
          """
          Setup the populations in msprime.

          Parameters:
             n_e_list:(:obj:'list') each entry is a list, recording the effective population sizes within an epoch

           Notes:
            - This function initializes demographic settings for a simulation, setting up populations
              and migration rates based on the graph structure.            
          """
           
          d = self.d
          self.ne_list=ne_list
          self.demography = msprime.Demography() 
           
          if ne_list is None:
             ne_list=[]
             # Default: equal effective population sizes
             for i in range(self.num_epochs):
                 ne_list.append(np.ones(d).tolist())
          
          
          for x in range(d):
              self.demography.add_population(initial_size=self.ne_list[0][x],
                                             initially_active=True)
              
          for x in range(d):
              for y in list(self.st_digraph.neighbors(x)):
                  self.demography.set_migration_rate(f"pop_{x}", f"pop_{y}", self.st_digraph[x][y]['weight_list'][0])   
              
          for i in range(self.num_epochs-1):
              for x in range(d):
                  self.demography.add_population_parameters_change(time=self.time_points[i+1],population=f"pop_{x}",initial_size=self.ne_list[i+1][x])
                  for y in list(self.st_digraph.neighbors(x)):
                      self.demography.add_migration_rate_change(time=self.time_points[i+1],rate=self.st_digraph[x][y]['weight_list'][i+1],
                                                                source=f"pop_{x}",dest=f"pop_{y}")                                                    
          return()        
           
      def simulate_average_pairwise_total_branch_length(self,
                                                        sequence_length,
                                                        target_n_replicates,
                                                        n_print):

          """ Simulates tree sequences (without mutations) based on the demographic model.
              Returns one tree sequence per replicate, with haploid samples.

              Parameters:
              sequence_length (float): Length of the genome to simulate.
              target_n_replicates (int): Number of independent tree sequences to simulate.
              n_print (int): Frequency of progress updates.

               Returns:
               list[msprime.TreeSequence]: Raw tree sequences (no mutations).
          """
          d = len(self.st_digraph.nodes)
          k=self.n_samples_per_node
          self.demography.sort_events()
          
          samples = []
          for i in range(self.num_epochs):     # Iterate epochs first
              for x in range(d):  # Then iterate populations
                  sample_size = self.st_digraph.nodes[x]['sample_size_list'][i]
                  if sample_size > 0:
                     samples.append(msprime.SampleSet(sample_size,
                                                      population=f"pop_{x}",
                                                      time=self.time_points[i],  # Use epoch's time
                                                       ))

          # Simulate tree sequences (ancestry only, no mutations)
          ts_generator = msprime.sim_ancestry(samples=samples,
                                         demography=self.demography,
                                         sequence_length=sequence_length,
                                         num_replicates=target_n_replicates,
                                         ploidy=1)
          
          matrix=np.zeros((d*self.num_epochs,d*self.num_epochs))
          for i, ts in enumerate(ts_generator):
              sample_sets= [ts.samples()[i*k : (i+1)*k] for i in range(d*self.num_epochs)]
              matrix=matrix+ts.divergence_matrix(sample_sets,mode='branch')
              if i % n_print == 0:
                  print(f"Simulated tree sequence {i}/{target_n_replicates}")

          average_pairwise_total_branch_length=matrix/target_n_replicates
          return average_pairwise_total_branch_length
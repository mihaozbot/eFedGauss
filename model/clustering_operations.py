
import torch
import numpy as np

# Attempt to load the line_profiler extension
'''
try:
    from line_profiler import LineProfiler
    profile = LineProfiler()  # If line_profiler is available, use it
except ImportError:
    #If line_profiler is not available, define a dummy profile decorator
    def profile(func): 
        return func
''' 
class ClusteringOps:
    def __init__(self, parent):
        self.parent = parent # Reference to the parent class
        self.feature_dim = parent.feature_dim # Number of features in the dataset
        self.Gamma_max = np.exp(-(parent.num_sigma**2)) # Maximum value of Gamma (used to determine if a new cluster should be added)
        #self.Gamma_max = np.exp(-(parent.num_sigma**2)*(self.feature_dim**np.sqrt(2))) # Maximum value of Gamma (used to determine if a new cluster should be added)
           
    def _add_new_cluster(self, z, label):
        ''' Add a new cluster to the model. This is called when no matching clusters are found or when the Gamma value is too low.'''
        
        # Ensure the parameters have enough space
        self.parent.overseer.ensure_capacity(self.parent.c+2)

        # Perform the updates in place
        with torch.no_grad():  # Temporarily disable gradient tracking
            self.parent.mu.data[self.parent.c] = z
            self.parent.S.data[self.parent.c] = self.parent.S_0
            self.parent.n.data[self.parent.c] = 1.0
        
        self.parent.score[self.parent.c] = 0
        self.parent.num_pred[self.parent.c] = 0
        self.parent.age[self.parent.c] = 0
        
        # Construct a diagonal matrix from these reciprocal values
        self.parent.S_inv[self.parent.c] = torch.diag(1.0 / (self.parent.S_0.diagonal()*self.parent.feature_dim))

        # Update cluster_labels
        # If cluster_labels is not a Parameter and does not require gradients, update as a regular tensor
        self.parent.cluster_labels[self.parent.c] = self.parent.one_hot_labels[label]
    
        #Add a new Gamma value for the new cluster equal to 1 (Gamma is the weight of the cluster) 
        self.parent.Gamma = torch.cat((self.parent.Gamma, torch.tensor([1.0], dtype=torch.float32, device=self.parent.device)))
        
        self.parent.matching_clusters = torch.cat((self.parent.matching_clusters, torch.tensor([self.parent.c], device=self.parent.device)))

        self.parent.c += 1# Increment the number of clusters
    

    def update_S_0(self):
        ''' Update the smallest cluster covariance matrix based on global statistics, before adding a new cluster. '''
        
        #Compute the parameter spread from the global statistics
        S_0 = self.parent.var_glo/(self.parent.N_r) 

        #Update smallest cluster covariance matrix
        self.parent.S_0 = torch.diag(S_0)
        self.parent.S_0 = torch.max(self.parent.S_0, self.parent.S_0_initial)

     
    def _increment_cluster(self, z, j):

        e = z - self.parent.mu[j] # Error between the sample and the cluster mean
        self.parent.mu[j] += 1 / (1 + self.parent.n[j]) * e

        # Check if self.parent.n[j] is equal to 1, the cluster can still be reset to the default
        if self.parent.n[j] < 2:
            with torch.no_grad():
                self.parent.S[j] = self.parent.S_0.clone()
    
        # Incremental clustering
        self.parent.S[j] = self.parent.S[j] + e.view(-1, 1) @ (z - self.parent.mu[j]).view(1, -1) #+ self.parent.S_0
        self.parent.n[j] = self.parent.n[j] + 1 
        self.parent.age[j] = 0

    def _increment_clusters(self, z):
        ''' Decide whether to increment an existing cluster or add a new cluster based on the current state. '''

        #Normalize membership functions 
        NGamma = self.parent.Gamma[self.parent.matching_clusters]/torch.sum(self.parent.Gamma[self.parent.matching_clusters])  
        
        if torch.isnan(NGamma).any().item():
            print("NaN detected in NGamma")

        # Calculate the error for the data point for each cluster
        z_expanded = z.unsqueeze(0).expand(len(self.parent.matching_clusters), self.parent.feature_dim)
        e_c = z_expanded - self.parent.mu[self.parent.matching_clusters]  # shape [self.parent.current_capacity, feature_dim]

        # Update cluster means
        #self.parent.mu[0:self.parent.c] += NGamma.unsqueeze(1) / (self.parent.n[0:self.parent.c].unsqueeze(1)) * e_c
        self.parent.mu[self.parent.matching_clusters] += NGamma.unsqueeze(1) / (self.parent.n[self.parent.matching_clusters].unsqueeze(1)) * e_c

        # e_c_transposed for matrix multiplication, shape [self.parent.current_capacity, feature_dim, 1]
        e_c_transposed = e_c.unsqueeze(-1)  # shape [self.parent.current_capacity, feature_dim, 1]
        self.parent.S[self.parent.matching_clusters]  = self.parent.S[self.parent.matching_clusters]*self.parent.forgeting_factor + NGamma.unsqueeze(-1).unsqueeze(-1) * torch.bmm((z_expanded - self.parent.mu[self.parent.matching_clusters]).unsqueeze(-1), e_c_transposed.transpose(1, 2))

        # Update number of samples in each cluster
        self.parent.n[self.parent.matching_clusters] = self.parent.n[self.parent.matching_clusters]*self.parent.forgeting_factor + NGamma
                
    def increment_or_add_cluster(self, z, label):
        ''' Increment an existing cluster if a cluster is activated enough, else add a new one'''
        
        self.parent.age[self.parent.matching_clusters] += 1 #/self.parent.n[self.parent.matching_clusters].clone()

        if len(self.parent.matching_clusters) == 0:
            self._add_new_cluster(z, label)
            # logging.info(f"Info. Added new cluster for label {label} due to no matching clusters. Total clusters now: {self.parent.c}")
            j = self.parent.c - 1

        else:

            # Find the index of the overall max Gamma value
            j_max = torch.argmax(self.parent.Gamma[self.parent.matching_clusters], dim=0).item()
            j = self.parent.matching_clusters[j_max]

            self.parent.age[j_max] = 0

            # Check if this index is in the matching_clusters
            if self.parent.Gamma[j] >= self.Gamma_max:
                # Increment the cluster with the max Gamma value
                self._increment_cluster(z, j)
            else:
                # Add a new cluster since the max Gamma cluster is not a matching one or its value is below the threshold
                self._add_new_cluster(z, label)
                j = self.parent.c - 1

        # Compute inverse of S[j]/n[j]
        try:
            self.parent.S_inv[j] = torch.linalg.inv((self.parent.S[j] / self.parent.n[j]) * self.parent.feature_dim)
        except:
            #self.parent.S_inv[j] = torch.linalg.pinv((self.parent.S[j] / self.parent.n[j]) * self.parent.feature_dim)
            with torch.no_grad():
                self.parent.removal_mech.remove_cluster(j)

    def update_global_statistics(self, z, label):
        ''' Update the global mean, covariance, and count based on the new data point. '''
        # Increment the count for the specific class
        self.parent.n_glo[label] += 1

        # Calculate the total number of samples
        total_samples = torch.sum(self.parent.n_glo)

        # Update global statistics
        e_glo = z - self.parent.mu_glo  # Error between the sample and the global mean
        self.parent.mu_glo += e_glo / total_samples  # Update mean
        self.parent.S_glo = self.parent.S_glo + (total_samples - 1) / total_samples * e_glo * e_glo # Update variance (not normalized to reduce computation)
        self.parent.var_glo = self.parent.S_glo / total_samples # Update standard deviation

        self.update_S_0() # Update S_0 based on the new sample
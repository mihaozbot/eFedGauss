import torch
import torch.nn as nn
import math
import numpy as np

from model.clustering_operations import ClusteringOps
from model.removal_mechanism import RemovalMechanism 
from model.merging_mechanism import MergingMechanism
from model.math_operations import MathOps
from model.consequence_operations import ConsequenceOps
from model.model_operations import ModelOps
from model.federated_operations import FederalOps

from utils.utils_train import test_model_in_batches
from collections import defaultdict

# Attempt to load the line_profiler extension

class eFedGauss(torch.nn.Module):
    def __init__(self, feature_dim, num_classes, kappa_n, num_sigma, kappa_join, S_0, N_r, c_max, device):
        super(eFedGauss, self).__init__()
        self.device = device
        self.feature_dim = feature_dim #Dimensionality of the features
        self.kappa_n = kappa_n #Minimal number of samples
        self.num_sigma = num_sigma/np.sqrt(self.feature_dim) #Activation distancethreshold*self.feature_dim**(1/np.sqrt(2)) 
        self.kappa_join = kappa_join #Merging threshold
        self.S_0 = S_0 * torch.eye(self.feature_dim, device=self.device) #Initialization covariance matrix
        self.S_0_initial = self.S_0.clone() #Initial covariance matrix, used to ensure that the clusters do not become too small
        self.N_r = N_r #Quantization number
        self.num_classes = num_classes #Max number of samples in clusters
        self.c_max = c_max #Max number of clusters

        # Dynamic properties initialized with tensors
        self.c = 0 # Number of active clusters
        self.Gamma = torch.empty(0, dtype=torch.float32, device=device,requires_grad=False)
        self.current_capacity = c_max #Initialize current capacity, which will be expanded as needed during training 
        self.cluster_labels = torch.empty((self.current_capacity, num_classes), dtype=torch.int32, device=device) #Initialize cluster labels
        
        self.score = torch.empty((self.current_capacity,), dtype=torch.float32, device=device) #Initialize cluster labels
        self.num_pred = torch.empty((self.current_capacity,), dtype=torch.float32, device=device) #Initialize number of predictions
        self.age = torch.empty((self.current_capacity,), dtype=torch.float32, device=device) #Initialize cluster age

        self.one_hot_labels = torch.eye(num_classes, dtype=torch.int32) #One hot labels 
        
        # Trainable parameters
        self.n = nn.Parameter(torch.zeros(self.current_capacity, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster sizes
        self.mu = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize cluster means
        self.S = nn.Parameter(torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device, requires_grad=False))  # Initialize covariance matrices
        self.S_inv = torch.zeros(self.current_capacity, feature_dim, feature_dim, dtype=torch.float32, device=device)  # Initialize covariance matrices

        # Global statistics
        self.n_glo = torch.zeros((num_classes), dtype=torch.float32, device=device)  # Global number of sampels per class
        self.mu_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)  # Global mean
        self.S_glo = torch.zeros((feature_dim), dtype=torch.float32, device=device)  # Sum of squares for global variance

        # Initialize subclasses
        self.overseer = ModelOps(self)
        self.mathematician = MathOps(self)
        self.clusterer = ClusteringOps(self)
        self.merging_mech = MergingMechanism(self)
        self.removal_mech = RemovalMechanism(self)
        self.consequence = ConsequenceOps(self)
        self.federal_agent = FederalOps(self)
          
    def toggle_evolving(self, enable=None):
        ''' Function to toggle the evolving state of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable. '''
        self.overseer.toggle_evolving(enable)

    def toggle_adding(self, enable=None):
        ''' Function to toggle the adding mechanism of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable.'''
        self.overseer.toggle_adding(enable)

    def toggle_merging(self, enable=None):
        ''' Function to toggle the merging mechanism of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable. '''
        self.overseer.toggle_merging(enable)

    def toggle_debugging(self, enable=None):
        ''' Function to toggle the debugging state of the model. If enable is None, the state will be toggled. Otherwise, the state will be set to the value of enable. '''
        self.overseer.toggle_debugging(enable)
        
    def federated_merging(self):
        ''' Executes the merging mechanism (and removal) for all rules. Conversely, the normal merging mechanism works on a subset of rules based on some conditions. '''
        self.federal_agent.federated_merging()

    def clustering(self, data, labels):
        
        #The method can not handle batched data directly
        for (z, label) in zip(data, labels): 

            # Update global statistics
            self.clusterer.update_global_statistics(z, label)
            
            # In training, match clusters based on the label
            self.matching_clusters = torch.where(self.cluster_labels[:self.c][:, label] == 1)[0]
            #self.matching_clusters = torch.arange(self.c, dtype=torch.int32, device=self.device) #In case all activations are needed
            
            # Compute activation
            self.Gamma = self.mathematician.compute_activation(z)
            
            # Evolving mechanisms
            if self.evolving:
                with torch.no_grad():
                    
                    #Incremental clustering and cluster addition
                    self.clusterer.increment_or_add_cluster(z, label)
                    
                    #Cluster merging
                    self.merging_mech.merging_mechanism()
    

    def forward(self, data):
        
        #Compute the activations of the cluster membership functions
        self.matching_clusters = torch.arange(self.c).repeat(data.shape[0], 1) #Used to select clusters during training, here we select all
        self.Gamma = self.mathematician.compute_batched_activation(data)
    
        # Defuzzify label scores for the entire batch
        label_scores, preds_max = self.consequence.defuzzify_batch()

        #Compute soft output probabilities
        scores = label_scores.clone().detach().requires_grad_(False)
        
        #Compute hard output classification
        clusters = self.Gamma.argmax(dim=1) 

        return scores, preds_max, clusters
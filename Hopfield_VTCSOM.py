#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import csv
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import imageio
import dhnn
import copy
from scipy.integrate import odeint



class Stochastic_Hopfield_nn(dhnn.DHNN):
    def __init__(self, x, y, pflag, nflag, patterns):
        """"patterns is a list like [p1,p2,p3,p4]"""
        self._x = x
        self._y = y
        self._neigx = np.arange(x)
        self._neigy = np.arange(y) 
        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        self.N = x*y
        self.pflag = pflag
        self.nflag = nflag
        self.state = 0
        self.dynamics_state = []
        self.Beta_maps = np.zeros((1, self._x, self._y))
        self.avg_state = np.zeros((1, self._x, self._y))
        self.face_mask = patterns[0]
        self.place_mask = patterns[1]
        self.limb_mask = patterns[2]
        self.object_mask = patterns[3]
        
    def rebuild_up_param(self):
        self.state = 0
        self.dynamics_state = []
        self.Betas = []
        self.avg_state = np.zeros((1, self._x, self._y))  
        self.Beta_maps = np.zeros((1, self._x, self._y))
        self.temp_Beta_map = np.zeros((1, self._x, self._y))
    
    def _gaussian_projection(self, c, sigma):
        """Returns a Gaussian centered in c."""
        d = 2*sigma*sigma
        ax = np.exp(-np.power(self._xx-self._xx.T[c], 2)/d)
        ay = np.exp(-np.power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T
    
    def _exponential_projection(self, c, lamda):
        """Returns a Exponential centered in c."""
        ax = np.power(self._xx-self._xx.T[c], 2)
        ay = np.power(self._yy-self._yy.T[c], 2)
        ds = np.sqrt(ax+ay)
        exp_proj = np.exp(-lamda*ds)
        return exp_proj
    
    def reconstruct_w(self, data):
        """Training pipeline.
        Arguments:
            data {list} -- each sample is vector
        Keyword Arguments:
            issave {bool} -- save weight or not (default: {True})
            wpath {str} -- the local weight path (default: {'weigh.npy'})
        """
        mat = np.vstack(data)
        self._w = np.dot(mat.T, mat)
        for i in range(self._w.shape[0]):
            self._w[i,i] = 0 
        self._w = (1/self.N) * self._w
            
    def reconstruct_w_with_structure_constrain(self, data, structure, parm):
        """"
        make weights by Hebb rule, but with structure constrain
        """
        # Hebb weights
        mat = np.vstack(data)
        self._w = np.dot(mat.T, mat)
        for i in range(self._w.shape[0]):
            self._w[i,i] = 0 
        # Gaussian structure
        if structure=='gaussian':
            projection = self._gaussian_projection
        if structure=='exponential':
            projection = self._exponential_projection
        for i in tqdm(range(self._x)):
            for j in range(self._y):
                index = i*self._x + j
                p = projection((i,j),parm).reshape(-1)
                self._w[index] = np.random.binomial(1, p, p.shape[0]) * self._w[index]
        self._w = (1/self.N) * self._w
        
    def reconstruct_w_with_structure_constrain_in_symmetric_way(self, data, structure, parm):
        """"
        make weights by Hebb rule, but with structure constrain
        """
        # Hebb weights
        mat = np.vstack(data)
        self._w = np.dot(mat.T, mat)
        for i in range(self._w.shape[0]):
            self._w[i,i] = 0 
        # Gaussian structure
        if structure=='gaussian':
            projection = self._gaussian_projection
        if structure=='exponential':
            projection = self._exponential_projection
        for i in tqdm(range(self._x)):
            for j in range(self._y):
                index = i*self._x + j
                p = projection((i,j),parm).reshape(-1)
                self._w[index] = np.random.binomial(1, p, p.shape[0]) * self._w[index]
        self._w = (1/self.N) * self._w
        w_triu = np.triu(self._w, k=1)
        self._w = w_triu + w_triu.T
        del w_triu

    def reconstruct_w_with_random_connection(self, data):
        """"
        make weights by Hebb rule, but with structure constrain
        """
        # Hebb weights
        mat = np.vstack(data)
        self._w = np.dot(mat.T, mat)
        for i in range(self._w.shape[0]):
            self._w[i,i] = 0 
        # random connection structure
        for i in tqdm(range(self._x)):
            for j in range(self._y):
                index = i*self._x + j
                self._w[index] = np.random.binomial(1, 0.5, self._w[index].shape[0]) * self._w[index]
        self._w = (1/self.N) * self._w
        
    def sampled_by_firing_rate(self, activation):
        def normalize(x):
            x_normalized = (x-x.min())/(x.max()-x.min())
            return x_normalized
        firing_rate = normalize(activation)
        return 2*np.random.binomial(1, firing_rate, firing_rate.shape)-1
        
    def stochastic_activation(self, beta, bi):
        return 1/(1+np.exp(-beta*bi))
    
    # def stochastic_dynamics(self, state, beta, H_prior, H_bottom_up, epochs, save_inter_step):
    #     """
    #     beta: noise
    #     H_prior: external field from higher area, the size is (200x200)
    #     save_inter_step: 1000
    #     """
    #     if isinstance(state, list):
    #         state = np.asarray(state)
    #     self.dynamics_state = state.reshape(1, self._x,self._y)
    #     indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
    #     H_prior = H_prior.reshape(self._x*self._y, 1)
    #     temp_avg_state = 0
    #     for step,ind in tqdm(enumerate(indexs)):
    #         bottom_up = self.sampled_by_firing_rate(H_bottom_up)
    #         H_top_bottom = H_prior + bottom_up.reshape(self._x*self._y, 1)
    #         diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
    #         diagonal = np.expand_dims(diagonal, -1)
    #         value = np.apply_along_axis(
    #             lambda x: self.pflag if self.stochastic_activation(beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
    #         for i in range(len(state)):
    #             state[i, ind[i]] = value[i]
    #         temp_avg_state += state.reshape(self._x,self._y)
    #         if step%save_inter_step == 0:
    #             self.dynamics_state = np.vstack((self.dynamics_state, state.reshape(1,self._x,self._y)))
    #             self.avg_state = np.vstack((self.avg_state, temp_avg_state.reshape(1,self._x,self._y)))
    #             temp_avg_state = 0
    #             plt.figure()
    #             plt.imshow(bottom_up)
    #     self.state = state.reshape(self._x,self._y)
    #     return state

    def stochastic_dynamics(self, state, beta, H_prior, H_bottom_up, epochs, save_inter_step):
        """
        beta: noise
        H_prior: external field from higher area, the size is (200x200)
        save_inter_step: 1000
        """
        if isinstance(state, list):
            state = np.asarray(state)
        self.dynamics_state = state.reshape(1, self._x,self._y)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        H_top_bottom = H_prior + H_bottom_up.reshape(self._x*self._y, 1)
        temp_avg_state = 0
        for step,ind in tqdm(enumerate(indexs)):
            diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
            diagonal = np.expand_dims(diagonal, -1)
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
            temp_avg_state += state.reshape(self._x,self._y)
            if step%save_inter_step == 0:
                self.dynamics_state = np.vstack((self.dynamics_state, state.reshape(1,self._x,self._y)))
                self.avg_state = np.vstack((self.avg_state, temp_avg_state.reshape(1,self._x,self._y)))
                temp_avg_state = 0
        self.state = state.reshape(self._x,self._y)
        return state
    
    def Field_dynamics(self, state, beta, H_prior, H_bottom_up, epochs, save_inter_step):
        """
        beta: noise
        H_prior: external field from higher area, the size is (200x200)
        save_inter_step: 1000
        """
        if isinstance(state, list):
            state = np.asarray(state)
        self.dynamics_state = state.reshape(1, self._x,self._y)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        H_top_bottom = H_prior + H_bottom_up.reshape(self._x*self._y, 1)
        Field = []
        Field_temp = []
        for step,ind in tqdm(enumerate(indexs)):
            diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
            # field start
            field = self._w[ind].reshape(self._x,self._y) * (state.reshape(self._x,self._y)+H_top_bottom.reshape(self._x,self._y))
            Field_temp.append(field)
            # field over
            diagonal = np.expand_dims(diagonal, -1)
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
            if step%save_inter_step == 0:
                Field.append(np.array(Field_temp).mean(0))
                Field_temp = []
        self.state = state.reshape(self._x,self._y)
        return state, np.array(Field)
    
    def stochastic_dynamics_changed_beta_in_mask(self, state, beta, mask_region_beta, H_prior, H_bottom_up, epochs, save_inter_step):
        """
        beta: noise
        H_prior: external field from higher area, the size is (200x200)
        save_inter_step: 1000
        """
        mask_name = {0:'face_mask', 1:'place_mask', 2:'limb_mask', 3:'object_mask'}
        if isinstance(state, list):
            state = np.asarray(state)
        self.dynamics_state = state.reshape(1, self._x,self._y)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        H_top_bottom = H_prior + H_bottom_up.reshape(self._x*self._y, 1)
        temp_avg_state = 0
        for step,ind in tqdm(enumerate(indexs)):
            pos = np.unravel_index(ind[0],(self._x,self._y))
            diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
            diagonal = np.expand_dims(diagonal, -1)
            index = np.array([mask[pos] for mask in [self.face_mask,self.place_mask,self.limb_mask,self.object_mask]])
            if sum(index)!=-4:
                real_beta = mask_region_beta[mask_name[np.where(index==1)[0][-1]]](beta, step, 10000)
            if sum(index)==-4:
                real_beta = beta
            self.temp_Beta_map[0,pos[0],pos[1]] += real_beta
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(real_beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
            temp_avg_state += state.reshape(self._x,self._y)
            if step%save_inter_step == 0:
                self.dynamics_state = np.vstack((self.dynamics_state, state.reshape(1,self._x,self._y)))
                self.avg_state = np.vstack((self.avg_state, temp_avg_state.reshape(1,self._x,self._y)))
                self.Beta_maps = np.vstack((self.Beta_maps, self.temp_Beta_map))
                temp_avg_state = 0
                self.temp_Beta_map = np.zeros((1,self._x,self._y))
        self.state = state.reshape(self._x,self._y)
        return state
    
    def stochastic_dynamics_changed_beta(self, state, beta, changed_beta_func, tao, H_prior, H_bottom_up, epochs, save_inter_step):
        """
        beta: an array or a list
        H_prior: external field from higher area, the size is (200x200)
        save_inter_step: 1000
        """
        if isinstance(state, list):
            state = np.asarray(state)
        self.dynamics_state = state.reshape(1, self._x,self._y)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        H_top_bottom = H_prior + H_bottom_up.reshape(self._x*self._y, 1)
        temp_avg_state = 0
        temp_Beta_map = np.zeros((1,self._x,self._y))
        for step,ind in tqdm(enumerate(indexs)):
            diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
            diagonal = np.expand_dims(diagonal, -1)
            pos = np.unravel_index(ind[0],(self._x, self._y))
            index = np.array([mask[pos] for mask in [self.face_mask,self.place_mask,self.limb_mask,self.object_mask]])
            if sum(index)!=-4:
                mask = [self.face_mask,self.place_mask,self.limb_mask,self.object_mask][np.where(index==1)[0][0]]
                region_avg_dynamics_state = self.avg_activation_in_mask_timeserise(self.dynamics_state, mask)
                real_beta = changed_beta_func(region_avg_dynamics_state, beta, tao)
            if sum(index)==-4:
                real_beta = beta
            temp_Beta_map[0,pos[0],pos[1]] += real_beta
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(real_beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
            temp_avg_state += state.reshape(self._x,self._y)
            if step%save_inter_step == 0:
                self.dynamics_state = np.vstack((self.dynamics_state, state.reshape(1,self._x,self._y)))
                self.avg_state = np.vstack((self.avg_state, temp_avg_state.reshape(1,self._x,self._y)))
                self.Beta_maps = np.vstack((self.Beta_maps, temp_Beta_map))
                temp_avg_state = 0
                temp_Beta_map = np.zeros((1,self._x,self._y))
        self.state = state.reshape(self._x,self._y)
        return state
    
    def stochastic_dynamics_changed_beta_with_IFG(self, state, beta, changed_beta_func, tao, H_prior, H_bottom_up, epochs, save_inter_step):
        """
        beta: an array or a list
        H_prior: external field from higher area, the size is (200x200)
        save_inter_step: 1000
        """
        if isinstance(state, list):
            state = np.asarray(state)
        self.dynamics_state = state.reshape(1, self._x,self._y)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        H_top_bottom = H_prior + H_bottom_up.reshape(self._x*self._y, 1)
        temp_avg_state = 0
        temp_Beta_map = np.zeros((1,self._x,self._y))
        for step,ind in tqdm(enumerate(indexs)):
            diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
            diagonal = np.expand_dims(diagonal, -1)
            pos = np.unravel_index(ind[0],(self._x, self._y))
            index = np.array([mask[pos] for mask in [self.face_mask,self.place_mask,self.limb_mask,self.object_mask]])
            if sum(index)!=-4:
                mask = [self.face_mask,self.place_mask,self.limb_mask,self.object_mask][np.where(index==1)[0][0]]
                region_avg_dynamics_state = self.avg_activation_in_mask_timeserise(self.dynamics_state, mask)
                real_beta = changed_beta_func(self, region_avg_dynamics_state, mask, beta, tao)
            if sum(index)==-4:
                real_beta = beta
            temp_Beta_map[0,pos[0],pos[1]] += real_beta
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(real_beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
            temp_avg_state += state.reshape(self._x,self._y)
            if step%save_inter_step == 0:
                self.dynamics_state = np.vstack((self.dynamics_state, state.reshape(1,self._x,self._y)))
                self.avg_state = np.vstack((self.avg_state, temp_avg_state.reshape(1,self._x,self._y)))
                self.Beta_maps = np.vstack((self.Beta_maps, temp_Beta_map))
                temp_avg_state = 0
                temp_Beta_map = np.zeros((1,self._x,self._y))
        self.state = state.reshape(self._x,self._y)
        return state
    
    def stochastic_dynamics_changed_beta_with_IFG_by_DDM(self, state, beta, changed_beta_func, tao, 
                                                         H_prior, H_bottom_up, epochs, save_inter_step):
        """
        beta: an array or a list
        H_prior: external field from higher area, the size is (200x200)
        save_inter_step: 1000
        """
        if isinstance(state, list):
            state = np.asarray(state)
        self.dynamics_state = state.reshape(1, self._x,self._y)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        H_top_bottom = H_prior + H_bottom_up.reshape(self._x*self._y, 1)
        temp_avg_state = 0
        temp_Beta_map = np.zeros((1,self._x,self._y))
        self.IFG_signal = []
        IFG_signal = 0
        IFG_output = 0
        for step,ind in tqdm(enumerate(indexs)):
            diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
            diagonal = np.expand_dims(diagonal, -1)
            pos = np.unravel_index(ind[0],(self._x, self._y))
            index = np.array([mask[pos] for mask in [self.face_mask,self.place_mask,self.limb_mask,self.object_mask]])
            if sum(index)!=-4:
                mask = [self.face_mask,self.place_mask,self.limb_mask,self.object_mask][np.where(index==1)[0][0]]
                region_avg_dynamics_state = self.avg_activation_in_mask_timeserise(self.dynamics_state, mask)
                # Calculate FFA and LO activation
                face_region_avg_dynamics_state = self.avg_activation_in_mask_timeserise(self.dynamics_state, self.face_mask)
                object_region_avg_dynamics_state = self.avg_activation_in_mask_timeserise(self.dynamics_state, self.object_mask)
                # IFG recieve signal with noise
                IFG_input = face_region_avg_dynamics_state[-1] + object_region_avg_dynamics_state[-1] + np.random.normal(0,1)
                # IFG signal acummulate
                IFG_signal += IFG_input
                self.IFG_signal.append(IFG_signal)
                # IFG out
                if IFG_signal>=100:
                    IFG_output = 1
                    IFG_signal = 0
                    temp_Beta_map[0] = np.where(self.face_mask + self.object_mask==0, 2, 0)
                else:
                    IFG_output = 0
                    # change beta by adaptation/bottom up/IFG top down
                    real_beta = changed_beta_func(region_avg_dynamics_state, beta, tao, IFG_output)
            if sum(index)==-4:
                real_beta = beta
            # record the Beta_map
            temp_Beta_map[0,pos[0],pos[1]] += real_beta
            # change the selected unit value with real_beta
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(real_beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
            # record the state
            temp_avg_state += state.reshape(self._x,self._y)
            if step%save_inter_step == 0:
                self.dynamics_state = np.vstack((self.dynamics_state, state.reshape(1,self._x,self._y)))
                self.avg_state = np.vstack((self.avg_state, temp_avg_state.reshape(1,self._x,self._y)))
                self.Beta_maps = np.vstack((self.Beta_maps, temp_Beta_map))
                temp_avg_state = 0
                temp_Beta_map = np.zeros((1,self._x,self._y))
        self.state = state.reshape(self._x,self._y)
        return state
    
    def stochastic_dynamics_changed_beta_with_IFG_by_RNN(self, state, beta, changed_beta_FFA, changed_beta_LO, 
                                                         H_prior, H_bottom_up, epochs, save_inter_step):
        """
        beta: an array or a list
        H_prior: external field from higher area, the size is (200x200)
        save_inter_step: 1000
        """
        if isinstance(state, list):
            state = np.asarray(state)
        self.dynamics_state = state.reshape(1, self._x,self._y)
        indexs = np.random.randint(0, len(self._w) - 1, (epochs, len(state)))
        H_prior = H_prior.reshape(self._x*self._y, 1)
        H_top_bottom = H_prior + H_bottom_up.reshape(self._x*self._y, 1)
        temp_avg_state = 0
        for step,ind in tqdm(enumerate(indexs)):
            # caculate beta
            pos = np.unravel_index(ind[0],(self._x, self._y))
            if pos[0] in np.where(self.face_mask==1)[0] and pos[1] in np.where(self.face_mask==1)[1]:
                #changed_beta = changed_beta_FFA[(step//save_inter_step)+1]
                changed_beta = changed_beta_FFA[step+100]
            if pos[0] in np.where(self.object_mask==1)[0] and pos[1] in np.where(self.object_mask==1)[1]:
                #changed_beta = changed_beta_FFA[(step//save_inter_step)+1]
                changed_beta = changed_beta_FFA[step+100]
            else:
                changed_beta = 15
            # SHNN dynamics
            diagonal = np.diagonal(np.dot(self._w[ind], state.T + H_top_bottom))
            diagonal = np.expand_dims(diagonal, -1)
            value = np.apply_along_axis(
                lambda x: self.pflag if self.stochastic_activation(changed_beta, x)>np.random.uniform(0,1,1) else self.nflag, 1, diagonal)
            for i in range(len(state)):
                state[i, ind[i]] = value[i]
            temp_avg_state += state.reshape(self._x,self._y)
            if step%save_inter_step == 0:
                self.dynamics_state = np.vstack((self.dynamics_state, state.reshape(1,self._x,self._y)))
                self.avg_state = np.vstack((self.avg_state, temp_avg_state.reshape(1,self._x,self._y)))
                temp_avg_state = 0
        self.state = state.reshape(self._x,self._y)
        return state

    
    def get_firing_rate(self, start_time, end_time):
        if self.dynamics_state.shape[0]>=3:
            if end_time=='end':
                temp = self.dynamics_state[start_time:, :, :].mean(axis=0)
            else:
                temp = self.dynamics_state[start_time:end_time, :, :].mean(axis=0)
        else:
            temp = self.dynamics_state[0:, :, :].mean(axis=0)
        #temp = (temp+1) / 2   # from [-1,1] to [0,1]
        return temp
    
    def firing_rate_model_dynamics(self, state):
        def F(x):
            return np.where(x<=0, 0, 0.8*x)
            #return x
        def diff_equation(v, t, M, H, tao):
            h = H
            Change = F((np.dot(M,v)+h))
            dvdt = tao * (-v + Change)
            return dvdt
        times = np.linspace(0,100,1000)
        H = state.reshape(-1)
        solution = odeint(diff_equation, H, times, args=(self._w, H, 1.2))   
        return solution
    
    def order_parameter(self, state, right):
        return np.mean(state * right)
    
    def create_w_table_for_Gephi(self, w_dir):
        with open(w_dir, 'a+', newline='') as csvfile:
            csv_write = csv.writer(csvfile)
            csv_write.writerow(['Source', 'Target'])
            for i in tqdm(range(self._x)):
                for j in range(self._y):
                    csv_write.writerow([i, j, self._w[i,j]])
                    



    #### Visulization ####
    ###########################################################################
    def dynamics_pattern(self, gif_name, patterns):    
        RGB_patterns = []
        for pattern in patterns:
            temp1 = np.where(pattern==1, 128, 192)
            temp2 = np.where(pattern==1, 42, 192)
            temp3 = np.where(pattern==1, 42, 192)
            RGB_patterns.append(np.array([temp1,temp2,temp3]))
        RGB_patterns = np.array(RGB_patterns)
        RGB_patterns = RGB_patterns.transpose((0,2,3,1))
        imageio.mimsave(gif_name, RGB_patterns, 'GIF', duration=0.05)
        
    def dynamics_Betas_map(self, gif_name, Betas_map):    
        imageio.mimsave(gif_name, Betas_map, 'GIF', duration=0.1)
        
    def avg_activation_in_mask_timeserise(self, patterns, mask):
        mean_avtivations = []
        for pattern in patterns:
            mean_avtivations.append(pattern[np.where(mask==1)].mean())
        mean_avtivations = np.array(mean_avtivations)
        return mean_avtivations
    
    def potential_change_map(self, pattern):
        temp = np.dot(self._w, self.state.reshape(-1,1)).reshape(200,200)
        prob_map = self.stochastic_activation(self.beta, temp)
        return prob_map
    
    def phase_space(self, patterns):
        fig = plt.figure(dpi=300)
        ax = fig.gca(projection='3d')
        x = self.avg_activation_in_mask_timeserise(patterns, self.face_mask) 
        y = self.avg_activation_in_mask_timeserise(patterns, self.limb_mask)  
        z = self.avg_activation_in_mask_timeserise(patterns, self.object_mask)   
        ax.plot(x, y, z, label='phase space')
        ax.legend()  
        plt.show()
            
    





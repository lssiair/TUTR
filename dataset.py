
from torch.utils import data
import numpy as np
import torch
import pickle


class TrajectoryDataset(data.Dataset):

    def __init__(self, dataset_path, dataset_name, dataset_type, translation=False, rotation=False, scaling=False, obs_len=8,
                max_neis_num=50, dist_threshold=2, smooth=False):
        
        self.translation = translation
        self.rotation = rotation
        self.obs_len = obs_len
        self.scaling = scaling
        self.max_neis_num = max_neis_num
        self.dist_threshold = dist_threshold
        self.smooth = smooth
        self.window_size = 3

        f = open(dataset_path + dataset_name + '_' + dataset_type + '.pkl', 'rb+')
        self.scenario_list = pickle.load(f)
        f.close()


    def coll_fn(self, scenario_list):

        # batch <list> [[ped, neis]]]
        ped, neis = [], []

        n_neighbors = []

        for item in scenario_list:  
            ped_obs_traj, ped_pred_traj, neis_traj = item[0], item[1], item[2] # [T 2] [N T 2] N is not a fixed number
            ped_traj = np.concatenate((ped_obs_traj[:, :2], ped_pred_traj), axis=0)
            neis_traj = neis_traj[:, :, :2].transpose(1, 0, 2)
            neis_traj = np.concatenate((np.expand_dims(ped_traj, axis=0), neis_traj), axis=0)
            distance = np.linalg.norm(np.expand_dims(ped_traj, axis=0) - neis_traj, axis=-1)
            distance = distance[:, :self.obs_len]
            distance = np.mean(distance, axis=-1) # mean distance
            # distance = distance[:, -1] # final distance
            neis_traj = neis_traj[distance < self.dist_threshold]

            n_neighbors.append(neis_traj.shape[0])
            if self.translation:
                origin = ped_traj[self.obs_len-1:self.obs_len] # [1, 2]
                ped_traj = ped_traj - origin
                if neis_traj.shape[0] != 0:
                    neis_traj = neis_traj - np.expand_dims(origin, axis=0) 
            
            if self.rotation:
                ref_point = ped_traj[0]
                angle = np.arctan2(ref_point[1], ref_point[0])
                rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                                          [np.sin(angle), np.cos(angle)]])
                ped_traj = np.matmul(ped_traj, rot_mat)
                if neis_traj.shape[0] != 0:
                    rot_mat = np.expand_dims(rot_mat, axis=0)
                    rot_mat = np.repeat(rot_mat, neis_traj.shape[0], axis=0)
                    neis_traj = np.matmul(neis_traj, rot_mat)

            if self.smooth:
                pred_traj = ped_traj[self.obs_len:]
                x_len = pred_traj.shape[0]
                x_list = []
                keep_num = int(np.floor(self.window_size / 2))
                for i in range(self.window_size):
                    x_list.append(pred_traj[i:x_len-self.window_size+1+i])
                x = sum(x_list) / self.window_size
                x = np.concatenate((pred_traj[:keep_num], x, pred_traj[-keep_num:]), axis=0)
                ped_traj = np.concatenate((ped_traj[:self.obs_len], x), axis=0)

            # if self.scaling:
            #     scale = np.random.randn(ped_traj.shape[0])*0.05+1
            #     scale = scale.reshape(ped_traj.shape[0], 1)
            #     ped_traj = ped_traj * scale
            #     if neis_traj.shape[0] != 0:
            #         neis_traj = neis_traj * scale
            
            ped.append(ped_traj)
            neis.append(neis_traj)
            
        max_neighbors = max(n_neighbors)
        neis_pad = []
        neis_mask = []
        for neighbor, n in zip(neis, n_neighbors):
            neis_pad.append(
                np.pad(neighbor, ((0, max_neighbors-n), (0, 0),  (0, 0)), 
                "constant")
            )
            mask = np.zeros((max_neighbors, max_neighbors))
            mask[:n, :n] = 1
            neis_mask.append(mask)

        ped = np.stack(ped, axis=0) 
        neis = np.stack(neis_pad, axis=0)
        neis_mask = np.stack(neis_mask, axis=0)

        ped = torch.tensor(ped, dtype=torch.float32)
        neis = torch.tensor(neis, dtype=torch.float32)
        neis_mask = torch.tensor(neis_mask, dtype=torch.int32)
        return ped, neis, neis_mask
        

    def __len__(self):
        return  len(self.scenario_list)

    def __getitem__(self, item):
        return self.scenario_list[item]
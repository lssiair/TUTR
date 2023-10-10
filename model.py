import torch
import torch.nn as nn

from transformer_encoder import Encoder
from transformer_decoder import Decoder


class TrajectoryModel(nn.Module):

    def __init__(self, in_size, obs_len, pred_len, embed_size, enc_num_layers, int_num_layers_list, heads, forward_expansion):
        super(TrajectoryModel, self).__init__()

        self.embedding = nn.Linear(in_size*(obs_len + pred_len), embed_size)

        self.mode_encoder = Encoder(embed_size, enc_num_layers, heads, forward_expansion, islinear=True)
        self.cls_head = nn.Linear(embed_size, 1)

        self.nei_embedding = nn.Linear(in_size*obs_len, embed_size)
        self.social_decoder =  Decoder(embed_size, int_num_layers_list[1], heads, forward_expansion, islinear=False)
        self.reg_head = nn.Linear(embed_size, in_size*pred_len)

    def spatial_interaction(self, ped, neis, mask):
        
        # ped [B K embed_size]
        # neis [B N obs_len 2]  N is the max number of agents of current scene
        # mask [B N N] is used to stop the attention from invalid agents

        neis = neis.reshape(neis.shape[0], neis.shape[1], -1)  # [B N obs_len*2]
        nei_embeddings = self.nei_embedding(neis)  # [B N embed_size]
        
        mask = mask[:, 0:1].repeat(1, ped.shape[1], 1)  # [B K N]
        int_feat = self.social_decoder(ped, nei_embeddings, mask)  # [B K embed_size]

        return int_feat # [B K embed_size]
    
    def forward(self, ped_obs, neis_obs, motion_modes, mask, closest_mode_indices, test=False, num_k=20):

        # ped_obs [B obs_len 2]
        # nei_obs [B N obs_len 2]
        # motion_modes [K pred_len 2]
        # closest_mode_indices [B]

        ped_obs = ped_obs.unsqueeze(1).repeat(1, motion_modes.shape[0], 1, 1)  # [B K obs_len 2]
        motion_modes = motion_modes.unsqueeze(0).repeat(ped_obs.shape[0], 1, 1, 1)

        ped_seq = torch.cat((ped_obs, motion_modes), dim=-2)  # [B K seq_len 2] seq_len = obs_len + pred_len
        ped_seq = ped_seq.reshape(ped_seq.shape[0], ped_seq.shape[1], -1)  # [B K seq_len*2]
        ped_embedding = self.embedding(ped_seq) # [B K embed_size]
        
        ped_feat = self.mode_encoder(ped_embedding)  # [B K embed_size]
        scores = self.cls_head(ped_feat).squeeze()  # [B K]

        if not test:
            index1 = torch.LongTensor(range(closest_mode_indices.shape[0])).cuda()  # [B]
            index2 = closest_mode_indices
            closest_feat = ped_feat[index1, index2].unsqueeze(1)  # [B 1 embed_size]

            int_feat = self.spatial_interaction(closest_feat, neis_obs, mask)  # [B 1 embed_size]
            pred_traj = self.reg_head(int_feat.squeeze())  # [B pred_len*2]

            return pred_traj, scores

        if test:
            top_k_indices = torch.topk(scores, k=num_k, dim=-1).indices  # [B num_k]
            top_k_indices = top_k_indices.flatten()  # [B*num_k]
            index1 = torch.LongTensor(range(ped_feat.shape[0])).cuda()  # [B]
            index1 = index1.unsqueeze(1).repeat(1, num_k).flatten() # [B*num_k]
            index2 = top_k_indices # [B*num_k]
            top_k_feat = ped_feat[index1, index2]  # [B*num_k embed_size]
            top_k_feat = top_k_feat.reshape(ped_feat.shape[0], num_k, -1)  # [B num_k embed_size]

            int_feats = self.spatial_interaction(top_k_feat, neis_obs, mask)  # [B num_k embed_size]
            pred_trajs = self.reg_head(int_feats)  # [B num_k pred_size*2]

            return pred_trajs, scores
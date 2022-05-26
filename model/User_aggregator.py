import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools
from model.attention import AttentionLayer, SelfAttentionLayer_tuser

class User_aggregator(nn.Module):
    def __init__(self, u2e, v2e, g2e, embedding_dim, u_groups_dict, group_member_dict, eta, drop_ratio, device):
        super(User_aggregator, self).__init__()
        self.u2e = u2e
        self.v2e = v2e
        self.g2e = g2e
        self.embedding_dim = embedding_dim
        self.u_groups_dict = u_groups_dict
        self.group_member_dict = group_member_dict
        self.eta = eta
        self.drop_ratio = drop_ratio
        self.device = device
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.attention_u = AttentionLayer(embedding_dim, drop_ratio)
        self.self_attention_tuser = SelfAttentionLayer_tuser(embedding_dim, drop_ratio)
        self.linear1 = nn.Linear(3 * embedding_dim, embedding_dim)
        
    def forward(self, user_inputs, item_inputs, type_a):
        
        if type_a == 'han':
            ####### Inherent user embedding ###################
            user_embeds_full = self.u2e(user_inputs) # [B, C]
            g_embeds_with_han = self.get_group_view_useremb(user_inputs, item_inputs)
            u_Attention = self.eta * g_embeds_with_han + (1 - self.eta) * user_embeds_full # [B, C] HAN-CDGR
            u_Attention = u_Attention.to(self.device)

            return u_Attention
        
        if type_a == 'fixed_agg':
            user_embeds_full = self.u2e(user_inputs) # [B, C]
            g_embeds_with_fixed_agg = torch.zeros([len(user_inputs), self.embedding_dim])
            ####### fuzzy_user_groups_agg #################
            groups_ids = [self.u_groups_dict[usr.item()] for usr in user_inputs] # [B,1]
            MAX_MENBER_SIZE = max([len(menb) for menb in groups_ids]) # the great group size = 44
            menb_ids, mask = [None]*len(groups_ids), [None]*len(groups_ids)
            for i in range(len(groups_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(groups_ids[i]))
                menb_ids[i] = groups_ids[i] + postfix
                mask[i] = [1]*len(groups_ids[i]) + postfix
            
            menb_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)
            
            menb_emb =  self.g2e(menb_ids) # [B, N, C]
            menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
            # g_embeds_with_fixed_agg = torch.mean(menb_emb, dim=1) # [B,C]
            # g_embeds_with_fixed_agg  = torch.min(menb_emb, dim=1).values
            # g_embeds_with_fixed_agg = torch.max(menb_emb, dim=1).values
            g_embeds_with_fixed_agg = torch.median(menb_emb, dim=1).values

            user_emb = user_embeds_full + g_embeds_with_fixed_agg
            user_emb = user_emb.to(self.device)

            return user_emb
    
    def get_group_view_useremb(self, user_inputs, item_inputs):
        g_embeds_with_han = torch.zeros([len(user_inputs), self.embedding_dim])
        ####### Group-view user embedding #################
        user_inputs_keys = [self.get_keys(self.group_member_dict, usr.item()) for usr in user_inputs]

        new_user_inputs = [None] * len(user_inputs)
        for i in range(len(user_inputs)):
            new_user_inputs[i] = [user_inputs[i]] * len(user_inputs_keys[i]) 

        new_user_inputs = [usr for u in new_user_inputs for usr in u] # flatten the nested list new_user_inputs  length = X
        new_user_inputs = torch.Tensor(new_user_inputs).long().to(self.device) # shape: (X,)
        new_user_embeds = self.u2e(new_user_inputs) # shape:[X, C]

        group_input = [group_id for group in user_inputs_keys for group_id in group] # flatten the user_inputs_keys which is a nested list
        group_user_ids = [self.group_member_dict[k] for k in group_input] # length = X

        # get the great group size
        MAX_MENBER_SIZE = max([len(menb) for menb in group_user_ids]) # the great group size = 4
        # menb_ids is group members and empty members, mask1 is to mask the empty members, mask is to mask all the other members that is not the user_input id
        menb_ids, mask1 = [None]*len(group_user_ids),  [None]*len(group_user_ids)
        for i in range(len(group_user_ids)):
            postfix = [0]*(MAX_MENBER_SIZE - len(group_user_ids[i])) 
            menb_ids[i] = group_user_ids[i] + postfix

            mask1[i] = [1]*len(group_user_ids[i]) + postfix

        menb_ids, mask1 = torch.Tensor(menb_ids).long().to(self.device),\
                                    torch.Tensor(mask1).float().to(self.device)
        # [X,N] : menb_ids, mask1
        
        mask = (menb_ids == new_user_inputs.unsqueeze(-1)).float().to(self.device) # [X, N]
        # mask = torch.where()

        # Get the menb_emb
        menb_emb =  self.u2e(menb_ids) # [B, N, C] 
        menb_emb *= mask1.unsqueeze(dim=-1) # [B, N, C] * [B,N,1] = [B,N,C] Turn the empty menber rows into empty rows
        
        ################################
        ## Self-attention part #########
        ################################
        proj_query_emb, proj_key_emb, proj_value_emb = self.self_attention_tuser(menb_emb) # [B, N, C/2], [B, N, C/2], [B, N, C]
        proj_query_emb_new = proj_query_emb * mask1.unsqueeze(dim=-1)
        proj_key_emb_new = proj_key_emb * mask1.unsqueeze(dim=-1)
        energy = torch.bmm(proj_query_emb_new, proj_key_emb_new.permute(0,2,1))/torch.sqrt(torch.tensor(menb_emb.shape[-1], dtype=torch.float32)) # [B, N , N]
        energy = torch.clip(energy, -50, 50)
        energy_exp = energy.exp() * mask1.unsqueeze(dim=1)

        energy_exp_softmax = energy_exp/torch.sum(energy_exp, dim=-1, keepdim=True) # [B, N, N] 
        
        menb_emb_out = torch.bmm(energy_exp_softmax, proj_value_emb) # [G, N, N] * [G, N, C] = [B, N, C]
        menb_emb_out_new = menb_emb_out * mask1.unsqueeze(dim=-1) # [B,N,C]
        user_emb_out = menb_emb_out_new * mask.unsqueeze(-1) # [B,N,C] * [B,N,1] = [B,N,C]
        user_emb_out_new = torch.sum(user_emb_out, dim=1) # collapse the rows of user_emb_out and get a [B, C] matrix
        overall_user_emb_out = 0.5 * user_emb_out_new + 0.5 * new_user_embeds # shape: [X, C]

        #############################
        ## Vanilla attention part ###
        #############################
        attn_weights = self.attention_u(overall_user_emb_out)# [X, 1]
        # Get a mask matrix to detect which user has joined more than one group
        mask2 = [None] * len(user_inputs)
        for i in range(len(user_inputs)):
            mask2[i] = (new_user_inputs == user_inputs[i]).cpu().numpy()

        mask2 = torch.Tensor(mask2).float().to(self.device) # The shape of mask2: [B, X]
        # multiplies each element of mask2 with the corresponding element of the attn_weights
        new_mask2 = torch.mul(mask2, attn_weights.view(1, -1)) #[B, X]
        # softmax
        new_mask2_sm = new_mask2/torch.sum(new_mask2, dim=-1, keepdim=True)
        # get each user's integrated preference
        g_embeds_with_han = torch.mm(new_mask2_sm, overall_user_emb_out) # [B, X] * [X, C] = [B, C]
        
        return g_embeds_with_han

    def get_keys(self, d, value):
    
        return [k for k, v in d.items() if value in v]



        








            






            






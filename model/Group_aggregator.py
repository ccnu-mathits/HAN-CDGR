import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools
from model.attention import AttentionLayer, SelfAttentionLayer_tuser

class Group_aggregator(nn.Module):
    def __init__(self, u2e, v2e, g2e, embedding_dim, group_member_dict, lmd, drop_ratio, device):
        super(Group_aggregator, self).__init__()
        self.u2e = u2e
        self.v2e = v2e
        self.g2e = g2e
        self.embedding_dim = embedding_dim
        self.group_member_dict = group_member_dict
        self.lmd = lmd
        self.drop_ratio = drop_ratio
        self.device = device
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.self_attention_tuser = SelfAttentionLayer_tuser(embedding_dim, drop_ratio)

    def forward(self, gro_inputs, item_inputs, type_a):
    
        if type_a == 'han':
            group_embeds_full = self.g2e(gro_inputs) # [B, C] M1-b
            g_embeds_with_attention = self.get_member_view_groupemb(gro_inputs, item_inputs) # M1-a
            gro_attention = self.lmd * g_embeds_with_attention + (1 - self.lmd) * group_embeds_full # HAN-CDGR
            gro_attention = gro_attention.to(self.device)

            return gro_attention
        
        elif type_a == 'fixed_agg':
            # get the group full embedding vectors
            group_embeds_full = self.g2e(gro_inputs)
            ####### group-members-agg #################
            start = time.time()
            user_ids = [self.group_member_dict[usr.item()] for usr in gro_inputs] # [B,1]
            MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 44
            menb_ids, mask = [None]*len(user_ids),  [None]*len(user_ids)
            for i in range(len(user_ids)):
                postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
                menb_ids[i] = user_ids[i] + postfix
                mask[i] = [1]*len(user_ids[i]) + postfix
            
            menb_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                        torch.Tensor(mask).float().to(self.device)
            
            menb_emb =  self.u2e(menb_ids) # [B, N, C] 
            # menb_emb =  self.u_aggregator(menb_ids) # [B, N, C]
            menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
            
            g_embeds_with_avg = torch.mean(menb_emb, dim=1) # [B,C]
            # g_embeds_with_lm  = torch.min(menb_emb, dim=1).values
            # g_embeds_with_ms = torch.max(menb_emb, dim=1).values
            # g_embeds_with_exp = torch.median(menb_emb, dim=1).values
            gro_emb = self.lmd * g_embeds_with_avg + group_embeds_full
            gro_emb = gro_emb.to(self.device)
            return gro_emb

    def get_member_view_groupemb_agree(self, gro_inputs, item_inputs):
        ####### group-members-agg #################
        g_embeds_with_attention = torch.zeros([len(gro_inputs), self.embedding_dim]).to(self.device)
        start = time.time()
        user_ids = [self.group_member_dict[usr.item()] for usr in gro_inputs] # [B,1]
        MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 44
        menb_ids, item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
        for i in range(len(user_ids)):
            postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
            menb_ids[i] = user_ids[i] + postfix
            item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
            mask[i] = [1]*len(user_ids[i]) + postfix
        
        menb_ids, item_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                    torch.Tensor(item_ids).long().to(self.device),\
                                    torch.Tensor(mask).float().to(self.device)
        
        menb_emb =  self.u2e(menb_ids) # [B, N, C] 
        # menb_emb =  self.u_aggregator(menb_ids) # [B, N, C]
        menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
        item_emb = self.v2e(item_ids) # [B, N, C] 
        item_emb *= mask.unsqueeze(dim=-1) # [B, N, C]
        ##########################################
        ### Vanilla attention layer 
        ##########################################
        group_item_emb = torch.cat((menb_emb, item_emb), dim=-1) # [B, N, 2C], N=MAX_MENBER_SIZE
        # group_item_emb = group_item_emb.view(-1, group_item_emb.size(-1)) # [B * N, 2C]
        attn_weights = self.attention(group_item_emb)# [B, N, 1]
        # attn_weights = attn_weights.squeeze(dim=-1) # [B, N]
        attn_weights = torch.clip(attn_weights.squeeze(dim=-1), -50, 50)
        attn_weights_exp = attn_weights.exp() * mask # [B, N]
        attn_weights_sm = attn_weights_exp/torch.sum(attn_weights_exp, dim=-1, keepdim=True) # [B, N] 
        attn_weights_sm = attn_weights_sm.unsqueeze(dim=1) # [B, 1, N]
        g_embeds_with_attention = torch.bmm(attn_weights_sm, menb_emb) # [B, 1, C]
        g_embeds_with_attention = g_embeds_with_attention.squeeze(dim=1)
        # print(time.time() - start)
        return g_embeds_with_attention.to(self.device)
    
    def get_member_view_groupemb(self, gro_inputs, item_inputs):
        ####### group-members-agg #################
        g_embeds_with_attention = torch.zeros([len(gro_inputs), self.embedding_dim]).to(self.device)
        start = time.time()
        user_ids = [self.group_member_dict[usr.item()] for usr in gro_inputs] # [B,1]
        MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 44
        menb_ids, item_ids, mask = [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
        for i in range(len(user_ids)):
            postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
            menb_ids[i] = user_ids[i] + postfix
            item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
            mask[i] = [1]*len(user_ids[i]) + postfix
        
        menb_ids, item_ids, mask = torch.Tensor(menb_ids).long().to(self.device),\
                                    torch.Tensor(item_ids).long().to(self.device),\
                                    torch.Tensor(mask).float().to(self.device)
        
        menb_emb =  self.u2e(menb_ids) # [B, N, C] 
        # menb_emb =  self.u_aggregator(menb_ids) # [B, N, C]
        menb_emb *= mask.unsqueeze(dim=-1) # [B, N, C] 
        item_emb = self.v2e(item_ids) # [B, N, C] 
        item_emb *= mask.unsqueeze(dim=-1) # [B, N, C]
        #######################################
        ### Self-attention layer 
        #######################################
        proj_query_emb, proj_key_emb, proj_value_emb = self.self_attention_tuser(menb_emb) # [B, N, C/2], [B, N, C/2], [B, N, C]
        proj_query_emb_new = proj_query_emb * mask.unsqueeze(dim=-1)
        proj_key_emb_new = proj_key_emb * mask.unsqueeze(dim=-1)
        energy = torch.bmm(proj_query_emb_new, proj_key_emb_new.permute(0,2,1)) # [B, N , N]
        
        energy = torch.clip(energy,-50,50)
        energy_exp = energy.exp() * mask.unsqueeze(dim=1)
        energy_exp_softmax = energy_exp/torch.sum(energy_exp, dim=-1, keepdim=True) # [B, N, N]
        # if energy_exp_softmax.shape[1] <= 10:
        #     np.savetxt('D:/Desktop/DSS/ruxia/AGREE_ruxia/Experiments/MaFengWo/SAGREE_trans/energy_exp_softmax', energy_exp_softmax.cpu().detach().numpy()[0], fmt='%1.4f', delimiter=' ') 
        menb_emb_out = torch.bmm(energy_exp_softmax, proj_value_emb) # [B, N, N] * [B, N, C] = [B, N, C]
        menb_emb_out_new = menb_emb_out * mask.unsqueeze(dim=-1)
        overall_menb_out = 0.5 * menb_emb_out_new + 0.5 * menb_emb # [B, N, C]

        if torch.isnan(overall_menb_out.mean()):
            print(overall_menb_out.mean())
            print(energy_exp.mean())
            print(energy.mean())
            quit()


        ######################################
        #### fixed aggregation strategy part ########
        ######################################
        # g_embeds_with_avg = torch.mean(overall_menb_out, dim=1) # [B,C]
        # # # g_embeds_with_lm  = torch.min(overall_menb_out, dim=1).values
        # # # g_embeds_with_ms = torch.max(overall_menb_out, dim=1).values
        # # # g_embeds_with_exp = torch.median(overall_menb_out, dim=1).values
        # g_embeds_with_attention = g_embeds_with_avg

        ##########################################
        ### Vanilla attention layer 
        ##########################################
        group_item_emb = torch.cat((overall_menb_out, item_emb), dim=-1) # [B, N, 2C], N=MAX_MENBER_SIZE
        # group_item_emb = group_item_emb.view(-1, group_item_emb.size(-1)) # [B * N, 2C]
        attn_weights = self.attention(group_item_emb)# [B, N, 1]
        # attn_weights = attn_weights.squeeze(dim=-1) # [B, N]
        attn_weights = torch.clip(attn_weights.squeeze(dim=-1), -50, 50)
        attn_weights_exp = attn_weights.exp() * mask # [B, N]
        attn_weights_sm = attn_weights_exp/torch.sum(attn_weights_exp, dim=-1, keepdim=True) # [B, N] 
        attn_weights_sm = attn_weights_sm.unsqueeze(dim=1) # [B, 1, N]
        g_embeds_with_attention = torch.bmm(attn_weights_sm, overall_menb_out) # [B, 1, C]
        g_embeds_with_attention = g_embeds_with_attention.squeeze(dim=1)
        # print(time.time() - start)
        return g_embeds_with_attention






            






            






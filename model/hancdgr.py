
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import itertools
from model.prediction import PredictLayer, PredDomainLayer
from utils.util import ReverseLayerF

class HANCDGR(nn.Module):
    def __init__(self, enc_u, env_i, enc_gro, enc_u_scr, env_i_scr, device):
        super(HANCDGR, self).__init__()
        self.enc_u = enc_u
        self.env_i = env_i
        self.enc_gro = enc_gro
        self.enc_u_scr = enc_u_scr
        self.env_i_scr = env_i_scr
        self.device = device
        self.embedding_dim_t = enc_u.embedding_dim
        self.embedding_dim_s = enc_u_scr.embedding_dim
        self.drop_ratio = enc_u.drop_ratio
        self.dim_adaption = nn.Linear(self.embedding_dim_s, self.embedding_dim_t)
        self.predictlayer_gro = PredictLayer(3 * self.embedding_dim_t, self.drop_ratio)
        self.predictlayer_user = PredictLayer(3 * self.embedding_dim_t, self.drop_ratio)
        # self.predictlayer_scr = PredictLayer(3 * self.embedding_dim_t, self.drop_ratio)
        self.pred_domain = PredDomainLayer(3 * self.embedding_dim_t)
        # self.pred_domain = PredDomainLayer(self.embedding_dim_t)
        
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, user_inputs, item_inputs, type_a, type_m, source=True, p=1):
        if type_m == 'group':
            # item_embeds_full = self.env_i(user_inputs, item_inputs, type_a)   # [B, C]
            item_embeds_full = self.env_i(item_inputs)   # [B, C]
            # get group preference
            combined = self.enc_gro(user_inputs, item_inputs, type_a) # [B, C]
            # c0_gro = torch.mul(combined, item_embeds_full)   # [B, C]
            # c_gro = torch.sigmoid(self.fcl(c0_gro)) # [B, 1]
            # element_embeds = torch.mul(combined, item_embeds_full)  # Element-wise product
            # new_embeds = torch.cat((element_embeds, combined, item_embeds_full), dim=1)
            # preds_gro = torch.sigmoid(self.predictlayer_gro(new_embeds))
            element_embeds = torch.mul(combined, item_embeds_full)
            preds_gro = torch.sigmoid(element_embeds.sum(1))

            # if torch.isnan(preds_gro.mean()):
            #     print(preds_gro)
            #     print(element_embeds.sum(1))
            #     print(combined.sum(1))
            #     print(item_embeds_full.sum(1))
            #     quit()

            return preds_gro

        elif type_m == 'target_user':
            # item_embeds_full = self.env_i(user_inputs, item_inputs, type_a)                # [B, C]
            item_embeds_full = self.env_i(item_inputs)   # [B, C]
            # get user preference
            combined = self.enc_u(user_inputs, item_inputs, type_a) # [B, C]
            
            # element_embeds = torch.mul(combined, item_embeds_full)  # Element-wise product
            # new_embeds = torch.cat((element_embeds, combined, item_embeds_full), dim=1)
            # preds_user = torch.sigmoid(self.predictlayer_user(new_embeds)).squeeze(-1)

            element_embeds = torch.mul(combined, item_embeds_full)  # Element-wise product
            new_embeds = torch.cat((element_embeds, combined, item_embeds_full), dim=1)
            preds_user = torch.sigmoid(element_embeds.sum(1))

            ##### Adversarial training layer
            domain_output = self.get_adversarial_result(new_embeds, source=source, p=p)
            # domain_output = self.get_adversarial_result(element_embeds, source=source, p=p)
            
            return preds_user, domain_output

        elif type_m == 'source_user':
            # get the target user and item embedding vectors
            user_embeds_scr = self.enc_u_scr(user_inputs)
            item_embeds_scr = self.env_i_scr(item_inputs)

            user_embeds_scr_pie = self.dim_adaption(user_embeds_scr)
            item_embeds_scr_pie = self.dim_adaption(item_embeds_scr)

            # element_embeds_scr = torch.mul(user_embeds_scr_pie, item_embeds_scr_pie).to(self.device)  # Element-wise product
            # new_embeds_scr = torch.cat((element_embeds_scr, user_embeds_scr_pie, item_embeds_scr_pie), dim=1)
            # preds_r_scr = torch.sigmoid(self.predictlayer_user(new_embeds_scr)).squeeze(-1)
            
            element_embeds_scr = torch.mul(user_embeds_scr_pie, item_embeds_scr_pie).to(self.device)
            new_embeds_scr = torch.cat((element_embeds_scr, user_embeds_scr_pie, item_embeds_scr_pie), dim=1)
            preds_r_scr = torch.sigmoid(element_embeds_scr.sum(1))

            # get the binary cross entropy loss between target R users true labels 0 and their predicted domain labels
            domain_output_scr = self.get_adversarial_result(new_embeds_scr, source=source, p=p)
            # domain_output_scr = self.get_adversarial_result(element_embeds_scr, source=source, p=p)
            return preds_r_scr, domain_output_scr


    def get_adversarial_result(self, x, source=True, p=0.0):
            loss_fn = nn.BCELoss()
            # loss_fn = nn.CrossEntropyLoss()
            
            if source:
                domain_label = torch.zeros(len(x)).long().to(self.device)
            else:
                domain_label = torch.ones(len(x)).long().to(self.device)
                
            # get the reversed feature
            x = ReverseLayerF.apply(x, p)

            domain_pred = self.pred_domain(x)
            loss_adv = loss_fn(domain_pred, domain_label.float().unsqueeze(dim=1))
            
            return loss_adv

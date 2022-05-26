import faulthandler; faulthandler.enable()
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import time
import os
from collections import Counter
# import matplotlib.pyplot as plt
import pandas as pd
from model.hancdgr import HANCDGR
from model.User_aggregator import User_aggregator
from model.Group_aggregator import Group_aggregator
from utils.util import Helper, AGREELoss, BPRLoss
from dataset import GDataset
import argparse
dataname = 'MaFengWo'
# dataname = 'CAMRa2011'
# dataname = 'ml-latest-small'
parser = argparse.ArgumentParser()
parser.add_argument('--save_name', type=str, default='newhanablation2')
parser.add_argument('--path', type=str, default='/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HANCDGRv1/data/' + dataname)
parser.add_argument('--user_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HANCDGRv1/data/' + dataname + '/userRating')
parser.add_argument('--group_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HANCDGRv1/data/' + dataname + '/groupRating')
parser.add_argument('--user_in_group_path', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HANCDGRv1/data/' + dataname + '/groupMember.txt')
parser.add_argument('--scr_user_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HANCDGRv1/data/'  + dataname + '/yelpuserRatingTrain.txt')
# parser.add_argument('--scr_user_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HANCDGRv1/data/'  + dataname + '/new_ml_25m_s0.txt')
# parser.add_argument('--scr_user_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/HANCDGRv1/data/'  + dataname + '/ml-25m_filtered_rating.txt')
parser.add_argument('--embedding_size_t_list', type=list, default=[32])
parser.add_argument('--embedding_size_s_list', type=list, default=[32])
parser.add_argument('--n_epoch', type=int, default=500)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--num_negatives', type=list, default=4)
parser.add_argument('--test_num_ng', type=int, default=100)
parser.add_argument('--batch_size_list', type=list, default=[128])
parser.add_argument('--lr', type=float, default=0.002) # ml dataset
# parser.add_argument('--lr', type=float, default=0.000005) # cam dataset
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--drop_ratio_list', type=list, default=[0.2])
parser.add_argument('--topK_list', type=list, default=[5, 10])
# parser.add_argument('--topK_list', type=int, default=5)
parser.add_argument('--type_a', type=str, default='han')
parser.add_argument('--type_m_gro', type=str, default='group')
parser.add_argument('--type_m_user', type=str, default='target_user')
parser.add_argument('--type_m_scr', type=str, default='source_user')
parser.add_argument('--lmd_list', type=list, default=[0, 1]) # 0.05, 0.3,  Group_aggregator 1: M1-a; 0: M1-b
parser.add_argument('--eta_list', type=list, default=[0]) # User_aggregator  1: M2-c; 0: M2-d
parser.add_argument('--gamma_weight_list', type=float, default=[0.002]) # user domain loss , 0.08, 0.1, 0.15
parser.add_argument('--beta_weight_list', type=float, default=[0.2]) # SCR user predict loss , 0.2, 1, 1.5, 2
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')

# train the model
def training_gro(model, train_loader, epoch_id, type_a, type_m_gro):
    # user training
    # optimizer
    # optimizer = optim.RMSprop(model.parameters(), lr)
    optimizer = optim.Adam(model.parameters(), args.lr)

    # loss function
    loss_function = AGREELoss()

    losses = [] 
    model = model.to(DEVICE)    
    for batch_id, (u, pi_ni) in enumerate(train_loader):
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]

        user_input, pos_item_input, neg_item_input = user_input.to(DEVICE), pos_item_input.to(DEVICE), neg_item_input.to(DEVICE)
        # Forward
        
        pos_prediction = model(user_input, pos_item_input, type_a, type_m_gro)
        neg_prediction = model(user_input, neg_item_input, type_a, type_m_gro)
        
        # Zero_grad
        model.zero_grad()
        # Loss value of one batch of examples
        loss = loss_function(pos_prediction, neg_prediction)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward(torch.ones_like(loss))
        # updata parameters
        optimizer.step()

    print('Iteration %d, group train loss is [%.4f ]' % (epoch_id, torch.mean(torch.tensor(losses))))

def training_user(model, train_loader_r, train_loader_scr, epoch_id, type_a, type_m_user, type_m_scr, beta_weight, gamma_weight):
    # user training
    # optimizer = optim.RMSprop(model.parameters(), lr)
    optimizer = optim.Adam(model.parameters(), args.lr)

    # loss function
    loss_function = AGREELoss()
    len_dataloader = min(len(train_loader_r), len(train_loader_scr))

    losses_preds = [] 
    losses_scr = [] 
    losses_domain = []
    losses = []
    i = 1
    model = model.to(DEVICE)   
    for (data_usr, data_scr) in zip(enumerate(train_loader_r), enumerate(train_loader_scr)):
        # Data Load
        _, (usr, pi_ni_usr) = data_usr
        _, (usr_scr, pi_ni_scr) = data_scr

        user_input = usr
        pos_item_input = pi_ni_usr[:, 0]
        neg_item_input = pi_ni_usr[:, 1]

        user_input_scr = usr_scr
        pos_item_input_scr = pi_ni_scr[:, 0]
        neg_item_input_scr = pi_ni_scr[:, 1]


        user_input, pos_item_input, neg_item_input = user_input.to(DEVICE), pos_item_input.to(DEVICE), neg_item_input.to(DEVICE)
        user_input_scr, pos_item_input_scr, neg_item_input_scr = user_input_scr.to(DEVICE), pos_item_input_scr.to(DEVICE), neg_item_input_scr.to(DEVICE)
        
        p = float(i + epoch_id * len_dataloader) / args.n_epoch / len_dataloader
        p = 2. / (1. + np.exp(-10 * p)) - 1

        # Forward
        # target user forward
        pos_preds_usr, pos_domain_output = model(user_input, pos_item_input, type_a, type_m_user, source=False, p=p)
        
        neg_preds_usr, neg_domain_output = model(user_input, neg_item_input, type_a, type_m_user, source=False, p=p)

        # source user forward
        pos_preds_scr, pos_domain_output_scr = model(user_input_scr, pos_item_input_scr, type_a, type_m_scr, source=True, p=p)
        
        neg_preds_scr, neg_domain_output_scr = model(user_input_scr, neg_item_input_scr, type_a, type_m_scr, source=True, p=p)
     
        # Zero_grad
        model.zero_grad()
        # Loss value of one batch of examples
        loss_usr = loss_function(pos_preds_usr, neg_preds_usr)
        loss_scr = loss_function(pos_preds_scr, neg_preds_scr)

        # loss_domain = pos_domain_output_r + neg_domain_output_r + pos_domain_output_scr + neg_domain_output_scr
        loss_domain_target = (pos_domain_output + neg_domain_output)/2
        loss_domain_source = (pos_domain_output_scr + neg_domain_output_scr)/2

        loss_scr =  beta_weight * loss_scr
        loss_preds = loss_usr + loss_scr
        loss_domain = gamma_weight * (loss_domain_target + loss_domain_source)
        loss = loss_preds - loss_domain

        # record loss history
        losses_preds.append(loss_preds)
        losses_scr.append(loss_scr)
        losses_domain.append(loss_domain)

        losses.append(loss)

        # Backward
        loss.backward(torch.ones_like(loss))
        # updata parameters
        optimizer.step()

        i +=1

    print('Iteration %d, loss is [%.4f ], loss_pred is [%.4f ], loss_scr is [%.4f ], loss_domain is [%.4f ]' % (epoch_id, torch.mean(torch.tensor(losses)), torch.mean(torch.tensor(losses_preds)),
    torch.mean(torch.tensor(losses_scr)), torch.mean(torch.tensor(losses_domain)))
    )


def evaluation(model, helper, test_data_loader, K, type_a, type_m, DEVICE):
    model = model.to(DEVICE)
    # set the module in evaluation mode
    model.eval()
    HR, NDCG = helper.metrics(model, test_data_loader, K, type_a, type_m, DEVICE)
    return HR, NDCG


if __name__ == '__main__':
    torch.random.manual_seed(1314)
    # initial helper
    helper = Helper()
    # initial dataSet class
    dataset = GDataset(dataname, args.user_dataset, args.group_dataset, args.scr_user_dataset, args.user_in_group_path, args.num_negatives, args.test_num_ng)
    g_m_d, u_g_d = dataset.gro_members_dict, dataset.user_groups_dict
    u_i_d = dataset.u_items_dict
    # get group number
    num_groups = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    num_users_scr, num_items_scr = dataset.num_users_scr, dataset.num_items_scr
    print('Data prepare is over!')

    # save_name = os.path.basename(args.save_name)
    # dir_name = os.path.dirname(args.path)
    dir_name = os.path.join(args.path, args.save_name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for embedding_size_t in args.embedding_size_t_list:
        for embedding_size_s in args.embedding_size_s_list:
            for beta_weight in args.beta_weight_list:
                for gamma_weight in args.gamma_weight_list:
                    for batch_size in args.batch_size_list:
                        for drop_ratio in args.drop_ratio_list:
                            for lmd in args.lmd_list:
                                for eta in args.eta_list:
                                    for i in range(5):
                                        u2e = nn.Embedding(num_users, embedding_size_t).to(DEVICE)
                                        v2e = nn.Embedding(num_items, embedding_size_t).to(DEVICE)
                                        g2e = nn.Embedding(num_groups, embedding_size_t).to(DEVICE)

                                        u2e_scr = nn.Embedding(num_users_scr, embedding_size_s).to(DEVICE)
                                        v2e_scr = nn.Embedding(num_items_scr, embedding_size_s).to(DEVICE)

                                        enc_u = User_aggregator(u2e, v2e, g2e, embedding_size_t, u_g_d, g_m_d, eta, drop_ratio, DEVICE)
                                        env_i = v2e
                                        enc_gro = Group_aggregator(u2e, v2e, g2e, embedding_size_t, g_m_d, lmd, drop_ratio, DEVICE)

                                        # build HANCDGR model
                                        model = HANCDGR(enc_u, v2e, enc_gro, u2e_scr, v2e_scr, DEVICE).to(DEVICE)

                                        # args information
                                        print("HANCDGR at embedding size_t %d, run Iteration:%d, drop_ratio at %1.2f" %(embedding_size_t, args.n_epoch, drop_ratio))

                                        # train the model
                                        HR_gro = []
                                        NDCG_gro = []
                                        HR_user = []
                                        NDCG_user = []
                                        user_train_time = []
                                        gro_train_time = []
                                        best_hr_gro = 0
                                        best_ndcg_gro = 0
                                        stop = 0
                                        for epoch in range(args.n_epoch):
                                            # set the module in training mode
                                            model.train()
                                            # 开始训练时间
                                            t1_user = time()
                                            # train the user
                                            training_user(model, dataset.get_user_dataloader(batch_size), dataset.get_scr_dataloader(batch_size), epoch, args.type_a, args.type_m_user, args.type_m_scr, beta_weight, gamma_weight)
                                            print("user training time is: [%.1f s]" % (time()-t1_user))
                                            user_train_time.append(time()-t1_user)
                                            
                                            # train the group
                                            t1_gro = time()
                                            training_gro(model, dataset.get_group_dataloader(batch_size), epoch, args.type_a, args.type_m_gro)
                                            print("group training time is: [%.1f s]" % (time()-t1_gro))
                                            gro_train_time.append(time()-t1_gro)
                                            
                                            # evaluation
                                            t2 = time()
                                            u_hr, u_ndcg = evaluation(model, helper, dataset.get_user_test_dataloader(), args.topK_list, args.type_a, args.type_m_user, DEVICE)
                                            HR_user.append(u_hr)
                                            NDCG_user.append(u_ndcg)

                                            t3 = time()
                                            hr, ndcg = evaluation(model, helper, dataset.get_gro_test_dataloader(), args.topK_list, args.type_a, args.type_m_gro, DEVICE)
                                            HR_gro.append(hr)
                                            NDCG_gro.append(ndcg)

                                            if hr[0] > best_hr_gro:
                                                best_hr_gro = hr[0]
                                                best_ndcg_gro = ndcg[0]
                                                stop = 0
                                            else:
                                                stop = stop + 1
                                            print('Test HR_user:', u_hr, '| Test NDCG_user:', u_ndcg)                                                
                                            print('Test HR_gro:', hr, '| Test NDCG_gro:', ndcg)
                                            if stop >= args.early_stop:
                                                print('*' * 20, 'stop training', '*' * 20)                                                                                            
                                                print('Group Iteration %d [%.1f s]: HR_group NDCG_group' % (epoch, time() - t1_user))
                                                print('HR_user:', HR_user[-1], '| NDCG_user:', NDCG_user[-1])
                                                print('Best HR_gro:', HR_gro[-1], '| Best NDCG_gro:', NDCG_gro[-1])
                                                break

                                        
                                        # EVA_user = np.column_stack((HR_user, NDCG_user, user_train_time))
                                        # EVA_gro = np.column_stack((HR_gro, NDCG_gro))

                                        EVA_data = np.column_stack((HR_user, NDCG_user, HR_gro, NDCG_gro))

                                        print("save to file...")

                                        filename = "EVAhan_%s_%s_E%d_beta%1.3f_gamma%1.5f_batch%d_drop_ratio%1.2f_lambda_%1.2f_eta_%1.2f_lr_%1.5f_%d" % (args.type_m_gro, args.type_m_user, embedding_size_t, beta_weight, gamma_weight, batch_size, drop_ratio, lmd, eta, args.lr, i)

                                        filename = os.path.join(dir_name, filename)

                                        np.savetxt(filename, EVA_data, fmt='%1.4f', delimiter=' ')

                                        print("Done!")

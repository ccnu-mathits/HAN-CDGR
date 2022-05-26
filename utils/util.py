import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
import math
import heapq
from collections import defaultdict


# 参考网址 https://github.com/jindongwang/transferlearning/blob/master/code/deep/DANN(RevGrad)/adv_layer.py
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

class AGREELoss(nn.Module):
    def __init__(self):
        super(AGREELoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        
        loss_function = (pos_preds - neg_preds - 1).clone().pow(2)

        loss = loss_function.mean()

        return loss

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
    
    def forward(self, pos_preds, neg_preds):
        # https://github.com/guoyang9/BPR-pytorch/blob/master/main.py
        loss = - (pos_preds - neg_preds).sigmoid().log().mean().clone()
        return loss


class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True

    def hit(self, gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    def ndcg(self, gt_item, pred_items):
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index+2))
        return 0

    def metrics(self, model, test_loader, top_k, type_a, type_m, device):
        p = 0
        hits, ndcgs = [], []
        for users_var, items_var in test_loader:
            users_var = users_var.to(device)
            items_var = items_var.to(device)
            # get the predictions from the trained model
            if (type_a == 'han') and (type_m == 'group'):
                predictions = model(users_var, items_var, 'han', 'group')
            elif (type_a == 'han') and (type_m == 'target_user'):
                predictions, _ = model(users_var, items_var, 'han', 'target_user', source=False, p=p)
            elif (type_a == 'han') and (type_m == 'source_user'):
                predictions, _ = model(users_var, items_var, 'han', 'source_user', source=True, p=p)
        
            elif (type_a == 'fixed_agg') and (type_m == 'group'):
                predictions = model(users_var, items_var, 'fixed_agg', 'group')
            elif (type_a == 'fixed_agg') and (type_m == 'target_user'):
                predictions, _ = model(users_var, items_var, 'fixed_agg', 'target_user', source=False, p=p)

            hr_list, ndcg_list = [], []
            for K in top_k:
                _, indices = torch.topk(predictions, K)
                recommends = torch.take(items_var, indices).cpu().numpy().tolist()

                gt_item = items_var[0].item()
                hr_list.append(self.hit(gt_item, recommends))
                ndcg_list.append(self.ndcg(gt_item, recommends))

            hits.append(hr_list)
            ndcgs.append(ndcg_list)

        return list(np.array(hits).mean(0)), list(np.array(ndcgs).mean(0))


    # The following functions are used to evaluate NCF_trans and group recommendation performance
    def evaluate_model(self, model, testRatings, testNegatives, K, type_a, type_m, device):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []

        for idx in range(len(testRatings)):
            (hr_list,ndcg_list) = self.eval_one_rating(model, testRatings, testNegatives, K, type_a, type_m, idx, device)
            hits.append(hr_list)
            ndcgs.append(ndcg_list)
        return list(np.array(hits).mean(0)), list(np.array(ndcgs).mean(0))


    def eval_one_rating(self, model, testRatings, testNegatives, K, type_a, type_m, idx, device):
        p = 0
        rating = testRatings[idx]
        items = testNegatives[idx]
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        # Get prediction scores
        map_item_score = {}
        users = np.full(len(items), u)

        users_var = torch.from_numpy(users).long().to(device)
        items_var = torch.LongTensor(items).to(device)

        # get the predictions from the trained model
        if (type_a == 'han') and (type_m == 'group'):
            predictions = model(users_var, items_var, 'han', 'group')
        elif (type_a == 'han') and (type_m == 'target_user'):
            predictions, _ = model(users_var, items_var, 'han', 'target_user', source=False, p=p)
        elif (type_a == 'han') and (type_m == 'source_user'):
            predictions, _ = model(users_var, items_var, 'han', 'source_user', source=True, p=p)
    
        elif (type_a == 'fixed_agg') and (type_m == 'group'):
            predictions = model(users_var, items_var, 'fixed_agg', 'group')
        elif (type_a == 'fixed_agg') and (type_m == 'target_user'):
            predictions, _ = model(users_var, items_var, 'fixed_agg', 'target_user', source=False, p=p)

        for i in range(len(items)):
            item = items[i]
            map_item_score[item] = predictions.cpu().data.numpy()[i]
        items.pop() # delete the last item in the list items

        hr_list, ndcg_list = [], []
        for topk in K:
            # Evaluate top rank list
            ranklist = heapq.nlargest(topk, map_item_score, key=map_item_score.get)
            hr = self.getHitRatio(ranklist, gtItem)
            ndcg = self.getNDCG(ranklist, gtItem)
            hr_list.append(hr)
            ndcg_list.append(ndcg)
        return (hr_list, ndcg_list)

    def getHitRatio(self, ranklist, gtItem):
        for item in ranklist:
            if item == gtItem:
                return 1
        return 0

    def getNDCG(self, ranklist, gtItem):
        for i in range(len(ranklist)):
            item = ranklist[i]
            if item == gtItem:
                return math.log(2) / math.log(i+2)
        return 0

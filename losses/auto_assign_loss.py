import torch
from utils.fcos import BoxCoder
from losses.commons import IOULoss
from commons.boxs_utils import box_iou


# class FCOSAutoAssignLoss(object):
#     def __init__(self,
#                  alpha=0.25,
#                  gamma=2.0,
#                  lambda_p=5.0,
#                  temperature=1./3,
#                  strides=None,
#                  iou_type='giou',
#                  positive_weights=0.1,
#                  negative_weights=1.0):
#         self.alpha = alpha
#         self.gamma = gamma
#         self.lambda_p = lambda_p   # 平衡正负样本的损失权重
#         self.temperature = temperature  # 突出具有高置信度的位置
#         # 正负样本损失的权值,论文中并未明确给出,待定
#         self.positive_weights = positive_weights
#         self.negative_weights = negative_weights
#         if strides is None:
#             strides = [8, 16, 32, 64, 128]
#         self.strides = strides
#         self.box_coder = BoxCoder()
#         self.iou_loss_func = IOULoss(iou_type=iou_type, coord_type='ltrb')
#
#     def __call__(self,cls_predicts,box_predicts,implicits,grids,gaussian,targets):
#         '''
#         params
#         :param cls_predicts: list(cls_predict) cls_predict [bs, num_cls, h, w]
#         :param box_predicts: list(box_predict) box_predict [bs, 4, h, w]
#         :param implicits: list(implicit) implicit[bs, 1, h, w]
#         :param grids: list(grid,len=5) grid [h, w, 2]    2==>(xc,yc)原图尺度
#         :param gaussian: [cls, 4]  4==>(ux,uy,theta_x,theta_y)
#         :param targets: [gt, 7] (bs, weights, label_id, x1, y1, x2, y2)
#         :return:
#         '''
#         device = cls_predicts[0].device
#         bs = cls_predicts[0].shape[0]
#         cls_num = cls_predicts[0].shape[1]
#
#         # expand_grid.shape=[grid_num,3]  3==>(xc,yc,stride)
#         expand_grid=torch.cat([
#             torch.cat([
#                 grid_item,torch.tensor(data=stride_item,device=device,dtype=torch.float).expand_as(grid_item[...,[0]])
#             ],dim=-1).view(-1,3) for stride_item,grid_item in zip(self.strides,grids)],dim=0)
#
#         for i in range(len(cls_predicts)):
#             if cls_predicts[i].dtype==torch.float16:
#                 cls_predicts[i]=cls_predicts[i].float()
#         for i in range(len(implicits)):
#             if implicits[i].dtype==torch.float16:
#                 implicits[i]=implicits[i].float()
#
#
#         negative_loss_list=list()
#         positive_loss_list=list()
#
#
#         for bi in range(bs):
#             # batch_cls_predicts [grid_num,cls_num]==>sigmoid
#             batch_cls_predicts = torch.cat(
#                 [cls_item[bi].permute(1, 2, 0).contiguous().view(-1, cls_num) for cls_item in cls_predicts],
#                 dim=0).sigmoid()
#             # batch_implicit [grid_num,1]
#             batch_implicit = torch.cat([
#                 implicit_item[bi].permute(1,2,0).contiguous().view(-1, 1) for implicit_item in implicits
#             ], dim=0).sigmoid()
#             # join_predicts=cls_predicts*implicit_predicts(分类*object)   [grid_num,cls_num]
#             batch_join_predicts=(batch_cls_predicts*batch_implicit).clamp(1e-6,1-1e-6)
#
#
#             # batch_box_predicts [grid_num, 4]
#             batch_box_predicts=torch.cat([
#                 box_item[bi].permute(1,2,0).contiguous().view(-1,4) for box_item in box_predicts], dim=0)
#             # target  [gt_num,6]  6==>(weights, label_id, x1, y1, x2, y2)
#             batch_targets=targets[targets[:,0]==bi,1:]
#
#
#             # 如果没有target,则直接loss= negative focal loss
#             if len(batch_targets) == 0:
#                 negative_loss = -(1 - self.alpha) * (batch_join_predicts ** self.gamma) * (
#                         1 - batch_join_predicts).log()
#                 negative_loss = negative_loss.sum()
#                 negative_loss_list.append(negative_loss)
#                 continue
#
#
#             ############################################################################################################
#             ### clac positive loss -------------------------------------------------------------------------------------
#
#             # [gt_num,6] (weights,label_idx,x1,y1,x2,y2)
#             gt_xy = (batch_targets[:, [2, 3]] + batch_targets[:, [4, 5]]) / 2.
#             # d=(grid_xy-gt_xy) 用来计算centerness weight [grid_num,gt_num,2]
#             xy_offset=(expand_grid[:,None,:2]-gt_xy[None,:,:])/expand_grid[:,None,[2]]
#             # 编码每个grid point的回归目标  [grid_num,gt_num,4]
#             batch_reg_targets = self.box_coder.encode(expand_grid[..., :2], batch_targets[..., 2:])
#             # shape=[1,N]  N=num of positive grid/location 假设所有在gt_box内部的点都是正样本
#             grid_idx,gt_idx=(batch_reg_targets.min(dim=-1)[0]>0).nonzero(as_tuple=False).t()
#
#
#             cls_prob=batch_join_predicts[grid_idx,batch_targets[gt_idx,1].long()]   # shape=[N,1]
#             iou_loss=self.iou_loss_func(batch_box_predicts[grid_idx,:],batch_reg_targets[grid_idx,gt_idx,:])
#             loc_prob= (-self.lambda_p * iou_loss).exp()  # P_loc, shape=[N,1]
#             joint_prob=cls_prob*loc_prob    # P_+=cls_prob*obj_prob ,P(confidence at the location) shape=[N,1]
#             confidence=(joint_prob/self.temperature).exp()   # C(P)  weight_function  shape=[N,1]
#
#             '''
#             G(d)=e{-1*(d-u)**2/(2*theta**2)}
#             d=xy_offset=grid_xy-gt_xy
#             u,theta are learnable parameters.
#             '''
#             gaussian_delta_mu = -(
#                     (xy_offset[grid_idx, gt_idx, :] - gaussian[batch_targets[gt_idx, 1].long(), :2]) ** 2
#             ).sum(-1)
#             gaussian_delta_theta = 2 * ((gaussian[batch_targets[gt_idx, 1].long(), 2:]) ** 2).sum(-1)
#             gaussian_weights = (gaussian_delta_mu / gaussian_delta_theta).exp()  # shape=[N,1]
#
#             # w+
#             positive_weights = confidence * gaussian_weights   # shape=[N,1]
#             positive_loss = torch.tensor(data=0., device=device)
#             for unique_gt_idx in gt_idx.unique():
#                 gt_idx_mask=gt_idx==unique_gt_idx
#                 instance_weights=positive_weights[gt_idx_mask]/positive_weights[gt_idx_mask].sum()
#                 instance_loss=-(instance_weights*joint_prob[gt_idx_mask]).sum().log()
#                 positive_loss+=instance_loss
#             positive_loss_list.append(positive_loss)
#
#
#             ##########################################################################################################################
#             ## calc negative loss ----------------------------------------------------------------------------------------------------
#             decode_box=self.box_coder.decoder(expand_grid[...,:2],batch_box_predicts).detach()  # shape=[grid_num,4]  4==>(x1,y1,x2,y2)
#             predict_targets_iou = box_iou(decode_box, batch_targets[..., 2:])  # shape=[grid_num,gt_num]
#             '''
#             max_iou=max{iou between the predicted_box and all gt_boxes}
#             '''
#             max_iou, max_iou_gt_idx = predict_targets_iou.max(dim=-1)   # shape=[grid_num]
#             func_iou = 1 / (1 - max_iou)
#             func_iou = 1 - (func_iou - 1) / (func_iou.max() - 1 + 1e-10)   # max_iou==>(0,1) if max_iou=1, func_iou=0.  if max_iou=0, func_iou=1.
#             # 任何gt_box区域之外的点w-=1.0
#             negative_weights=torch.ones(size=(expand_grid.shape[0],cls_num),device=device).float()   # shape=[grid_num, cls_num]
#             negative_weights[grid_idx,batch_targets[gt_idx,1].long()]=func_iou[grid_idx]
#             weighted_negative_prob = negative_weights * batch_join_predicts
#             negative_loss = -(1 - self.alpha) * (weighted_negative_prob ** self.gamma) * (
#                     1 - weighted_negative_prob).log()
#             negative_loss = negative_loss.sum()
#             negative_loss_list.append(negative_loss)
#
#         total_negative_loss = torch.stack(negative_loss_list).sum() / max(1, len(targets))
#         if len(targets) == 0:
#             return total_negative_loss, torch.stack([total_negative_loss, torch.tensor(0., device=device)]).detach(), len(targets)
#         total_positive_loss = torch.stack(positive_loss_list).sum() / max(1, len(targets))
#         total_negative_loss = total_negative_loss * self.negative_weights
#         total_positive_loss = total_positive_loss * self.positive_weights
#         total_loss = total_negative_loss + total_positive_loss
#         return total_loss, torch.stack([total_negative_loss, total_positive_loss]).detach(), len(targets)















class FCOSAutoAssignLoss(object):
    def __init__(self,
                 alpha=0.25,
                 gamma=2.0,
                 lambda_p=5.0,
                 temperature=1./3,
                 strides=None,
                 iou_type='giou',
                 positive_weights=0.1,
                 negative_weights=1.0):
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_p = lambda_p   # 平衡正负样本的损失权重
        self.temperature = temperature  # 突出具有高置信度的位置
        # 正负样本损失的权值,论文中并未明确给出,待定
        self.positive_weights = positive_weights
        self.negative_weights = negative_weights
        if strides is None:
            strides = [8, 16, 32, 64, 128]
        self.strides = strides
        self.box_coder = BoxCoder()
        self.iou_loss_func = IOULoss(iou_type=iou_type, coord_type='ltrb')

    def __call__(self,cls_predicts,box_predicts,implicits,grids,gaussian,targets):
        '''
        params
        :param cls_predicts: list(cls_predict) cls_predict [bs, num_cls, h, w]
        :param box_predicts: list(box_predict) box_predict [bs, 4, h, w]
        :param implicits: list(implicit) implicit[bs, 1, h, w]
        :param grids: list(grid,len=5) grid [h, w, 2]    2==>(xc,yc)原图尺度
        :param gaussian: [cls, 4]  4==>(ux,uy,theta_x,theta_y)
        :param targets: [gt, 7] (bs, weights, label_id, x1, y1, x2, y2)
        :return:
        '''
        device = cls_predicts[0].device
        bs = cls_predicts[0].shape[0]
        cls_num = cls_predicts[0].shape[1]

        # expand_grid.shape=[grid_num,3]  3==>(xc,yc,stride)
        expand_grid=torch.cat([
            torch.cat([
                grid_item,torch.tensor(data=stride_item,device=device,dtype=torch.float).expand_as(grid_item[...,[0]])
            ],dim=-1).view(-1,3) for stride_item,grid_item in zip(self.strides,grids)],dim=0)

        for i in range(len(cls_predicts)):
            if cls_predicts[i].dtype==torch.float16:
                cls_predicts[i]=cls_predicts[i].float()
        for i in range(len(implicits)):
            if implicits[i].dtype==torch.float16:
                implicits[i]=implicits[i].float()


        negative_loss_list=list()
        positive_loss_list=list()
        num_neg_grids=0



        for bi in range(bs):
            # batch_cls_predicts [grid_num,cls_num]==>sigmoid
            batch_cls_predicts = torch.cat(
                [cls_item[bi].permute(1, 2, 0).contiguous().view(-1, cls_num) for cls_item in cls_predicts],
                dim=0).sigmoid()
            # batch_implicit [grid_num,1]
            batch_implicit = torch.cat([
                implicit_item[bi].permute(1,2,0).contiguous().view(-1, 1) for implicit_item in implicits
            ], dim=0).sigmoid()
            # join_predicts=cls_predicts*implicit_predicts(分类*object)   [grid_num,cls_num]
            batch_join_predicts=(batch_cls_predicts*batch_implicit).clamp(1e-6,1-1e-6)


            # batch_box_predicts [grid_num, 4]
            batch_box_predicts=torch.cat([
                box_item[bi].permute(1,2,0).contiguous().view(-1,4) for box_item in box_predicts], dim=0)
            # target  [gt_num,6]  6==>(weights, label_id, x1, y1, x2, y2)
            batch_targets=targets[targets[:,0]==bi,1:]


            # 如果没有target,则直接loss= negative focal loss
            if len(batch_targets) == 0:
                negative_loss = -1 * (batch_join_predicts ** self.gamma) * (
                        1 - batch_join_predicts).log()
                negative_loss = negative_loss.sum()
                negative_loss_list.append(negative_loss)
                continue


            ############################################################################################################
            ### clac positive loss -------------------------------------------------------------------------------------

            # [gt_num,6] (weights,label_idx,x1,y1,x2,y2)
            gt_xy = (batch_targets[:, [2, 3]] + batch_targets[:, [4, 5]]) / 2.
            # d=(grid_xy-gt_xy) 用来计算centerness weight [grid_num,gt_num,2]
            xy_offset=(expand_grid[:,None,:2]-gt_xy[None,:,:])/expand_grid[:,None,[2]]
            # 编码每个grid point的回归目标  [grid_num,gt_num,4]
            batch_reg_targets = self.box_coder.encode(expand_grid[..., :2], batch_targets[..., 2:])
            # shape=[1,N]  N=num of positive grid/location 假设所有在gt_box内部的点都是正样本
            grid_idx,gt_idx=(batch_reg_targets.min(dim=-1)[0]>0).nonzero(as_tuple=False).t()

            # debug
            num_neg_grids+=grid_idx.shape[0]


            cls_prob=batch_join_predicts[grid_idx,batch_targets[gt_idx,1].long()]   # shape=[N,1]
            iou_loss=self.iou_loss_func(batch_box_predicts[grid_idx,:],batch_reg_targets[grid_idx,gt_idx,:])
            loc_prob= (-self.lambda_p * iou_loss).exp()  # P_loc, shape=[N,1]
            joint_prob=cls_prob*loc_prob    # P_+=cls_prob*obj_prob ,P(confidence at the location) shape=[N,1]
            confidence=(joint_prob/self.temperature).exp()   # C(P)  weight_function  shape=[N,1]

            '''
            G(d)=e{-1*(d-u)**2/(2*theta**2)}
            d=xy_offset=grid_xy-gt_xy
            u,theta are learnable parameters.
            '''
            gaussian_delta_mu = -(
                    (xy_offset[grid_idx, gt_idx, :] - gaussian[batch_targets[gt_idx, 1].long(), :2]) ** 2
            ).sum(-1)
            gaussian_delta_theta = 2 * ((gaussian[batch_targets[gt_idx, 1].long(), 2:]) ** 2).sum(-1)
            gaussian_weights = (gaussian_delta_mu / gaussian_delta_theta).exp()  # shape=[N,1]

            # w+
            positive_weights = confidence * gaussian_weights   # shape=[N,1]
            positive_loss = torch.tensor(data=0., device=device)
            for unique_gt_idx in gt_idx.unique():
                gt_idx_mask=gt_idx==unique_gt_idx
                instance_weights=positive_weights[gt_idx_mask]/positive_weights[gt_idx_mask].sum()
                instance_loss=-(instance_weights*joint_prob[gt_idx_mask]).sum().log()
                positive_loss+=instance_loss
            positive_loss_list.append(positive_loss)


            ##########################################################################################################################
            ## calc negative loss ----------------------------------------------------------------------------------------------------
            decode_box=self.box_coder.decoder(expand_grid[...,:2],batch_box_predicts).detach()  # shape=[grid_num,4]  4==>(x1,y1,x2,y2)
            predict_targets_iou = box_iou(decode_box, batch_targets[..., 2:])  # shape=[grid_num,gt_num]
            '''
            max_iou=max{iou between the predicted_box and all gt_boxes}
            '''
            max_iou, max_iou_gt_idx = predict_targets_iou.max(dim=-1)   # shape=[grid_num]
            func_iou = 1 / (1 - max_iou)
            func_iou = 1 - (func_iou - 1) / (func_iou.max() - 1 + 1e-10)   # max_iou==>(0,1) if max_iou=1, func_iou=0.  if max_iou=0, func_iou=1.

            # 任何gt_box区域之外的点w-=1.0
            negative_weights=torch.ones(size=(expand_grid.shape[0],cls_num),device=device).float()   # shape=[grid_num, cls_num]
            negative_weights[grid_idx,batch_targets[gt_idx,1].long()]=func_iou[grid_idx]
            weighted_negative_prob = negative_weights * batch_join_predicts
            negative_loss = -1 * (weighted_negative_prob ** self.gamma) * (
                    1 - weighted_negative_prob).log()
            negative_loss = negative_loss.sum()
            negative_loss_list.append(negative_loss)

        total_negative_loss = torch.stack(negative_loss_list).sum() / max(1, len(targets))
        # total_negative_loss = torch.stack(negative_loss_list).sum() / num_neg_grids
        if len(targets) == 0:
            return total_negative_loss, torch.stack([total_negative_loss, torch.tensor(0., device=device)]).detach(), len(targets)
        total_positive_loss = torch.stack(positive_loss_list).sum() / max(1, len(targets))
        total_negative_loss = total_negative_loss * (1-self.alpha)
        total_positive_loss = total_positive_loss * self.alpha
        total_loss = total_negative_loss + total_positive_loss
        return total_loss, torch.stack([total_negative_loss, total_positive_loss]).detach(), len(targets)






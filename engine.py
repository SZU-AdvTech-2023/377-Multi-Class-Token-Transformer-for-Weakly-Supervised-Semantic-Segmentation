import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import utils
from sklearn.metrics import average_precision_score
import numpy as np
import cv2
import os
from pathlib import Path
from PAR import *
from models import BCEWithLogitsLoss2d


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True,args=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets, img_box in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        patch_outputs = None
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            if not isinstance(outputs, torch.Tensor):
                outputs, patch_outputs, class_embeddings, patch_embeddings = outputs

            loss = F.multilabel_soft_margin_loss(outputs, targets)
            metric_logger.update(cls_loss=loss.item())
            if class_embeddings is not None:      # class_embeddings torch.Size([12, 64, 20, 384])
                # print('class_embeddings',class_embeddings.shape)
                class_embeddings = class_embeddings[-args.num_cct:]   # class_embeddings torch.Size([12, 64, 20, 384])
                output_cls_embeddings = F.normalize(class_embeddings, dim=-1)  # 12xBxCxD   output_cls_embeddings torch.Size([12, 64, 20, 384])
                scores = output_cls_embeddings @ output_cls_embeddings.permute(0, 1, 3, 2)  # 12xBxCxC  scores torch.Size([12, 64, 20, 20])
                ground_truth = torch.arange(targets.size(-1), dtype=torch.long, device=device)  # C   ground_truth torch.Size([20])
                ground_truth = ground_truth.unsqueeze(0).unsqueeze(0).expand(class_embeddings.shape[0], class_embeddings.shape[1],
                                                                             class_embeddings.shape[2])  # 12xBxC   ground_truth torch.Size([12, 64, 20])
                regularizer_loss = torch.nn.CrossEntropyLoss(reduction='none')(scores.permute(1, 2, 3, 0),
                                                                               ground_truth.permute(1, 2, 0))  # BxCx12
                regularizer_loss = torch.mean(
                    torch.mean(torch.sum(regularizer_loss * targets.unsqueeze(-1), dim=-2), dim=-1) / (
                                torch.sum(targets, dim=-1) + 1e-8))
                metric_logger.update(attn_loss=regularizer_loss.item())
                loss = loss + args.loss_weight*regularizer_loss
            # if patch_embeddings is not None:    # patch_embeddings torch.Size([12, 64, 196, 384])
            #     par = PAR(num_iter=10, dilations=[1,2,4,8,12,24])
            #     par.to(device)
            #     inputs_denorm = utils.denormalize_img2(samples.clone())
            #     refine_cams = utils.refine_cams_with_bkg_v2(par, inputs_denorm, cams=refine_cams, cls_labels=targets, cfg=args, img_box=img_box, down_scale=16)
            #     attn_mask = utils.get_mask_by_radius(h=14, w=14, radius=args.radius)   #attn_mask (196, 196)
            #     aff_label = utils.cams_to_affinity_label(refine_cams, mask=attn_mask, ignore_index=args.ignore_index) # aff_label torch.Size([64, 196, 196])
            #     patch_embeddings = patch_embeddings[-args.num_cct:]    # patch_embeddings torch.Size([12, 64, 196, 384])
            #     output_patch_embeddings = F.normalize(patch_embeddings, dim=-1)   # output_patch_embeddings torch.Size([12, 64, 196, 384])
            #     # scores_patch = output_patch_embeddings @ output_patch_embeddings.permute(0, 1, 3, 2)   # scores_patch torch.Size([12, 64, 196, 196])
            #     aff_loss, pos_count, neg_count = get_aff_loss(output_patch_embeddings, aff_label)
            #     metric_logger.update(aff_loss=aff_loss.item())
            #     loss = loss + aff_loss

            if  patch_outputs is not None:
                ploss = F.multilabel_soft_margin_loss(patch_outputs, targets)
                metric_logger.update(pat_loss=ploss.item())
                loss = loss + ploss
            #loss_D.backward()
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target,img_box in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        img_box = img_box.to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast():
            output = model(images)
            if not isinstance(output, torch.Tensor):
                output, patch_output, class_embeddings, patch_embeddings = output
            loss = criterion(output, target)
            output = torch.sigmoid(output)


            mAP_list = compute_mAP(target, output)
            mAP = mAP + mAP_list
            metric_logger.meters['mAP'].update(np.mean(mAP_list), n=batch_size)


        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* mAP {mAP.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(mAP=metric_logger.mAP, losses=metric_logger.loss))


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count

def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) > 0:
            ap_i = average_precision_score(y_true[i], y_pred[i])
            AP.append(ap_i)
            # print(ap_i)
    return AP


@torch.no_grad()
def generate_attention_maps_ms(data_loader, model, get_att, device, args):
    metric_logger = utils.MetricLogger(delimiter="  ")  #output-class
    header = 'Generating attention maps:'
    if args.attention_dir is not None:
        Path(args.attention_dir).mkdir(parents=True, exist_ok=True)   #Create files based on the path
    if args.cam_npy_dir is not None:
        Path(args.cam_npy_dir).mkdir(parents=True, exist_ok=True)

    # switch to evaluation mode
    model.eval()

    img_list = open(os.path.join(args.img_list, 'train_aug_id.txt')).readlines()
    index = 0
    for image_list, target in metric_logger.log_every(data_loader, 10, header):
    # for iter, (image_list, target) in enumerate(data_loader):
        images1 = image_list[0].to(device, non_blocking=True)

        # 输入数据改成 224
        # images1 = F.interpolate(images1, size=(224,224))
        target = target.to(device, non_blocking=True)
        batch_size = images1.shape[0]
        img_name = img_list[index].strip()
        index += 1

        img_temp = images1.permute(0, 2, 3, 1).detach().cpu().numpy()
        orig_images = np.zeros_like(img_temp)
        orig_images[:, :, :, 0] = (img_temp[:, :, :, 0] * 0.229 + 0.485) * 255.
        orig_images[:, :, :, 1] = (img_temp[:, :, :, 1] * 0.224 + 0.456) * 255.
        orig_images[:, :, :, 2] = (img_temp[:, :, :, 2] * 0.225 + 0.406) * 255.

        w_orig, h_orig = orig_images.shape[1], orig_images.shape[2]
        # w, h = images1.shape[2] - images1.shape[2] % args.patch_size, images1.shape[3] - images1.shape[3] % args.patch_size
        # w_featmap = w // args.patch_size
        # h_featmap = h // args.patch_size


        with torch.cuda.amp.autocast():
            cam_list = []
            vitattn_list = []
            cam_maps = None
            for s in range(len(image_list)):
                images = image_list[s].to(device, non_blocking=True)

                w, h = images.shape[2] - images.shape[2] % args.patch_size, images.shape[3] - images.shape[3] % args.patch_size
                w_featmap = w // args.patch_size
                h_featmap = h // args.patch_size

                if 'MCTformerV1' in args.model:
                    output, cls_attentions, patch_attn = model(images, return_att=True, n_layers=args.layer_index)
                    cls_attentions = cls_attentions.reshape(batch_size, args.nb_classes, w_featmap, h_featmap)
                    patch_attn = torch.sum(patch_attn, dim=0)

                elif 'MCTformerV2' in args.model:
                    output, cls_attentions, patch_attn= model(images, return_att=True,get_att= get_att, n_layers=args.layer_index, attention_type=args.attention_type)
                    patch_attn = torch.sum(patch_attn, dim=0)

                if args.patch_attn_refine:
                    # b = cls_attentions
                    # b11= cls_attentions.shape[0]
                    # b12= cls_attentions.shape[1]
                    # b13 = cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)
                    # b2 = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1))
                    cls_attentions = torch.matmul(patch_attn.unsqueeze(1), cls_attentions.view(cls_attentions.shape[0],cls_attentions.shape[1], -1, 1)).reshape(cls_attentions.shape[0],cls_attentions.shape[1], w_featmap, h_featmap)
                cls_attentions = F.interpolate(cls_attentions, size=(w_orig, h_orig), mode='bilinear', align_corners=False)[0]
                cls_attentions = cls_attentions.cpu().numpy() * target.clone().view(args.nb_classes, 1, 1).cpu().numpy()

                if s % 2 == 1:
                    cls_attentions = np.flip(cls_attentions, axis=-1)
                cam_list.append(cls_attentions)
                vitattn_list.append(cam_maps)

            sum_cam = np.sum(cam_list, axis=0)
            sum_cam = torch.from_numpy(sum_cam)
            sum_cam = sum_cam.unsqueeze(0).to(device)

            output = torch.sigmoid(output)

        if args.visualize_cls_attn:
            for b in range(images.shape[0]):
                if (target[b].sum()) > 0:
                    cam_dict = {}
                    for cls_ind in range(args.nb_classes):
                        if target[b,cls_ind]>0:
                            cls_score = format(output[b, cls_ind].cpu().numpy(), '.3f')
                            cls_attention = sum_cam[b,cls_ind,:]
                            cls_attention = (cls_attention - cls_attention.min()) / (cls_attention.max() - cls_attention.min() + 1e-8)
                            cls_attention = cls_attention.cpu().numpy()
                            cam_dict[cls_ind] = cls_attention

                            if args.attention_dir is not None:
                                fname = os.path.join(args.attention_dir, img_name + '_' + str(cls_ind) + '_' + str(cls_score) + '.png')
                                show_cam_on_image(orig_images[b], cls_attention, fname)

                    if args.cam_npy_dir is not None:
                        np.save(os.path.join(args.cam_npy_dir, img_name + '.npy'), cam_dict)

                    if args.out_crf is not None:
                        for t in [args.low_alpha, args.high_alpha]:
                            orig_image = orig_images[b].astype(np.uint8).copy(order='C')
                            crf = _crf_with_alpha(cam_dict, t, orig_image)
                            folder = args.out_crf + ('_%s' % t)
                            if not os.path.exists(folder):
                                os.makedirs(folder)
                            np.save(os.path.join(folder, img_name + '.npy'), crf)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from psa.tool.imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al


def show_cam_on_image(img, mask, save_path):
    img = np.float32(img) / 255.
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    cv2.imwrite(save_path, cam)

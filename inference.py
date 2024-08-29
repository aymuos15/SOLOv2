import os
from glob import glob

from modules.solov2 import SOLOV2

from scipy import ndimage
import torch
import numpy as np
import cv2 as cv

from data.compose import Compose
from data.config import cfg, process_funcs_dict
from data.imgutils import imresize

import pycocotools.mask as mask_util

import warnings
warnings.filterwarnings("ignore")

home = '/home/localssk23/SOLOv2/'

COCO_LABEL = cfg.label
COCO_LABEL_MAP = cfg.label_map
COCO_CLASSES = cfg.class_names
CLASS_NAMES=(COCO_CLASSES, COCO_LABEL)

def get_masks(result, num_classes=80):
    for cur_result in result:
        masks = [[] for _ in range(num_classes)]
        if cur_result is None:
            return masks
        seg_pred = cur_result[0].cpu().numpy().astype(np.uint8)
        cate_label = cur_result[1].cpu().numpy().astype(np.int)
        cate_score = cur_result[2].cpu().numpy().astype(np.float)
        num_ins = seg_pred.shape[0]
        for idx in range(num_ins):
            cur_mask = seg_pred[idx, ...]
            rle = mask_util.encode(
                np.array(cur_mask[:, :, np.newaxis], order='F'))[0]
            rst = (rle, cate_score[idx])
            masks[cate_label[idx]].append(rst)

        return masks

#set requires_grad False
def gradinator(x):
    x.requires_grad = False
    return x

def build_process_pipeline(pipeline_confg):
    assert isinstance(pipeline_confg, list)
    process_pipelines = []
    for pipconfig in pipeline_confg:
        assert isinstance(pipconfig, dict) and 'type' in pipconfig
        args = pipconfig.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            process_pipelines.append(process_funcs_dict[obj_type](**args))
            
    return process_pipelines

class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None 
        img = cv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

class LoadImageInfo(object):
    def __call__(self, frame):
        results={}
        results['filename'] = None 
        results['img'] = frame
        results['img_shape'] = frame.shape
        results['ori_shape'] = frame.shape
        return results


def show_result_ins(img,
                    result,
                    score_thr=0.3,
                    sort_by_density=False):
    if isinstance(img, str):
        img = cv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]

    for idx in range(num_mask):
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        cur_cate = cate_label[idx]
        realclass = COCO_LABEL[cur_cate]
        cur_score = cate_score[idx]

        name_idx = COCO_LABEL_MAP[realclass]
        label_text = COCO_CLASSES[name_idx-1]
        label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv.putText(img_show, label_text, vis_pos,
                        cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
 
    return img_show


def eval(valmodel_weight, data_path, test_mode, save_imgs=False):
    
    test_pipeline = []
    transforms=[ dict(type='Resize', keep_ratio=True),
                dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='TestCollect', keys=['img']),
    ]
    transforms_piplines = build_process_pipeline(transforms)
    Multest = process_funcs_dict['MultiScaleFlipAug'](transforms = transforms_piplines, img_scale = (480, 448), flip=False)

    if test_mode == "video":
        test_pipeline.append(LoadImageInfo())
    elif test_mode == "images":
        test_pipeline.append(LoadImage())
    else:
        raise NotImplementedError("not support mode!")
    test_pipeline.append(Multest)
    test_pipeline = Compose(test_pipeline)

    model = SOLOV2(cfg, pretrained=valmodel_weight, mode='test')
    model = model.cuda()

    if test_mode == "images":
        img_ids = []
        images = []
        test_imgpath = data_path

        test_imgpath = test_imgpath + '/*'
        images = glob(test_imgpath)
        for img in images:
            pathname, filename = os.path.split(img)
            prefix, suffix = os.path.splitext(filename)
            img_id = int(prefix)
            img_ids.append(str(img_id))  

        k = 0
        for imgpath in images:
            img_id = img_ids[k]
            data = dict(img=imgpath)
            data = test_pipeline(data)
            imgs = data['img']

            img = imgs[0].cuda().unsqueeze(0)
            img_info = data['img_metas']
            with torch.no_grad():
                seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)
            img_show = show_result_ins(imgpath,seg_result)

            save_path = f"{home}results/" + os.path.basename(imgpath)

            k = k + 1
            if save_imgs:
                print("save image: ", save_path)
                cv.imwrite(save_path, img_show)

# eval(valmodel_weight=f'{home}pretrained/solov2_448_r18_epoch_36.pth',data_path=f"{home}data/casia-SPT_val/val/JPEGImages", test_mode="images", save_imgs=True)
eval(valmodel_weight=f'{home}weights/solov2_resnet18_epoch_30.pth',data_path=f"{home}datasets/casia-SPT_val/val/JPEGImages", test_mode="images", save_imgs=True)
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad

from data.config import cfg, process_funcs_dict
from data.coco import CocoDataset
from data.loader import build_dataloader

from modules.solov2 import SOLOV2

from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

def clip_grads(params):
    params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return clip_grad.clip_grad_norm_(params, max_norm=35, norm_type=2)

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

#set requires_grad False
def gradinator(x):
    x.requires_grad = False
    return x

#Better Logging for all:)
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def log_progress(iter_nums, base_loop, total_epochs, j, loss_sum, loss_ins, loss_cate, cur_lr, log_interval=50):
    if j % log_interval == 0 and j != 0:
        
        epoch_progress = f"Epoch: [{iter_nums + base_loop}/{total_epochs}]"
        
        losses = {
            "Total": loss_sum / log_interval,
            "Instance": loss_ins / log_interval,
            "Category": loss_cate / log_interval
        }
        
        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
        
        print(f"{epoch_progress})")
        print(f"Losses - {loss_str}")
        print(f"Learning rate: {cur_lr:.5f}")
        print("-" * 50)

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


def get_warmup_lr(cur_iters, warmup_iters, bash_lr, warmup_ratio, warmup='linear'):
    if warmup == 'constant':
        warmup_lr = bash_lr * warmup_ratio 
    elif warmup == 'linear':
        k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
        warmup_lr = bash_lr * (1 - k)
    elif warmup == 'exp':
        k = warmup_ratio**(1 - cur_iters / warmup_iters)
        warmup_lr = bash_lr * k
    return warmup_lr


def train(epoch_iters = 1, total_epochs = 10):
    
    #train process pipelines func
    transforms_piplines = build_process_pipeline(cfg.train_pipeline)

    # #build datashet
    casiadata = CocoDataset(ann_file=cfg.dataset.train_info,
                            pipeline = transforms_piplines,
                            img_prefix = cfg.dataset.trainimg_prefix,
                            data_root=cfg.dataset.train_prefix)
    
    torchdata_loader = build_dataloader(casiadata, cfg.imgs_per_gpu, cfg.workers_per_gpu, num_gpus=cfg.num_gpus, shuffle=True)
    
    #todo: Add Checkpointing for training
    model = SOLOV2(cfg, pretrained=None, mode='train')
    model = model.cuda()
    model = model.train()

    optimizer_config = cfg.optimizer
    optimizer = optim.SGD(model.parameters(), lr=optimizer_config['lr'], momentum=optimizer_config['momentum'], weight_decay=optimizer_config['weight_decay'])

    #epoch has trained times, start loop times
    base_loop = epoch_iters

    #left epoch need traind times
    left_loops = total_epochs - base_loop + 1

    #all left iter nums
    total_nums = left_loops * total_epochs
    left_nums = total_nums
    base_nums = (base_loop - 1)* total_epochs

    loss_sum = 0.0 
    loss_ins = 0.0 
    loss_cate = 0.0

    base_lr = optimizer_config['lr']
    cur_lr = base_lr
    cur_nums = 0   

    print()
    print("Start training...")
    print()
 
    try:
        for iter_nums in range(left_loops):

            #every epoch set lr
            epoch_iters = iter_nums + base_loop
            if epoch_iters < cfg.lr_config['step'][0]:
                set_lr(optimizer, 0.01)
                base_lr = 0.01
                cur_lr = 0.01
            elif epoch_iters >= cfg.lr_config['step'][0] and epoch_iters < cfg.lr_config['step'][1]:
                set_lr(optimizer, 0.001)
                base_lr = 0.001
                cur_lr = 0.001
            elif epoch_iters >=  cfg.lr_config['step'][1] and epoch_iters <= total_epochs:
                set_lr(optimizer, 0.0001)
                base_lr = 0.0001
                cur_lr = 0.0001
            else:
                raise NotImplementedError("train epoch is done!")
           

            for j, data in enumerate(torchdata_loader):

                if cfg.lr_config['warmup'] is not None and base_nums < cfg.lr_config['warmup_iters']:
                    warm_lr = get_warmup_lr(base_nums, cfg.lr_config['warmup_iters'],
                                            optimizer_config['lr'], cfg.lr_config['warmup_ratio'],
                                            cfg.lr_config['warmup'])
                    set_lr(optimizer, warm_lr)
                    cur_lr = warm_lr
                else:
                    set_lr(optimizer, base_lr)
                    cur_lr = base_lr
                
                imgs = gradinator(data['img'].data[0].cuda())
                img_meta = data['img_metas'].data[0]
                gt_bboxes = []
                for bbox in data['gt_bboxes'].data[0]:
                    bbox = gradinator(bbox.cuda())
                    gt_bboxes.append(bbox)
                
                gt_masks = data['gt_masks'].data[0]  #cpu numpy data
                
                gt_labels = []
                for label in data['gt_labels'].data[0]:
                    label = gradinator(label.cuda())
                    gt_labels.append(label)
                
      
                loss = model.forward(img=imgs,
                        img_meta=img_meta,
                        gt_bboxes=gt_bboxes,
                        gt_labels=gt_labels,
                        gt_masks=gt_masks)


                losses = loss['loss_ins'] + loss['loss_cate']
                loss_sum = loss_sum + losses.cpu().item()
                loss_ins = loss_ins + loss['loss_ins'].cpu().item()
                loss_cate = loss_cate + loss['loss_cate'].cpu().item()

                optimizer.zero_grad()
                losses.backward()

                if torch.isfinite(losses).item():
                    optimizer.step()
                else:
                    NotImplementedError("loss type error!can't backward!")

                left_nums = left_nums - 1
                base_nums = base_nums + 1
                cur_nums = cur_nums + 1

                #ervery iter 50 times, print some logger
                if j%50 == 0 and j != 0:

                    log_progress(iter_nums, base_loop, total_epochs, j, loss_sum, loss_ins, loss_cate, cur_lr)

                    loss_sum = 0.0 
                    loss_ins = 0.0 
                    loss_cate = 0.0
                    
            left_loops = left_loops -1
            save_name = "./weights/solov2_" + cfg.backbone.name + "_epoch_" + str(iter_nums + base_loop) + ".pth"
            model.save_weights(save_name)        

    except KeyboardInterrupt:
        save_name = "./weights/solov2_" + cfg.backbone.name + "_epoch_" + str(total_epochs-left_loops) + "interrupt.pth"
        model.save_weights(save_name)      

if __name__ == '__main__':
    # train(epoch_iters=cfg.epoch_iters_start, total_epochs = cfg.total_epoch) 
    train(epoch_iters=1, total_epochs = 30)
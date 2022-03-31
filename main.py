#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/5/27 11:27
# @Author : Hw-Zhao
# @Site : 
# @File : main.py
# @Software: PyCharm

import os
import time
from tqdm import tqdm
import torch
# from config.paviaU_config import config
# from config.IGRSS2013_config import config
from config.config_Salinas import config
# from data.dataloader import NewPaviaLoader
# from data.dataloader import NewGrss2013Loader
from data.dataloader import NewSalinasLoader
from loss_function import loss as loss_function
from model.freenet import FreeNet
from util.class_2_rgb import classmap_2_rgbmap
from util.metric import confusion_matrix, evaluate_metric


def fcn_evaluate_fn(model: FreeNet, test_dataloader, device, config, path):
    model.eval()
    oa = 0
    with torch.no_grad():
        for (im, mask, w) in test_dataloader:
            im = im.to(device)
            mask = mask.to(device)
            w = w.to(device)
            a = w.cpu().numpy()
            target = torch.softmax(model(im), dim=1).squeeze()
            target = target.argmax(dim=0) + torch.tensor(1, dtype=torch.int)

            img = target.cpu().numpy()[:config['image_size'][0], :config['image_size'][1]]
            rgb = classmap_2_rgbmap(img, config['palette'])
            rgb.save(path)

            w.unsqueeze_(dim=0)
            w = w.bool()
            mask = torch.masked_select(mask.view(-1), w.view(-1)).cpu().numpy() - 1
            target = torch.masked_select(target.view(-1), w.view(-1)).cpu().numpy() - 1

            c_matrix = confusion_matrix(target, mask, test_dataloader.dataset.num_classes)
            acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = evaluate_metric(c_matrix)
    return acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa


if __name__ == '__main__':
    # configure GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # configure dataset
    # dataloader = NewPaviaLoader(config=config['data']['train']['params'])
    # tets_dataloader = NewPaviaLoader(config=config['data']['test']['params'])
    # dataloader = NewGrss2013Loader(config=config['data']['train']['params'])
    # tets_dataloader = NewGrss2013Loader(config=config['data']['test']['params'])
    dataloader = NewSalinasLoader(config=config['data']['train']['params'])
    tets_dataloader = NewSalinasLoader(config=config['data']['test']['params'])

    # configure model
    model = FreeNet(config=config['model']['params']).to(DEVICE)

    optimizer = torch.optim.SGD(params=model.parameters(), momentum=config['optimizer']['params']['momentum'],
                                weight_decay=config['optimizer']['params']['weight_decay'],
                                lr=config['learning_rate']['params']['base_lr'])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                       gamma=config['learning_rate']['params']['power'])

    bar = tqdm(list(range(config['learning_rate']['params']['max_iters'])))
    for i in bar:
        time.sleep(0.1)
        training_loss = 0.0
        num = 0
        for (data, y, weight) in dataloader:
            data = data.to(DEVICE)
            y = y.to(DEVICE)
            weight = weight.to(DEVICE)
            target = model(data)
            loss = loss_function(target, y, weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            num += 1

        if not os.path.exists(config['out_config']['params']['save_path']):
            os.makedirs(config['out_config']['params']['save_path'])
        fig_save_path = os.path.join(config['out_config']['params']['save_path'], str(i) + '.png')
        acc, acc_per_class, acc_cls, IoU, mean_IoU, kappa = fcn_evaluate_fn(model, test_dataloader=tets_dataloader,
                                                                            device=DEVICE,
                                                                            config=config['out_config']['params'],
                                                                            path=fig_save_path)

        bar.set_description(
            'loss: %.4f,OA: %.4f, mIOU: %.4f, kappa: %.4f' % (training_loss / num, acc, mean_IoU, kappa))

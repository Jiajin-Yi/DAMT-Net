import datetime
import sys
import os.path as osp
import os

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import makedatalist as ml
from model.advanced_model import *
from metrics import *
from model.discriminator import labelDiscriminator, featureDiscriminator
from dataset.source_dataset import sourceDataSet
from dataset.target_dataset import targetDataSet,targetDataSet_val
from utils.loss import CrossEntropy2d,MSELoss
from val import validate_model
import re
from add_arguments import get_arguments


class Logger(object):
    def __init__(self, filename='logprocess.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def loss_calc(pred, label, gpu, usecuda):
    if usecuda:
        label = Variable(label.long()).cuda(gpu)
        criterion = CrossEntropy2d().cuda(gpu)
        result = criterion(pred, label)
    else:
        label = Variable(label.long())
        criterion = CrossEntropy2d()
        result = criterion(pred, label)

    return result


def main():
    """Create the model and start the training."""
    # start logger
    sys.stdout = Logger(stream=sys.stdout)
    sys.stderr = Logger(stream=sys.stderr)

    usecuda = True
    cudnn.enabled = True
    args = get_arguments()

    # makedatalist
    ml.makedatalist(args.data_dir_img, args.data_list)
    ml.makedatalist(args.data_dir_target, args.data_list_target)
    ml.makedatalist(args.data_dir_val, args.data_list_val)

    # setting logging directory
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    """
        load the data
    """
    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    trainloader = data.DataLoader(
        sourceDataSet(args.data_dir_img, args.data_dir_label, args.data_list,
                     max_iters=args.num_steps,
                     crop_size=input_size),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(
        targetDataSet(args.data_dir_target, args.data_dir_target_label, args.data_list_target,
                      max_iters=args.num_steps, crop_size=input_size_target),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    targetloader_iter = enumerate(targetloader)

    valloader = data.DataLoader(
        targetDataSet_val(args.data_dir_val, args.data_dir_val_label, args.data_list_val,
                           crop_size=input_size_target),
        batch_size=1, shuffle=False)

    """
        build the network
    """
    model = source2targetNet(in_channels=1, out_channels=2)

    model_label = labelDiscriminator(num_classes=args.num_classes)

    input_channels = 64
    level = 1
    model_feature = featureDiscriminator(input_channels=input_channels,input_size=w/(2**(level-1)),num_classes=args.num_classes, fc_classifier=3)

    model.train()
    model_label.train()
    model_feature.train()

    if usecuda:
        cudnn.benchmark = True
        model.cuda(args.gpu)
        model_label.cuda(args.gpu)
        model_feature.cuda(args.gpu)

    """
        Loading the pretrain model
    """
    if args.pretrain == 1:
        old_model = torch.load(args.restore_from,map_location='cuda:'+str(args.gpu))

        model_encoder_dict = model.encoder.state_dict()
        model_domain_decoder_dict = model.domain_decoder.state_dict()

        pretrained_dict = old_model.module.state_dict()

        frozen_layer = args.frozen_layer
        random_layer = args.random_layer
        for k, v in pretrained_dict.items():
            flag_random = [True for pattern in random_layer if re.search(pattern, k) is not None]
            flag_frozen = [True for pattern in frozen_layer if re.search(pattern, k) is not None]
            if len(flag_frozen) != 0:
                v.requires_grad = False
                print('frozen layer: %s ' % k)
            if len(flag_random) == 0:
                if k in model_encoder_dict:
                    model_encoder_dict[k]=v
                    print(k)
                if k in model_domain_decoder_dict:
                    print(k)
        model.encoder.load_state_dict(model_encoder_dict)
        model.domain_decoder.load_state_dict(model_domain_decoder_dict)
        print('copy pretrain layer finish!')

    if args.iter_start != 0:
        args.restore_from = args.snapshots_dir + str(args.iter_start) + '.pth'
        args.Dlabelrestore_from = args.snapshots_dir  + str(args.iter_start) + '_D.pth'
        args.Dfeaturerestore_from = args.snapshots_dir  + str(args.iter_start) + '_D2.pth'
        model.load_state_dict(torch.load(args.restore_from))
        model_label.load_state_dict(torch.load(args.Dlabelrestore_from))
        model_feature.load_state_dict(torch.load(args.Dfeaturerestore_from))
        print('load old model')

    """
        Setup optimization for training
    """
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.9)

    optimizer.zero_grad()

    optimizer_label = optim.Adam(model_label.parameters(), lr=args.learning_rate_Dl, betas=(0.9, 0.99))
    scheduler_label = torch.optim.lr_scheduler.StepLR(optimizer_label, step_size=args.step_size_Dl, gamma=0.9)

    optimizer_label.zero_grad()

    optimizer_feature = optim.Adam(model_feature.parameters(), lr=args.learning_rate_Df, betas=(0.9, 0.99))
    scheduler_feature = torch.optim.lr_scheduler.StepLR(optimizer_feature, step_size=args.step_size_Df, gamma=0.9)

    optimizer_feature.zero_grad()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    entropy_loss = torch.nn.CrossEntropyLoss()
    rec_loss = MSELoss()

    # labels for adversarial training
    source_label = 1
    target_label = 0

    for i_iter in range(args.iter_start, args.num_steps):

        loss_seg_value = 0
        loss_adv_label_value = 0
        loss_adv_feature_value = 0

        loss_Dlabel_value = 0
        loss_Dfeature_value = 0
        loss_rec_value = 0
        optimizer.zero_grad()
        optimizer_label.zero_grad()
        optimizer_feature.zero_grad()

        # train G
        for param in model_label.parameters():
            param.requires_grad = False

        for param in model_feature.parameters():
            param.requires_grad = False

        # train with source
        _, batch = trainloader_iter.__next__()
        images, labels, _, _ = batch
        if usecuda:
            images_source = Variable(images).cuda(args.gpu)
        else:
            images_source = Variable(images)

        recimg_source, feature_source, pred_source = model(images_source)

        loss_seg = loss_calc(pred_source, labels, args.gpu, usecuda)
        loss_seg_value += loss_seg.data.cpu().numpy()

        loss_rec_source = rec_loss(recimg_source, images_source)

        label = torch.argmax(pred_source, dim=1).float()
        sdice, sjac = dice_coeff(label.cpu(), labels.cpu())

        _, batch = targetloader_iter.__next__()
        images, tlabels, _, _ = batch
        if usecuda:
            images_target = Variable(images).cuda(args.gpu)
        else:
            images_target = Variable(images)

        recimg_target, feature_target, pred_target = model(images_target)

        loss_rec_target = rec_loss(recimg_target, images_target)

        loss_rec = (loss_rec_source + loss_rec_target) / 2
        loss_rec_value += loss_rec.data.cpu().numpy()

        loss = loss_seg + args.lambda_rec * loss_rec
        loss.backward()

        # Target Domain Adv loss
        # acc the target domain adv loss
        _, feature_target, pred_target = model(images_target)

        Dlabel_out = model_label(F.softmax(pred_target, dim=1))
        if usecuda:
            adv_source_label = Variable(torch.FloatTensor(Dlabel_out.data.size()).fill_(source_label).cuda(args.gpu))
        else:
            adv_source_label = Variable(torch.FloatTensor(Dlabel_out.data.size()).fill_(source_label))
        loss_adv_label = bce_loss(Dlabel_out, adv_source_label)

        Dfeature_out = model_feature(feature_target)
        if usecuda:
            adv_source_label = Variable(torch.LongTensor(Dfeature_out.size(0)).fill_(source_label).cuda(args.gpu))
        else:
            adv_source_label = Variable(torch.LongTensor(Dfeature_out.size(0)).fill_(source_label))
        loss_adv_feature = entropy_loss(Dfeature_out, adv_source_label)

        loss = args.lambda_adv_label * loss_adv_label + args.lambda_adv_feature * loss_adv_feature
        loss.backward()

        loss_adv_label_value += loss_adv_label.data.cpu().numpy()
        loss_adv_feature_value += loss_adv_feature.data.cpu().numpy()

        # train domain label classifier
        for param in model_label.parameters():
            param.requires_grad = True

        # source domain D loss
        pred_source = pred_source.detach()
        D_out = model_label(F.softmax(pred_source, dim=1))
        if usecuda:
            D_source_label = Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label).cuda(args.gpu))
        else:
            D_source_label = Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label))
        loss_D = bce_loss(D_out, D_source_label)
        loss_D = loss_D / 2
        loss_D.backward()
        loss_Dlabel_value += loss_D.data.cpu().numpy()

        # target domain D loss
        pred_target = pred_target.detach()
        D_out = model_label(F.softmax(pred_target, dim=1))
        if usecuda:
            D_target_label = Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label).cuda(args.gpu))
        else:
            D_target_label = Variable(torch.FloatTensor(D_out.data.size()).fill_(target_label))
        loss_D = bce_loss(D_out, D_target_label)
        loss_D = loss_D / 2
        loss_D.backward()
        loss_Dlabel_value += loss_D.data.cpu().numpy()

        # train domain feature classifier
        for param in model_feature.parameters():
            param.requires_grad = True

        # train with source
        feature_source = feature_source.detach()
        D_out = model_feature(feature_source)
        if usecuda:
            D_source_label = Variable(torch.LongTensor(D_out.size(0)).fill_(source_label).cuda(args.gpu))
        else:
            D_source_label = Variable(torch.LongTensor(D_out.size(0)).fill_(source_label))
        loss_D = entropy_loss(D_out, D_source_label)
        loss_D = loss_D / 2
        loss_D.backward()
        loss_Dfeature_value += loss_D.data.cpu().numpy()

        # train with target
        feature_target = feature_target.detach()
        D_out = model_feature(feature_target)
        if usecuda:
            D_target_label = Variable(torch.LongTensor(D_out.size(0)).fill_(target_label).cuda(args.gpu))
        else:
            D_target_label = Variable(torch.LongTensor(D_out.size(0)).fill_(target_label))
        loss_D = entropy_loss(D_out, D_target_label)
        loss_D = loss_D / 2
        loss_D.backward()
        loss_Dfeature_value += loss_D.data.cpu().numpy()

        optimizer.step()
        optimizer_label.step()
        optimizer_feature.step()

        if scheduler is not None:
            scheduler.step(epoch=i_iter)
            args.learning_rate = scheduler.get_lr()[0]

        if scheduler_label is not None:
            scheduler_label.step(epoch=i_iter)
            args.learning_rate_Dl = scheduler_label.get_lr()[0]

        if scheduler_feature is not None:
            scheduler_feature.step(epoch=i_iter)
            args.learning_rate_Df = scheduler_feature.get_lr()[0]

        if (i_iter % 50 == 0):
            print('time = {0},lr = {1: 5f},lr_Dl = {2: 6f},lr_Df = {3: 6f}'.format(datetime.datetime.now(), args.learning_rate,
                                                                  args.learning_rate_Dl,args.learning_rate_Df))
            print(
                'iter = {0:8d}/{1:8d}, loss_seg = {2:.5f} loss_rec = {3:5f} loss_adv1 = {4:.5f}, loss_adv2 = {5:.5f}, loss_Dlabel = {6:.5f} loss_Dfeature = {7:.5f}'.format(
                    i_iter, args.num_steps, loss_seg_value, loss_rec_value, loss_adv_label_value, loss_adv_feature_value, loss_Dlabel_value,loss_Dfeature_value))
            print(
                'sdice2 = {0:.5f} sjac2 = {1:.5f}'.format(
                    sdice, sjac,
                ))

        if i_iter % args.save_pred_every == 0:
            dice, jac = validate_model(model.get_target_segmentation_net(), valloader, './val/cvlab', i_iter,
                                       args.gpu,usecuda)
            print('val dice: %4f' % dice, 'val jac: %4f' % jac)
            if jac > args.best_tjac:
                args.best_tjac = jac
                print('best val dice: %4f' % dice, 'best val jac: %f' % jac)
                torch.save(model.state_dict(),
                           osp.join(args.snapshot_dir, 'CVbest' + str(i_iter) + '_' + str(jac) + '.pth'))
                torch.save(model_label.state_dict(),
                           osp.join(args.snapshot_dir, 'CVbest' + str(i_iter) + '_' + str(jac) + '_D.pth'))
                torch.save(model_feature.state_dict(),
                           osp.join(args.snapshot_dir, 'CVbest' + str(i_iter) + '_' + str(jac) + '_D2.pth'))

            else:
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'CV_' + str(i_iter) + '.pth'))
                torch.save(model_label.state_dict(),
                               osp.join(args.snapshot_dir, 'CV_' + str(i_iter) + '_D.pth'))
                torch.save(model_feature.state_dict(),
                               osp.join(args.snapshot_dir, 'CV_' + str(i_iter) + '_D2.pth'))



if __name__ == '__main__':
    main()

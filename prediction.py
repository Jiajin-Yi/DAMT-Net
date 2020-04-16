import argparse
import datetime
import os.path as osp
import sys

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data

import makedatalist as ml
from model.advanced_model import *
from dataset.target_dataset import targetDataSet_val
from metrics import *
from utils.loss import CrossEntropy2d,MSELoss
from val import validate_model
from postprocessing import *
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

def prediction():
    """Create the model and start the training."""
    # start logger
    sys.stdout = Logger(stream=sys.stdout)
    sys.stderr = Logger(stream=sys.stderr)

    usecuda = True
    cudnn.enabled = True
    args = get_arguments()

    # makedatalist
    ml.makedatalist(args.data_dir_test, args.data_list_test)
    pred_final = np.zeros((165,768,1024))

    for test_aug in range(4):
        print('the %d test_aug'%test_aug,'for %s'%args.savedir)

        """
            load the data
        """
        input_size_target = (512, 512)

        testloader = data.DataLoader(
            targetDataSet_val(args.data_dir_test, args.data_dir_test_label, args.data_list_test,
                           test_aug,crop_size=input_size_target),
            batch_size=1, shuffle=False)

        if args.model == 'AE':
            model = source2targetNet(in_channels=1, out_channels=2)

        if usecuda:
            cudnn.benchmark = True
            model.cuda(args.gpu)

        model.load_state_dict(torch.load(args.test_model_path,map_location='cuda:0'))
        # model = torch.load(args.restore_from)

        if args.model == 'AE':
            testmodel = model.get_target_segmentation_net()

        pred_total,original_msk_total =  validate_model(testmodel, testloader, args.savedir,args.gpu, usecuda,test_aug)

        pred_total = postpre(pred_total,args.savedir,test_aug)

        D3_dice, D3_jac = dice_coeff(pred_total,original_msk_total)

        total_dice = 0
        total_jac = 0
        pics = pred_total.shape[0]
        for i in range(pics):
            pred = pred_total[i,:,:]
            msk = original_msk_total[i,:,:]
            dice, jac = dice_coeff(pred, msk)
            total_dice = total_dice + dice
            total_jac = total_jac + jac

        print('3D dice: %4f' % D3_dice, '3D jac: %4f' % D3_jac,
                    '2D dice: %4f' % (total_dice/(pics)), '2D jac: %4f' % (total_jac/(pics)))

        if test_aug == 0:
            msk_final = original_msk_total
        if test_aug == 1:
            for i in range(pred_total.shape[0]):
                pred_total[i,:,:] = cv2.flip(pred_total[i,:,:], 1)
        if test_aug == 2:
            for i in range(pred_total.shape[0]):
                pred_total[i,:,:] = cv2.flip(pred_total[i,:,:], 0)
        if test_aug == 3:
            for i in range(pred_total.shape[0]):
                pred_total[i,:,:] = cv2.flip(pred_total[i,:,:], -1)

        pred_final = pred_final + pred_total

    pred_final = pred_final/4
    pred_final[pred_final>=0.5]=1
    pred_final[pred_final<0.5]=0

    desired_path = args.savedir + '_final'+'/'

    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    export_name = 'test.nii.gz'
    save_array_as_nii_volume(pred_final, desired_path + export_name)

    final3_dice,final3_jac = dice_coeff(pred_final,msk_final)
    final2_dice = 0
    final2_jac = 0
    pics = pred_final.shape[0]
    for i in range(pics):
        pred = pred_final[i, :, :]
        msk = msk_final[i, :, :]
        dice, jac = dice_coeff(pred, msk)
        final2_dice = final2_dice + dice
        final2_jac = final2_jac + jac

    print('final 3D dice: %4f' % final3_dice, 'final 3D jac: %4f' % final3_jac,
          'final 2D dice: %4f' % (final2_dice / (pics)), 'final 2D jac: %4f' % (final2_jac / (pics)))

    pred_final = postpre(pred_final, args.savedir,5)

    final3_dice,final3_jac = dice_coeff(pred_final,msk_final)
    final2_dice = 0
    final2_jac = 0
    pics = pred_final.shape[0]
    for i in range(pics):
        pred = pred_final[i, :, :]
        msk = msk_final[i, :, :]
        dice, jac = dice_coeff(pred, msk)
        final2_dice = final2_dice + dice
        final2_jac = final2_jac + jac

    print('final 3D dice: %4f' % final3_dice, 'final 3D jac: %4f' % final3_jac,
          'final 2D dice: %4f' % (final2_dice / (pics)), 'final 2D jac: %4f' % (final2_jac / (pics)))

    desired_path = args.savedir + '_finalpostpre'+'/'

    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    export_name = 'test.nii.gz'
    save_array_as_nii_volume(pred_final, desired_path + export_name)


if __name__ == '__main__':
    prediction()
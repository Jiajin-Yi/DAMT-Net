import sys

import torch.backends.cudnn as cudnn

import makedatalist as ml
from model.advanced_model import *
from metrics import *
from utils.postprocessing import *
from utils.test_augmentation import test_augmentation
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

    if args.model == 'AE':
        model = source2targetNet(in_channels=1, out_channels=2)

    if usecuda:
        cudnn.benchmark = True
        model.cuda(args.gpu)

    model.load_state_dict(torch.load(args.test_model_path, map_location='cuda:0'))
    # model = torch.load(args.restore_from)

    if args.model == 'AE':
        testmodel = model.get_target_segmentation_net()

    pred_ori = np.zeros((165,768,1024))
    input_size_target = (512, 512)

    pred_final,msk_final = test_augmentation(testmodel,pred_ori,input_size_target,args,usecuda=True)

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

    desired_path = args.savedir + 'final_postpre'+'/'

    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    export_name = 'test.nii.gz'
    save_array_as_nii_volume(pred_final, desired_path + export_name)


if __name__ == '__main__':
    prediction()
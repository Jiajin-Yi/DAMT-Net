from torch.utils import data
from dataset.target_dataset import targetDataSet_test
from metrics import *
from val import test_model
from utils.postprocessing import *


def test_augmentation(testmodel, pred_ori, input_size_target, args, usecuda):

    pred_final = pred_ori

    for test_aug in range(args.test_aug):
        print('the %d test_aug' % test_aug, 'for %s' % args.save_dir)

        """
            load the data
        """

        testloader = data.DataLoader(
            targetDataSet_test(args.data_dir_test, args.data_dir_test_label, args.data_list_test,
                              test_aug, crop_size=input_size_target),
            batch_size=1, shuffle=False)

        pred_total, original_msk_total = test_model(testmodel, testloader, args.save_dir, args.gpu, usecuda,
                                                        test_aug)

        pred_total = postpre(pred_total, args.save_dir, test_aug)

        D3_dice, D3_jac = dice_coeff(pred_total, original_msk_total)

        total_dice = 0
        total_jac = 0
        pics = pred_total.shape[0]
        for i in range(pics):
            pred = pred_total[i, :, :]
            msk = original_msk_total[i, :, :]
            dice, jac = dice_coeff(pred, msk)
            total_dice = total_dice + dice
            total_jac = total_jac + jac

        print('3D dice: %4f' % D3_dice, '3D jac: %4f' % D3_jac,
              '2D dice: %4f' % (total_dice / (pics)), '2D jac: %4f' % (total_jac / (pics)))

        if test_aug == 0:
            msk_final = original_msk_total
        if test_aug == 1:
            for i in range(pred_total.shape[0]):
                pred_total[i, :, :] = cv2.flip(pred_total[i, :, :], 1)
        if test_aug == 2:
            for i in range(pred_total.shape[0]):
                pred_total[i, :, :] = cv2.flip(pred_total[i, :, :], 0)
        if test_aug == 3:
            for i in range(pred_total.shape[0]):
                pred_total[i, :, :] = cv2.flip(pred_total[i, :, :], -1)

        pred_final = pred_final + pred_total

    pred_final = pred_final / 4
    pred_final[pred_final >= 0.5] = 1
    pred_final[pred_final < 0.5] = 0

    desired_path = args.save_dir + 'final' + '/'

    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    export_name = 'test.nii.gz'
    save_array_as_nii_volume(pred_final, desired_path + export_name)

    return pred_final, msk_final

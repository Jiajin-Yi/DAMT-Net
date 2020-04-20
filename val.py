import os

from model.advanced_model import *
from metrics import *
from pre_processing import *
import math


def validate_model(model, valloader, save_dir,i_iter,gpu,usecuda):
    if usecuda:
        model.cuda(gpu)
    model.eval()
    total_dice = 0
    total_jac = 0
    count = 0
    for i_pic, (images_v, masks_v, original_msk,_,name)in enumerate(valloader):
        if usecuda:
            stacked_img = torch.Tensor([]).cuda(gpu)
        else:
            stacked_img = torch.Tensor([])
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                if usecuda:
                    image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda(gpu))
                else:
                    image_v = Variable(images_v[:, index, :, :].unsqueeze(0))
            try:
                _, output = model(image_v)
                output = torch.argmax(output, dim=1).float()
                stacked_img = torch.cat((stacked_img, output))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
        pred, original_msk = save_prediction_image(stacked_img, name,i_iter,save_dir,original_msk)

        dice, jac = dice_coeff(pred, original_msk)

        total_dice = total_dice+dice
        total_jac = total_jac+jac

        count = count + 1

    return total_dice/count, total_jac/count

def test_model(model, valloader, save_dir,i_iter,gpu,usecuda,test_aug):
    if usecuda:
        model.cuda(gpu)
    model.eval()
    total_dice = 0
    total_jac = 0
    count = 0
    for i_pic, (images_v, masks_v, original_msk,_,name)in enumerate(valloader):
        if usecuda:
            stacked_img = torch.Tensor([]).cuda(gpu)
        else:
            stacked_img = torch.Tensor([])
        for index in range(images_v.size()[1]):
            with torch.no_grad():
                if usecuda:
                    image_v = Variable(images_v[:, index, :, :].unsqueeze(0).cuda(gpu))
                else:
                    image_v = Variable(images_v[:, index, :, :].unsqueeze(0))
            try:
                _, output = model(image_v)
                output = torch.argmax(output, dim=1).float()
                stacked_img = torch.cat((stacked_img, output))
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory')
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise e
        pred, original_msk = save_prediction_image(stacked_img, name,i_iter,save_dir,original_msk)
        dim = pred.shape

        dice, jac = dice_coeff(pred, original_msk)
        count = count + 1

        total_dice = total_dice + dice
        total_jac = total_jac + jac
        #
        print("%d.  val_jac is:%f . val_dice is:%f " % (i_pic, jac, dice))

        pred_total = np.append(pred_total, pred)
        original_msk_total = np.append(original_msk_total, original_msk)

    D3_dice, D3_jac = dice_coeff(pred_total, original_msk_total)
    D2_dice = total_dice / count
    D2_jac = total_jac / count
    print('3D dice: %4f' % D3_dice, '3D jac: %4f' % D3_jac,
          '2D dice: %4f' % D2_dice, '2D jac: %4f' % D2_jac)

    pred_total = pred_total.reshape(count, dim[0], dim[1])
    original_msk_total = original_msk_total.reshape(count, dim[0], dim[1])

    return pred_total, original_msk_total


def save_prediction_image(stacked_img, im_name, iter, save_folder_name, original_msk):
    """save images to save_path
    Args:
        stacked_img (numpy): stacked cropped images
        save_folder_name (str): saving folder name
        division_array(388, 2, 3, 768, 1024):
                388: label patch size
                2, divide num in heigh
                3, divide num in width
                768: image height
                1024: image width

    """
    crop_size = stacked_img[0].size()

    maxsize = original_msk.shape[1:]

    output_shape = original_msk.shape[1:]
    crop_n1 = math.ceil(output_shape[0] / crop_size[0])
    crop_n2 = math.ceil(output_shape[1] / crop_size[1])
    if crop_n1 == 1:
        crop_n1 = crop_n1
    else:
        crop_n1 = crop_n1 + 1
    if crop_n2 == 1:
        crop_n2 = crop_n2
    else:
        crop_n2 = crop_n2 + 1

    div_arr = division_array(stacked_img.size(1), crop_n1, crop_n2, output_shape[0], output_shape[1])
    img_cont = image_concatenate(stacked_img.cpu().data.numpy(), crop_n1, crop_n2, output_shape[0], output_shape[1])

    img_cont = polarize((img_cont) / div_arr)
    img_cont_np = img_cont.astype('uint8')

    img_cont = Image.fromarray(img_cont_np * 255)
    # organize images in every epoch
    desired_path = save_folder_name + '_iter_' + str(iter) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    img_cont.save(desired_path + export_name)
    return img_cont_np, original_msk


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img

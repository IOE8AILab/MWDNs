import cv2.cv2
import torch, math
import time
from model import MWDNet, MWDNet_CPSF
from utils import *
from lpips_criterion import Lpips


def test_dataset(model, test_data, device):
    cost_mse = torch.nn.MSELoss()
    cost_lpips = Lpips(net='alex', device=device)
    num = np.array([63, 88, 324, 641, 738, 934, 1234])

    total_time = 0
    img_lpips = 0
    img_psnr = 0
    model.eval()
    center_img = (320 - 128) // 2
    img_blur = torch.zeros([7, 3, 320, 320])
    img_gt = torch.zeros([7, 3, 128, 128])
    img_pre = torch.zeros([7, 3, 128, 128])
    k = 0
    with torch.no_grad():
        for (i, data) in enumerate(test_data):
            blur = data[1].to(device)
            gt = data[0].to(device)
            
            st = time.time()
            net_pre = model(blur)
            et = time.time()
            net_pre = net_pre[:, :, center_img:-center_img, center_img:-center_img]
            gt = gt[:, :, center_img:-center_img, center_img:-center_img]
            
            loss_mse = cost_mse(net_pre, gt)
            loss_lpips = cost_lpips(net_pre, gt)
            psnr = 20 * math.log10(1. / (loss_mse.item() ** 0.5))
            
            img_lpips += loss_lpips.item()
            total_time += (et - st)
            img_psnr += psnr
            
            if i in num:
                print('psnr: %.4f    lpips: %.4f' % (psnr, loss_lpips.item()))
                img_blur[k] = blur[0]
                img_gt[k] = gt[0]
                img_pre[k] = torch.clip(net_pre[0], 0., 1.)
                k += 1

    show_img(img_blur, 'dataset_blur')
    show_img(img_gt, 'dataset_gt')
    show_img(img_pre, 'dataset_mwdn')
    time_loss = total_time / len(test_data)
    img_lpips = img_lpips / len(test_data)
    img_psnr = img_psnr / len(test_data)

    print('psnr: %.4f     lpips: %.4f     time: %.4f' % (img_psnr, img_lpips, time_loss))


def test_real_object(model, blur):
    center_img = (320 - 128) // 2
    show_img(blur, 'real_object_blur')

    model.eval()
    with torch.no_grad():
        net_pre = model(blur)[:, :, center_img:-center_img, center_img:-center_img]
        show_img(torch.clip(net_pre, 0, 1), 'real_object_mwdn')


if __name__ == '__main__':
    device = 'cuda:1'
    batch_size = 1
    path = '/media/gym/HDD3/LY_Data/lensless_320/'
    psf = cv2.cv2.imread(path + 'psf.png', -1).astype(np.float32).transpose(2, 0, 1) / 255.
    psfs = torch.tensor(psf, dtype=torch.float32, device=device).unsqueeze(0)
    
    model = MWDNet(3, 3, psfs).to(device)
    al = torch.load('./results/train_MWDN_best.tar', map_location=device)
    model.load_state_dict(al['net_state_dict'])
    for name, param in model.named_parameters():
        param.requires_grad = False

    _, _, test_data = load_data(path + 'lensless_320.h5', 25000, batch_size)

    test_dataset(model, test_data, device)
    
    blur_im = torch.zeros([7, 3, 320, 320], dtype=torch.float32, device=device)
    for j in range(7):
        path_blur = path + 'data_real_object/' + str(j) + '.png'
        blur_ = cv2.cv2.imread(path_blur).transpose(2, 0, 1) / 255.
        blur_im[j] = torch.tensor(blur_, dtype=torch.float32, device=device)
    test_real_object(model, blur_im)

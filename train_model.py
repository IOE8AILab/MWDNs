import torch.nn.functional as F
from model import MWDNet, MWDNet_CPSF
from utils import *
from lpips_criterion import Lpips


def train(model, optim, sched, train_data, val_data, epoch, device):
    center_img = (320 - 128) // 2
    vla = float('inf')
    cost_mse = torch.nn.MSELoss()
    cost_lpips = Lpips(net='alex', device=device)

    for e in range(epoch):
        print('Epoch:' + str(e + 1) + '/' + str(epoch))
        for param_group in optim.param_groups:
            LR = param_group['lr']
            if LR < 2e-8:
                return model
            elif LR > 1e-6:
                w_mse = 1.
                w_lpips = 1.
            else:
                w_mse = 1.
                w_lpips = 0.05
        
        model.train()
        for (i, data) in enumerate(train_data):
            blur = data[1].to(device)
            gt = data[0].to(device)
            optim.zero_grad()
            Y_pre = model(blur)
            loss_mse = cost_mse(Y_pre, gt)
            loss_lpips = cost_lpips(Y_pre, gt)
            loss = w_mse * loss_mse + w_lpips * loss_lpips
            loss.backward()
            optim.step()
            if i % 200 == 0:
                with torch.no_grad():
                    y1 = Y_pre[:, :, center_img:-center_img, center_img:-center_img]
                    y2 = gt[:, :, center_img:-center_img, center_img:-center_img]
                    loss_mse = cost_mse(y1, y2)
                    loss_lpips = cost_lpips(y1, y2)
                print('Iterations: %d    mse_loss: %.4f    lpips_loss: %.4f' % (i, loss_mse.item(), loss_lpips.item()))

        img_mse = 0
        img_lpips = 0
        model.eval()
        with torch.no_grad():
            for (i, data) in enumerate(val_data):
                blur = data[1].to(device)
                gt = data[0].to(device)
                y2 = gt[:, :, center_img:-center_img, center_img:-center_img]
                Y_pre = model(blur)
                y1 = Y_pre[:, :, center_img:-center_img, center_img:-center_img]
                loss_mse = cost_mse(Y_pre, gt)
                loss_lpips = cost_lpips(Y_pre, gt)
                img_mse += loss_mse.item()
                img_lpips += loss_lpips.item()
        img_mse = img_mse / len(val_data)
        img_lpips = img_lpips / len(val_data)
        dict_save = {'net_state_dict': model.state_dict()}
        losscost = img_mse + img_lpips
        if losscost < vla:
            vla = losscost
            torch.save(dict_save, './results/train_MWDN_best.tar')
        sched.step(vla)
        print('val mse loss: %.4f    val lpips loss: %.4f' % (img_mse, img_lpips))

        torch.save(dict_save, './results/train_MWDN_latest.tar')
    return model

if __name__ == '__main__':
    device = 'cuda:1'
    LR = 1e-4 
    epoch = 300
    batch_size = 16
    path = '/media/gym/HDD3/LY_Data/lensless_320/'
    psf = cv2.cv2.imread(path + 'psf.png', -1).astype(np.float32).transpose(2, 0, 1) / 255.
    psfs = torch.tensor(psf, dtype=torch.float32, device=device).unsqueeze(0)
    model = MWDNet(3, 3, psfs).to(device)
    train_data, val_data, _ = load_data(path + 'lensless_320.h5', 25000, batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5, verbose=True,
                                                       threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-8)
    model = train(model, optim, sched, train_data, val_data, epoch, device=device)

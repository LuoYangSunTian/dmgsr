import math
import os
import time

import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.model_E = torch.nn.DataParallel(self.model.get_model().E, range(self.args.n_GPUs))
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)

        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'), )
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

    def train(self):
        self.loss.step()
        self.scheduler.step()
        epoch = self.scheduler.last_epoch + 1
        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = 1e-4 + (1e-3 - 1e-4) * (1 + math.cos(epoch * math.pi / 100)) / 2
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            lr = 1e-7 + (1e-4 - 1e-7) * (1 + math.cos((epoch - 100) * math.pi / 600)) / 2
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        degrade = util.SRMDPreprocessing(
            self.scale[0],
            kernel_size=self.args.blur_kernel,
            blur_type=self.args.blur_type,
            sig_min=self.args.sig_min,
            sig_max=self.args.sig_max,
            lambda_min=self.args.lambda_min,
            lambda_max=self.args.lambda_max,
            noise=self.args.noise
        )

        timer = utility.timer()
        losses_contrast, losses_sr = utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            hr = hr.cuda()  # b, n, c, h, w
            lr, b_kernels, sigma_kernel = degrade(hr)  # b, n, c, h, w
            #print("[DEBUG]----b_kernels.shape: {}".format(b_kernels.shape))
            #print("[DEBUG]----sigma_kernel.shape: {}".format(sigma_kernel))

            # 检查 sigma_kernel 的条件
            # if sigma_kernel is None or sigma_kernel.shape[0] != 18:
            #     print("[DEBUG]----Skipping batch due to invalid sigma_kernel.")
            #     continue

            self.optimizer.zero_grad()

            timer.tic()
            # forward
            # train degradation encoder
            if epoch <= self.args.epochs_encoder:
                _, output, target = self.model_E(im_q=lr[:, 0, ...], im_k=lr[:, 1, ...], kernel=sigma_kernel)
                loss_constrast = self.contrast_loss(output, target)
                loss = loss_constrast

                losses_contrast.update(loss_constrast.item())
            # train the whole network
            else:
                sr, output, target = self.model((lr[:, 0, ...], lr[:, 1, ...], sigma_kernel))
                loss_SR = self.loss(sr, hr[:, 0, ...])
                loss_constrast = self.contrast_loss(output, target)
                loss = loss_constrast + loss_SR

                losses_sr.update(loss_SR.item())
                losses_contrast.update(loss_constrast.item())

            # backward
            loss.backward()
            self.optimizer.step()
            # self.scheduler.step()
            timer.hold()
        self.loss.end_log(len(self.loader_train))
        self.ckp.write_log(
            'Epoch: [{:04d}]\t'
            'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
            'Time [{:.1f}s]'.format(
                epoch, losses_sr.avg, losses_contrast.avg, timer.release()
            ))
        # save model
        target = self.model.get_model()
        model_dict = target.state_dict()
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
        )

    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=self.args.sig,
                    lambda_1=self.args.lambda_1,
                    lambda_2=self.args.lambda_2,
                    theta=self.args.theta,
                    noise=self.args.noise
                )

                for idx_img, (hr, filename, _) in enumerate(self.loader_test):
                    hr = hr.cuda()  # b, n, c, h, w
                    hr = self.crop_border(hr, scale)
                    lr, _, kernel = degrade(hr, random=False)  # b, 1, c, h, w
                    hr = hr[:, 0, ...]
                    # inference
                    timer_test.tic()
                    sr = self.model((lr[:, 0, ...], lr[:, 0, ...], kernel))
                    end_time = time.time()
                    timer_test.hold()
                    sr = utility.quantize(sr, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)
                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        self.ckp.save_results(filename, save_list, scale, self.args.sig, self.args.data_test)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f} Time [{:.1f}s]'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                        timer_test.release()
                    ))

    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h // (self.scale[0]*32)) * (self.scale[0]*32), :int(w // (self.scale[0]*32)) * (self.scale[0]*32)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr

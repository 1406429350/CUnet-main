import torch.nn as nn
import glob, os
import argparse
from torchcontrib.optim import SWA
import torch.optim as optim
from torch.utils.data import DataLoader

from DCUNet.sedataset import SEDataset
from DCUNet.source_separator import SourceSeparator
from DCUNet.criterion import cyc, STFT
from DCUNet.metric import STFT_metric

import PinkBlack.trainer

def get_dataset(args):
    def get_wav(dir):
        wavs = []
        wavs.extend(glob.glob(os.path.join(dir, "**/*.wav"), recursive=True))
        # wavs.extend(glob.glob(os.path.join(dir, "**/*.flac"), recursive=True))
        # wavs.extend(glob.glob(os.path.join(dir, "**/*.pcm"), recursive=True))
        return wavs

    train_heart = get_wav(args.train_heart)
    train_lung = get_wav(args.train_lung)
    train_mix = get_wav(args.train_mix)
    # print("train_signals shape:", np.shape(train_signals))
    #  train_heart = get_wav(args.train_heart)
    #  train_lung = get_wav(args.train_lung)

    test_heart = get_wav(args.test_heart)
    test_lung = get_wav(args.test_lung)
    test_mix = get_wav(args.test_mix)

    train_dset = SEDataset(train_heart, train_lung, train_mix, sequence_length=args.sequence_length,
                              is_validation=False, preload=args.preload)
    valid_dset = SEDataset(test_heart, test_lung, test_mix, sequence_length=args.sequence_length,
                              is_validation=True, preload=args.preload)

    return dict(train_dset=train_dset,
                valid_dset=valid_dset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MODEL SETTING OPTION...')
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--batch_size", type=int, default=20, help="Input batch size")
    parser.add_argument("--train_heart", type=str, default="D:/心音文件夹/数据集/sz_mix_train/s1/", help="Input train heart WAV")
    parser.add_argument("--train_lung", type=str, default="D:/心音文件夹/数据集/sz_mix_train/s2/", help="Input train lung WAV")
    parser.add_argument("--train_mix", type=str, default="D:/心音文件夹/数据集/sz_mix_train/mix/", help="Input train mix WAV")
    parser.add_argument("--test_heart", type=str, default="D:/心音文件夹/数据集/sz_mix_eval/s1/", help="Input test heart WAV")
    parser.add_argument("--test_lung", type=str, default="D:/心音文件夹/数据集/sz_mix_eval/s2/", help="Input test lung WAV")
    parser.add_argument("--test_mix", type=str, default="D:/心音文件夹/数据集/sz_mix_eval/mix/", help="Input test mix WAV")
    parser.add_argument("--sequence_length", type=int, default=16384, help="sequence_length")
    parser.add_argument("--num_step", type=int, default=1000, help="num_step")
    parser.add_argument("--validation_interval", type=int, default=1, help="validation_interval")
    parser.add_argument("--num_workers", type=int, default=0, help="Input epochs")
    parser.add_argument("--ckpt", type=str, default="unet/ckpt.pth", help="Input save file name")
    parser.add_argument("--model_complexity", type=int, default=45, help="model_complexity")
    parser.add_argument("--model_depth", type=int, default=10, help="model_depth")
    parser.add_argument("--lr", type=float, default=0.001, help="Inputs learning rate")
    parser.add_argument("--lr_decay", type=float, default=0, help="Inputs learning rate")
    parser.add_argument("--num_signal", type=int, default=0, help="num_signal")
    parser.add_argument("--num_noise", type=int, default=0, help="num_noise")
    parser.add_argument("--optimizer", type=str, default="adam", help="Input optimizer option")
    parser.add_argument("--momentum", type=int, default=0, help="momentum")
    parser.add_argument("--multi_gpu", type=bool, default=False, help="multi_gpu")
    parser.add_argument("--complex", type=bool, default=True, help="complex")
    parser.add_argument("--swa", type=bool, default=False, help="swa")
    parser.add_argument("--loss", type=str, default="STFT", help="Input Loss function")
    parser.add_argument("--log_amp", type=bool, default=False, help="log_amp")
    parser.add_argument("--metric", type=str, default="pesq", help="metric")
    parser.add_argument("--train_dataset", type=str, default="mix", help="dataset type")
    parser.add_argument("--valid_dataset", type=str, default="mix", help="dataset type")
    parser.add_argument("--preload", type=bool, default=False, help="preload ")
    parser.add_argument("--padding_mode", type=str, default="reflect", help="padding_mode")
    args = parser.parse_args()

    dset = get_dataset(args)
    train_dset, valid_dset = dset['train_dset'], dset['valid_dset']

    train_dl = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=False)
    valid_dl = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                          pin_memory=False)

    if args.loss == "STFT":
        loss = STFT()
    elif args.loss == "cyc":
        loss = cyc()
    else:
        raise NotImplementedError(f"unknown loss ({args.loss})")
 #   opt = parse(args.opt, is_tain=True)
    metric = STFT_metric()

    net = SourceSeparator(complex=args.complex, model_complexity=args.model_complexity, model_depth=args.model_depth,
                          log_amp=args.log_amp, padding_mode=args.padding_mode).cuda()
   # print(net)

    if args.multi_gpu:
        net = nn.DataParallel(net).cuda()

    if args.optimizer == "adam":
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), lr=args.lr,
                                  momentum=args.momentum)
    else:
        raise ValueError(f"Unknown optimizer - {args.optimizer}")

    if args.swa:
        steps_per_epoch = args.validation_interval
        optimizer = SWA(optimizer, swa_start=int(20) * steps_per_epoch, swa_freq=1 * steps_per_epoch)

    if args.lr_decay >= 1 or args.lr_decay <= 0:
        scheduler = None
    else:
        if args.optimizer == "swa":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer.optimizer, mode="max", patience=5,
                                                                 factor=args.lr_decay)
        else:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5,
                                                                 factor=args.lr_decay)

    trainer = PinkBlack.trainer.Trainer(net,
                                            criterion=loss,
                                            metric=metric,
                                            train_dataloader=train_dl,
                                            val_dataloader=valid_dl,
                                            ckpt=args.ckpt,
                                            optimizer=optimizer,
                                            lr_scheduler=scheduler,
                                            is_data_dict=True,
                                            )

    trainer.train(step=args.num_step, validation_interval=args.validation_interval)

    if args.swa:
        trainer.swa_apply(bn_update=True)
        trainer.train(1, phases=['val'])






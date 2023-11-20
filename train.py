from model.model import Siege
from model.ema import ExponentialMovingAverage
from model.model_config import config_backbone
from data.dataset import BackboneDataset
from utils.training_utils import dsm, sample_noise, setup, sync_tensor_across_gpus
from time import time
import datetime
import torch
import sys
import argparse
import numpy as np
import os

from torch_geometric.loader import DataLoader
import torch.distributed as dist

torch.autograd.set_detect_anomaly(True)


def train(model, epochs, output_file, batch_size, lr, sde, ema_decay,
          gradient_clip=None, eps=1e-5, saved_params=None, data_path="./", distributed=True):
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not saved_params is None:
        optimizer.load_state_dict(torch.load(saved_params)["optimizer_state_dict"])

    if ema_decay is not None:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

        if saved_params is not None:
            ema.load_state_dict(torch.load(saved_params)["ema_state_dict"])
    '''

    # Dataset loading
    dataset = BackboneDataset(data_dir=data_path, mode='train')
    val_dataset = BackboneDataset(data_dir=data_path, mode='test')

    if distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler_train, num_workers=4,
                            drop_last=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=4,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    start_time = time()

    log_step = 100

    iters = len(loader)
    print(iters)

    for e in range(epochs):

        model.train()
        losses = []

        for value, features in enumerate(loader):
            if features is None:
                continue
            torch.cuda.empty_cache()

            optimizer.zero_grad()

            z, t, perturbed_data, mean, std = sample_noise(sde, features.pos, features.batch,
                                                           device="cuda")
            features = features.cuda()
            
            # Get network prediction
            prediction = model(f_in=features.x, pos=perturbed_data, batch=features.batch,
                               node_atom=features.z, sde=sde, t=t,)

            all_losses, loss = dsm(prediction, std, z, )

            for index, i in enumerate(all_losses):
                losses.append(torch.sum(i).cpu().detach().numpy())

            loss.requires_grad_(True)
            loss.backward()

            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            if ema_decay is not None:
                ema.update(model.parameters())

            if (value + 1) % log_step == 0 or value == iters - 1:
                elapsed = time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                log = "Elapsed [{}], Epoch [{}/{}], Iter [{}/{}]".format(
                    elapsed, e + 1, epochs, value + 1, iters)
                log += ", {}: {:.5f}".format('Loss', np.mean(losses))
                log += ", {}: {:.5f}".format('Std', np.std(losses))

                print(log)
                sys.stdout.flush()

                losses = []

            if (value + 1) % 2500 == 0 or (value + 1) == iters:

                if local_rank == 0:
                    if not os.path.isdir("checkpoints_"):
                        os.mkdir("checkpoints_")

                    torch.save({"model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                },
                               "checkpoints_" + "/" + output_file.replace(".pth", "_" + str(value + 1)
                                                                          + "_" + str(e) + ".pth"))


def prepare_model_dict(input_dict):

    already_ddp = False

    for keys in input_dict.keys():
        if 'module' in keys:
            already_ddp = True
        break
           
    if already_ddp:
        return {key.replace('module.', ''): value for key, value in input_dict.items()} 
    else:
        return input_dict            


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-sv", dest="use_saved", help="Flag for whether or not to use a saved model",
                        required=False, default=False, action='store_true')
    parser.add_argument("-sm", dest="saved_model", help="File containing saved params",
                        required=False, type=str, default="Saved.pth")
    parser.add_argument("-o", dest="output_file",
                        help="File for output of model parameters", required=True, type=str)
    parser.add_argument("-d", dest="data_path",
                        help="Directory where data is stored", required=False,
                        type=str, default="../NMR_data/processed")
    parser.add_argument("-ep", dest="epochs", help="Number of epochs",
                        required=False, type=int, default=100)
    parser.add_argument("--seed", type=int, default=12)  # My lucky number

    args = parser.parse_args()

    local_rank, device = setup()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = config_backbone
    model = Siege()
    model.cuda()

    # load model parameters
    model.load_state_dict(torch.load(args.saved_model)["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    ema_decay = config.training.ema

    if args.saved_model is not None:
        optimizer.load_state_dict(torch.load(args.saved_model)["optimizer_state_dict"])

    if ema_decay is not None:
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

        if args.saved_model is not None:
            ema.load_state_dict(torch.load(args.saved_model)["ema_state_dict"])

    print('Setting completed, start training!')

    # distributed training
    model = torch.nn.parallel.DistributedDataParallel(model)

    sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max)

    torch.cuda.empty_cache()

    if args.use_saved:

        # model.load_state_dict(prepare_model_dict(torch.load(args.saved_model)["model_state_dict"]))
        train(model, args.epochs, args.output_file,
              config.training.batch_size, config.training.lr, sde,
              ema_decay=config.training.ema, gradient_clip=config.training.gradient_clip,
              saved_params=args.saved_model, data_path=args.data_path, )
    else:

        train(model, args.epochs, args.output_file, config.training.batch_size,
              config.training.lr, sde, ema_decay=config.training.ema, gradient_clip=config.training.gradient_clip,
              data_path=args.data_path, )

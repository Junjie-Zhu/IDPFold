from model.model import IDPFold, predict_forces
from model.ema import ExponentialMovingAverage
from data.dataset import
from utils.dataset_utils import collate, collate_multiple_coords
from utils.training_utils import sample_noise, dsm
from time import time
import datetime
import torch
import sys
import argparse
import numpy as np
import os

torch.autograd.set_detect_anomaly(True)


def to_cuda(features):
    for feature in features:

        if "atom_names" not in feature and "cyclic_bond_indices" not in feature:
            features[feature] = features[feature].cuda()

    return features


def train(model, model_type, epochs, output_file, batch_size, lr, sde, ema_decay,
          gradient_clip=None, eps=1e-5, saved_params=None, data_path="./",):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if not saved_params is None:
        optimizer.load_state_dict(torch.load(saved_params)["optimizer_state_dict"])

    if ema_decay is not None:

        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

        if saved_params is not None:
            ema.load_state_dict(torch.load(saved_params)["ema_state_dict"])

    # Dataset loading


    start_time = time()

    log_step = 100

    validation_losses = []

    iters = len(loader)

    for e in range(epochs):

        model.train()
        losses = []

        for value, features in enumerate(loader):

            torch.cuda.empty_cache()

            optimizer.zero_grad()

            z, t, perturbed_data, mean, std = sample_noise(sde, features["coordinates_backbone"],
                                                           features["atom_mask_backbone"],
                                                           device="cuda")

            score_fn = model.get_score_fn(to_cuda(features), sde)

            prediction = score_fn(perturbed_data, t)
            all_losses, loss = dsm(prediction, std, z,
                                   features["atom_mask_backbone"])

            for index, i in enumerate(all_losses):
                losses.append(torch.sum(i).cpu().detach().numpy())

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

            if (value + 1) % 10000 == 0 or (value == iters - 1):
                model.eval()

                losses = []

                if ema_decay is not None:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())

                for value_val, features in enumerate(validation_loader):

                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        z, t, perturbed_data, mean, std = sample_noise(sde, features["coordinates_backbone"],
                                                                       features["atom_mask_backbone"],
                                                                       device="cuda")

                        score_fn = model.get_score_fn(to_cuda(features), sde)

                        prediction = score_fn(perturbed_data, t)
                        all_losses, loss = dsm(prediction, std, z,
                                               features["atom_mask_backbone"])

                        for index, i in enumerate(all_losses):
                            losses.append(torch.sum(i).cpu().detach().numpy())

                if ema_decay is not None:
                    ema.restore(model.parameters())

                losses = torch.stack(losses, dim=0)

                losses = losses.cpu().numpy()

                validation_losses.append(np.mean(losses))

                log = "This is validation, Epoch [{}/{}]".format(
                    e + 1, epochs)

                log += ", {}: {:.5f}".format('Loss', np.mean(losses))
                log += ", {}: {:.5f}".format('Std', np.std(losses))

                if validation_losses[-1] == min(validation_losses):

                    param_dict = {"model_state_dict": model.state_dict(),
                                  "optimizer_state_dict": optimizer.state_dict()
                                  }

                    if ema_decay != None:
                        param_dict["ema_state_dict"] = ema.state_dict()

                    print("Saving model with new minimum validation loss")
                    torch.save(param_dict, output_file)
                    print("Saved model successfully!")

                print(log)

                if (value + 1) % 100000 == 0:
                    if not os.path.isdir("checkpoints_" + model_type):
                        os.mkdir("checkpoints_" + model_type)

                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()
                                }, "checkpoints_" + model_type + "/" + \
                               output_file.replace(".pth", "_" + str(value + 1) + ".pth"))

                losses = []

                sys.stdout.flush()
                model.train()


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
                        type=str, default="./")
    parser.add_argument("-ep", dest="epochs", help="Number of epochs",
                        required=False, type=int, default=10)
    parser.add_argument("-dt", dest="dataset", type=str,
                        help="Dataset to train on, fragment or MD",
                        required=False, default="fragment")
    args = parser.parse_args()

    config = config_rotamer
    model = IDPFold(config.model_config)

    sde = config.sde_config.sde(beta_min=config.sde_config.beta_min,
                                beta_max=config.sde_config.beta_max)

    torch.cuda.empty_cache()

    if args.use_saved:
        model.load_state_dict(torch.load(args.saved_model)["model_state_dict"])

    model.cuda()

    if args.use_saved:
        train(model, args.model, args.epochs, args.output_file,
              config.training.batch_size, config.training.lr, sde,
              ema_decay=config.training.ema, gradient_clip=config.training.gradient_clip,
              saved_params=args.saved_model, data_path=args.data_path,
              dataset_type=args.dataset)
    else:

        train(model, args.model, args.epochs, args.output_file, config.training.batch_size,
              config.training.lr, sde, ema_decay=config.training.ema, gradient_clip=config.training.gradient_clip,
              data_path=args.data_path, dataset_type=args.dataset)

import datetime
import os
import random

os.environ["CUDA_VISIBLE_DEVICES"]="0"
from models.FNO import FNO2d, FNO3d
from models.PeFNN import PeFNN
from models.Unet import UNet2d, UNet3d

from PIL import Image
import imageio
import matplotlib.pyplot as plt

from utils import pde_data, LpLoss

import scipy
import numpy as np
from timeit import default_timer
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import h5py
import xarray as xr
from tqdm import tqdm
torch.set_num_threads(1)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def get_eval_pred(model, x, strategy, T, times):

    if strategy == "oneshot":
        pred = model(x)
    else:

        for t in range(T):
            t1 = default_timer()
            im = model(x)
            times.append(default_timer() - t1)
            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -2)
            if strategy == "markov":
                x = im
            else:
                x = torch.cat((x[..., 1:, :], im), dim=-2)

    return pred

################################################################
# configs
################################################################

parser = argparse.ArgumentParser()

parser.add_argument("--results_path", type=str, default="/PaTH/", help="path to store results")
parser.add_argument("--suffix", type=str, default="seed1", help="suffix to add to the results path")
parser.add_argument("--txt_suffix", type=str, default="Flood_PeFNN_seed1", help="suffix to add to the results txt")
parser.add_argument("--data_path", type=str, default='PATH', help="path to the data")
parser.add_argument("--super_path", type=str, default='PATH', help="path to the superresolution data")
parser.add_argument("--super", type=str, default='False', help="enable superres testing")
parser.add_argument("--verbose",type=str, default='True')

parser.add_argument("--T", type=int, default=39, help="number of timesteps to predict")
parser.add_argument("--ntrain", type=int, default=720, help="training sample size")
parser.add_argument("--nvalid", type=int, default=144, help="valid sample size")
parser.add_argument("--ntest", type=int, default=144, help="test sample size")
parser.add_argument("--nsuper", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--model_type", type=str, default='PeFNN')
parser.add_argument("--depth", type=int, default=4)
parser.add_argument("--modes", type=int, default=12)
parser.add_argument("--width", type=int, default=10)
parser.add_argument("--Gwidth", type=int, default=10, help="hidden dimension of equivariant layers if model_type=hybrid")
parser.add_argument("--grid", type=str, default=None, help="[symmetric, cartesian]")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--early_stopping", type=int, default=100, help="stop if validation error does not improve for successive epochs")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--step", action="store_true", help="use step scheduler")
parser.add_argument("--gamma", type=float, default=None, help="gamma for step scheduler")
parser.add_argument("--step_size", type=int, default=None, help="step size for step scheduler")
parser.add_argument("--lmbda", type=float, default=0.0001, help="weight decay for adam")
parser.add_argument("--strategy", type=str, default="markov", help="markov, recurrent or oneshot")
parser.add_argument("--time_pad", type=str, default='False', help="pad the time dimension for strategy=oneshot")
parser.add_argument("--noise_std", type=float, default=0.00, help="amount of noise to inject for strategy=markov")

args = parser.parse_args()

assert args.model_type in ["FNO2d", "FNO3d",
                           "PeFNN", "GFNO2d_p4",
                           "GFNO3d_p4", "PeRCNN",
                           "UNet2d", "UNet3d"], f"Invalid model type {args.model_type}"
assert args.strategy in ["markov", "recurrent", "oneshot"], "Invalid training strategy"

torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)

TRAIN_PATH = args.data_path

# FNO data specs
Sy = 810
Sx = 441
S = 64 # spatial res
S_super = 4 * S # super spatial res
T_in = 1 # number of input times
T = args.T
T_super = 4 * T # prediction temporal super res
d = 2 # spatial res
num_channels = 1

# adjust data specs based on model type and data path
threeD = args.model_type in ["FNO3d",
                             "GFNO3d_p4",
                             "UNet3d"]
extension = TRAIN_PATH.split(".")[-1]
swe = False
rdb = False
grid_type = "cartesian"

ntrain = args.ntrain
nvalid = args.nvalid
ntest = args.ntest

time_modes = None
time = args.strategy == "oneshot" # perform convolutions in space-time
if time and not args.time_pad:
    time_modes = 6
elif time:
    time_modes = 8

modes = args.modes
width = args.width
n_layer = args.depth
batch_size = args.batch_size

epochs = args.epochs
learning_rate = args.learning_rate
scheduler_step = args.step_size
scheduler_gamma = args.gamma

initial_step = 1 if args.strategy == "markov" else T_in

root = args.results_path + f"/{'_'.join(str(datetime.datetime.now()).split())}"
if args.suffix:
    root += "_" + args.suffix
os.makedirs(root)
path_model = os.path.join(root, 'model.pt')
writer = SummaryWriter(root)

################################################################
# Model init
################################################################
if args.model_type in ["FNO2d"]:
    model = FNO2d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, width=width,
                  grid_type=grid_type).cuda()
elif args.model_type in ["FNO3d"]:
    modes3 = time_modes if time_modes else modes
    model = FNO3d(num_channels=num_channels, initial_step=initial_step, modes1=modes, modes2=modes, modes3=modes3,
                  width=width, time=time, time_pad=args.time_pad).cuda()
elif "PeFNN" in args.model_type:
    model = PeFNN(num_channels=num_channels, initial_step=initial_step, modes=modes, width=width, grid_type=grid_type).cuda()

elif "GFNO2d" in args.model_type:
    model = GFNO2d(num_channels=num_channels, initial_step=initial_step, modes=modes, width=width,
                   reflection=False, grid_type=grid_type).cuda()
elif "GFNO3d" in args.model_type:
    model = GFNO3d(num_channels=num_channels, initial_step=initial_step, modes=modes, time_modes=time_modes,
                   width=width, reflection=False, grid_type=grid_type, time_pad=args.time_pad).cuda()
elif "UNet2d" in args.model_type:
    model = UNet2d(in_channels=initial_step * num_channels, out_channels=num_channels, init_features=32,
                   grid_type=grid_type).cuda()
elif "UNet3d" in args.model_type:
    model = UNet3d(in_channels=initial_step * num_channels, out_channels=num_channels, init_features=32,
                   grid_type=grid_type, time=time, time_pad=args.time_pad).cuda()
else:
    raise NotImplementedError("Model not recognized")


################################################################
# load data
################################################################
full_data = None # for superres

def data_load(i):
    t0 = 40
    days = 7
    # T = 86400
    dt0 = 30
    l = int((days * 24 * 60 * 60) / dt0)
    t00, tfinal = (i * (t0)) * dt0, ((i+1)*(t0)) * dt0
    dx = 30.0 * 16
    # # model
    resolution = dx
    # target_time = tfinal
    path = "/Path/To/Flood/"
    h_gt = []
    for i in range(t00,tfinal,dt0):
        current_time = torch.tensor(i, dtype=torch.float).to(device)
        print('current_time', current_time)
        path_h = os.path.join(path, "h_gt", str(current_time)+".tiff")
        h_current = Image.open(path_h)
        h_current = np.array(h_current)
        h_current = torch.from_numpy(h_current)
        h_current = h_current.float()
        h_current = torch.unsqueeze(h_current, -1)
        h_gt.append(h_current)
    h_gt = torch.stack(h_gt, 0)
    print('shape_every_batch', h_gt.size())
    return h_gt

days = 14
dt = 40
l = int((days * 24 * 60 * 60) / dt)
t00 = 40
nt = int(l / t00)
h_gt_list = []
for i in range(nt):
    # model.train()
    h_gt = data_load(i)
    h_gt_list.append(h_gt)
data = torch.stack(h_gt_list, 0).permute(0, 2, 3, 1, 4)
print('shape_supervised', data.size())
print('len of supervised', len(data))


assert len(data) >= ntrain + nvalid + ntest, f"not enough data; {len(data)}"

train = data[:ntrain]
assert len(train) == ntrain, "not enough training data"

test = data[-ntest:]
assert len(test) == ntest, "not enough test data"

valid = data[-(ntest + nvalid):-ntest]
assert len(valid) == nvalid, "not enough validation data"

# if args.verbose:
print(f"{args.model_type}: Train/valid/test data shape: ")
print(train.shape)
print(valid.shape)
print(test.shape)


train_data = pde_data(train, strategy=args.strategy, T_in=T_in, T_out=T, std=args.noise_std)
ntrain = len(train_data)
valid_data = pde_data(valid, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
nvalid = len(valid_data)
test_data = pde_data(test, train=False, strategy=args.strategy, T_in=T_in, T_out=T)
ntest = len(test_data)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

################################################################
# training and evaluation
################################################################

complex_ct = sum(par.numel() * (1 + par.is_complex()) for par in model.parameters())
real_ct = sum(par.numel() for par in model.parameters())
if args.verbose:
    print(f"{args.model_type}; # Params: complex count {complex_ct}, real count: {real_ct}")
writer.add_scalar("Parameters/Complex", complex_ct)
writer.add_scalar("Parameters/Real", real_ct)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.lmbda)
if args.step:
    assert args.step_size is not None, "step_size is None"
    assert scheduler_gamma is not None, "gamma is None"
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=scheduler_gamma)
else:
    num_training_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

lploss = LpLoss(size_average=False)

best_valid = float("inf")

x_train, y_train = next(iter(train_loader))
x = x_train.cuda()
y = y_train.cuda()
x_valid, y_valid = next(iter(valid_loader))
if args.verbose:
    print(f"{args.model_type}; Input shape: {x.shape}, Target shape: {y.shape}")
if args.strategy == "oneshot":
    assert x_train[0].shape == torch.Size([Sy, Sx, T, T_in, num_channels]), x_train[0].shape
    assert y_train[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_train[0].shape
    assert x_valid[0].shape == torch.Size([Sy, Sx, T, T_in, num_channels]), x_valid[0].shape
    assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
elif args.strategy == "markov":
    assert x_train[0].shape == torch.Size([Sy, Sx, 1, num_channels]), x_train[0].shape
    assert y_train[0].shape == torch.Size([Sy, Sx, num_channels]), y_train[0].shape
    assert x_valid[0].shape == torch.Size([Sy, Sx, 1, num_channels]), x_valid[0].shape
    assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
else: # strategy == recurrent
    assert x_train[0].shape == torch.Size([Sy, Sx, T_in, num_channels]), x_train[0].shape
    assert x_valid[0].shape == torch.Size([Sy, Sx, T_in, num_channels]), x_valid[0].shape
    assert y_valid[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_valid[0].shape
    if args.strategy == "recurrent":
        assert y_train[0].shape == torch.Size([Sy, Sx, T, num_channels]), y_train[0].shape

model.eval()
start = default_timer()
if args.verbose:
    print("Training...")
step_ct = 0
train_times = []
eval_times = []
for ep in range(epochs):
    model.train()
    t1 = default_timer()

    train_l2 = 0

    for xx, yy in tqdm(train_loader, disable=not args.verbose):
        loss = 0
        xx = xx.cuda()
        yy = yy.cuda()

        if args.strategy == "recurrent":
            for t in range(yy.shape[-2]):
                y = yy[..., t, :]
                im = model(xx)
                loss += lploss(im.reshape(len(im), -1, num_channels), y.reshape(len(y), -1, num_channels))
                xx = torch.cat((xx[..., 1:, :], im), dim=-2)
            loss /= yy.shape[-2]
        else:
            im = model(xx)
            if args.strategy == "oneshot":
                im = im.squeeze(-1)
            loss = lploss(im.reshape(len(im), -1, num_channels), yy.reshape(len(yy), -1, num_channels))

        train_l2 += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not args.step:
            scheduler.step()
        writer.add_scalar("Train/LR", scheduler.get_last_lr()[0], step_ct)
        step_ct += 1
    if args.step:
        scheduler.step()

    train_times.append(default_timer() - t1)

    # validation
    valid_l2 = 0
    valid_loss_by_channel = None
    with torch.no_grad():
        model.eval()
        model(xx)
        for xx, yy in valid_loader:

            xx = xx.cuda()
            yy = yy.cuda()

            pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=eval_times).view(len(xx), Sy, Sx, T, num_channels)

            valid_l2 += lploss(pred.reshape(len(pred), -1, num_channels), yy.reshape(len(yy), -1, num_channels)).item()

    t2 = default_timer()
    if args.verbose:
        print(f"Ep: {ep}, time: {t2 - t1}, train: {train_l2 / ntrain}, valid: {valid_l2 / nvalid}")

    writer.add_scalar("Train/Loss", train_l2 / ntrain, ep)
    writer.add_scalar("Valid/Loss", valid_l2 / nvalid, ep)

    if valid_l2 < best_valid:
        best_epoch = ep
        best_valid = valid_l2
        torch.save(model.state_dict(), path_model)
    if args.early_stopping:
        if ep - best_epoch > args.early_stopping:
            break

stop = default_timer()
train_time = stop - start
train_times = torch.tensor(train_times).mean().item()
num_eval = len(eval_times)
eval_times = torch.tensor(eval_times).mean().item()
model.eval()
# test
def generate_movie_2D(key, test_y, preds_y, plot_title='', field=0, val_cbar_index=-1, err_cbar_index=-1,
                      val_clim=None, err_clim=None, font_size=None, movie_dir='', movie_name='movie.gif',
                      frame_basename='movie', frame_ext='jpg', remove_frames=True):
    frame_files = []

    if movie_dir:
        os.makedirs(movie_dir, exist_ok=True)

    if font_size is not None:
        plt.rcParams.update({'font.size': font_size})

    if len(preds_y.shape) == 4:
        Nsamples, Nx, Ny, Nt = preds_y.shape
        preds_y = preds_y.reshape(Nsamples, Nx, Ny, Nt, 1)
        test_y = test_y.reshape(Nsamples, Nx, Ny, Nt, 1)
    Nsamples, Nx, Ny, Nt, Nfields = preds_y.shape
    print('preds_y', preds_y.shape)

    pred = preds_y[key, ..., field]
    true = test_y[key, ..., field]
    error = pred - true

    # a = test_x[key]
    x = torch.linspace(0, Nx + 1, Nx + 1)[:-1]
    y = torch.linspace(0, Ny + 1, Ny + 1)[:-1]
    X, Y = torch.meshgrid(x, y)
    # t = a[0, 0, :, 2]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    pcm1 = ax1.pcolormesh(X, Y, true[..., val_cbar_index], cmap='plasma', label='true', shading='gouraud')
    pcm2 = ax2.pcolormesh(X, Y, pred[..., val_cbar_index], cmap='plasma', label='pred', shading='gouraud')
    pcm3 = ax3.pcolormesh(X, Y, error[..., err_cbar_index], cmap='plasma', label='error', shading='gouraud')

    if val_clim is None:
        val_clim = pcm1.get_clim()
    if err_clim is None:
        err_clim = pcm3.get_clim()

    pcm1.set_clim(val_clim)
    plt.colorbar(pcm1, ax=ax1)
    ax1.set_aspect('auto')

    pcm2.set_clim(val_clim)
    plt.colorbar(pcm2, ax=ax2)
    ax2.set_aspect('auto')

    pcm3.set_clim(err_clim)
    plt.colorbar(pcm3, ax=ax3)
    ax3.set_aspect('auto')

    plt.tight_layout()

    for i in range(Nt):
        # Exact
        ax1.clear()
        pcm1 = ax1.pcolormesh(X, Y, true[..., i], cmap='jet', label='true', shading='gouraud')
        pcm1.set_clim(val_clim)
        # ax1.set_title(f'Ground Truth')
        ax1.set_aspect('auto')
        ax1.set_adjustable('datalim')

        # Predictions
        ax2.clear()
        pcm2 = ax2.pcolormesh(X, Y, pred[..., i], cmap='jet', label='pred', shading='gouraud')
        pcm2.set_clim(val_clim)
        # ax2.set_title(f'PeFNN')
        ax2.set_aspect('auto')
        ax2.set_adjustable('datalim')

        # Error
        ax3.clear()
        pcm3 = ax3.pcolormesh(X, Y, error[..., i], cmap='jet', label='error', shading='gouraud')
        pcm3.set_clim(err_clim)
        # ax3.set_title(f'Error')
        ax3.set_aspect('auto')
        ax3.set_adjustable('datalim')

        #         plt.tight_layout()
        fig.canvas.draw()

        if movie_dir:
            frame_path = os.path.join(movie_dir, f'{frame_basename}-{i:03}.{frame_ext}')
            frame_files.append(frame_path)
            plt.savefig(frame_path)

    if movie_dir:
        movie_path = os.path.join(movie_dir, movie_name)
        with imageio.get_writer(movie_path, mode='I') as writer:
            for frame in frame_files:
                image = imageio.imread(frame)
                writer.append_data(image)

    # if movie_dir and remove_frames:
    #     for frame in frame_files:
    #         try:
    #             os.remove(frame)
    #         except:
    #             pass
model.load_state_dict(torch.load(path_model))
model.eval()
test_l2 = test_vort_l2 = test_pres_l2 = 0
rotations_l2 = 0
reflections_l2 = 0
test_loss_by_channel = None
key = 0
i = 0
with torch.no_grad():
    for xx, yy in test_loader:
        xx = xx.cuda()
        yy = yy.cuda()
        pred = get_eval_pred(model=model, x=xx, strategy=args.strategy, T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels)
        test_l2 += lploss(pred.reshape(len(pred), -1, num_channels), yy.reshape(len(yy), -1, num_channels)).item()

print(f"{args.model_type} done training; \nTest: {test_l2 / ntest}")
summary = f"Args: {str(args)}" \
          f"\nParameters: {complex_ct}" \
          f"\nTrain time: {train_time}" \
          f"\nMean epoch time: {train_times}" \
          f"\nMean inference time: {eval_times}" \
          f"\nNum inferences: {num_eval}" \
          f"\nTrain: {train_l2 / ntrain}" \
          f"\nValid: {valid_l2 / nvalid}" \
          f"\nTest: {test_l2 / ntest}" \
          f"\nBest Valid: {best_valid / nvalid}" \
          f"\nBest epoch: {best_epoch + 1}" \
          f"\nEpochs trained: {ep}"
txt = "results"
if args.txt_suffix:
    txt += f"_{args.txt_suffix}"
txt += ".txt"

with open(os.path.join(root, txt), 'w') as f:
    f.write(summary)
writer.flush()
writer.close()
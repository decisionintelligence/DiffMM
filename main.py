from utils.mbr import MBR
from utils.map import RoadNetworkMapFull
from utils.spatial_func import SPoint
from utils.model_utils import AttrDict, gps2grid, get_rn_grid
from utils.evaluation_utils import cal_id_acc

from models.model import TrajEncoder, ModelAll, ModelAllShortCut, init_weights
from models.diffusion import GaussianDiffusion, DenoiseNet
from models.short_cut import ShortCut, DiT, get_targets

import os
import random
import pickle
import numpy as np
import torch
import torch.optim as optim
import time
import logging
import argparse
from tqdm import tqdm
import multiprocessing

from dataset import MMDataset, collate_fn
from torch.utils.data import DataLoader


def init_setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def batch2model(batch, encoder, device):
    lengths, norm_gps_seq, trg_rid, trg_onehot, segs_id, segs_feat, segs_mask = map(lambda x: x.to(device), batch)

    enc_out = encoder(norm_gps_seq, lengths, segs_id, segs_feat, segs_mask)
    # assert enc_out.size(0) == trg_rid.size(0)

    traj_cond = []
    trg_rid_diff = []
    trg_onehot_diff = []
    src_segs_id = []
    src_segs_mask = []

    for index in range(enc_out.shape[0]):
        length = lengths[index].item()
        if length > 0:
            traj_cond += [i.unsqueeze(0) for i in enc_out[index][:length]]
            trg_rid_diff += [i.unsqueeze(0) for i in trg_rid[index][:length]]
            trg_onehot_diff += [i.unsqueeze(0) for i in trg_onehot[index][:length]]
            src_segs_id += [i.unsqueeze(0) for i in segs_id[index][:length]]
            src_segs_mask += [i.unsqueeze(0) for i in segs_mask[index][:length]]

    traj_cond = torch.cat(traj_cond, dim=0)
    trg_rid_diff = torch.cat(trg_rid_diff, dim=0)
    trg_onehot_diff = torch.cat(trg_onehot_diff, dim=0)
    src_segs_id = torch.cat(src_segs_id, dim=0)
    src_segs_mask = torch.cat(src_segs_mask, dim=0)

    trg_rid_diff = trg_rid_diff.reshape(-1, 1, 1)
    trg_onehot_diff = trg_onehot_diff.reshape(trg_rid_diff.shape[0], 1, -1)
    traj_cond = traj_cond.reshape(trg_rid_diff.shape[0], 1, -1)
    src_segs_id = src_segs_id.reshape(trg_rid_diff.shape[0], 1, -1)
    src_segs_mask = src_segs_mask.reshape(trg_rid_diff.shape[0], 1, -1)

    diff_mask = torch.zeros((trg_rid_diff.shape[0], 1, encoder.id_size-1), device=device)
    for i, src_segs in enumerate(src_segs_id):
        seg_num = src_segs_mask[i, 0].sum().item()
        diff_mask[i, 0, src_segs[0, :seg_num]-1] = 1

    trg_rid_diff.to(device)
    trg_onehot_diff.to(device)
    traj_cond.to(device)

    return traj_cond, trg_rid_diff, trg_onehot_diff, lengths, src_segs_id, src_segs_mask, diff_mask


def lr_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num


def cal_eval(pred, trg):
    acc, recall, precision, f1 = [], [], [], []
    for pred_single, trg_single in zip(pred, trg):
        acc_tmp, recall_tmp, precision_tmp, f1_tmp = cal_id_acc(pred_single, trg_single)
        acc.append(acc_tmp)
        recall.append(recall_tmp)
        precision.append(precision_tmp)
        f1.append(f1_tmp)
    acc, recall, precision, f1 = np.mean(acc), np.mean(recall), np.mean(precision), np.mean(f1)
    return acc, recall, precision, f1

def train_shortcut(model, iterator, optimizer, denoise_steps, device, bootstrap_every=8):
    model.train()
    
    loss_all, total_num = 0.0, 0.0

    for i, batch in enumerate(tqdm(iterator, desc='train')):
        traj_cond, trg_rid, trg_onehot, _, _, _, mask = batch2model(batch, model.encoder, device)
        x_t, v_t, t, dt_base = get_targets(model.shortcut.model, trg_onehot, traj_cond, denoise_steps, device, mask, bootstrap_every)

        model.train()
        loss = model.shortcut(x_t, v_t, t, dt_base, traj_cond, trg_onehot, mask)

        optimizer.zero_grad()
        loss.backward()

        loss_all += loss.item() * traj_cond.shape[0]

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        total_num += traj_cond.shape[0]

        if i % 200 == 0:
            logging.info(f'step: {i}, train loss: {loss_all / total_num}')

    return loss_all / total_num


def train(model, iterator, optimizer, device):
    model.train()

    loss_all, vb_all, total_num = 0.0, 0.0, 0.0
    for i, batch in enumerate(tqdm(iterator, desc='train')):
        traj_cond, trg_rid, trg_onehot, _, _, _, mask = batch2model(batch, model.encoder, device)
        loss = model.diffusion(trg_onehot, traj_cond, segs_mask=mask)

        optimizer.zero_grad()
        loss.backward()

        loss_all += loss.item() * traj_cond.shape[0]

        # vb = model.diffusion.NLL_cal(trg_onehot, traj_cond, segs_mask=mask)
        # vb_all += vb

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

        total_num += traj_cond.shape[0]

        if i % 200 == 0:
            logging.info(f'step: {i}, train loss: {loss_all / total_num}, vb: {vb_all / total_num}')

    return loss_all / total_num, vb_all / total_num

def validate_shortcut(model, iterator, device):
    with torch.no_grad():
        model.eval()

        total_num = 0.0
        pred, trg = [], []
        acc, recall, precision, f1 = 0.0, 0.0, 0.0, 0.0
        for batch in tqdm(iterator, desc='val'):
            traj_cond, trg_rid, trg_onehot, lengths, segs_id, segs_mask, mask = batch2model(batch, model.encoder, device)

            sampled_seq = model.shortcut.inference(batch_size=traj_cond.shape[0], cond=traj_cond, segs_mask=mask)

            cur_len = 0
            for length in lengths.detach().cpu().tolist():
                tmp_pred, tmp_trg = [], []
                for i in range(length):
                    tmp_pred.append(torch.argmax(sampled_seq[cur_len + i, 0]).item())
                    tmp_trg.append(trg_rid[cur_len + i, 0].int().item())
                pred.append(tmp_pred)
                trg.append(tmp_trg)
                cur_len += length
            # assert len(pred) == len(trg)

            total_num += sampled_seq.shape[0]

        # logging.info(f'avg segs: {total_segs / total_num}')
        acc, recall, precision, f1 = cal_eval(pred, trg)

        return acc, recall, precision, f1


def validate(model, iterator, device):
    with torch.no_grad():
        model.eval()

        loss_all, vb_all, total_num = 0.0, 0.0, 0.0
        pred, trg = [], []
        # total_segs = 0
        acc, recall, precision, f1 = 0.0, 0.0, 0.0, 0.0
        for batch in tqdm(iterator, desc='val'):
            traj_cond, trg_rid, trg_onehot, lengths, segs_id, segs_mask, mask = batch2model(batch, model.encoder, device)

            sampled_seq = model.diffusion.sample(batch_size=traj_cond.shape[0], cond=traj_cond, segs_mask=mask)

            loss = model.diffusion(trg_onehot, traj_cond, segs_mask=mask)
            # vb = model.diffusion.NLL_cal(trg_onehot, traj_cond, segs_mask=mask)

            # vb_all += vb
            loss_all += loss.item() * traj_cond.shape[0]

            # assert sampled_seq.shape[0] == trg_rid.shape[0]
    
            cur_len = 0
            for length in lengths.detach().cpu().tolist():
                tmp_pred, tmp_trg = [], []
                for i in range(length):
                    # seg_num = torch.sum(segs_mask[cur_len + i, 0]).item()
                    # total_segs += seg_num
                    # segs = segs_id[cur_len + i, 0, :seg_num] - 1
                    # mask = torch.zeros(sampled_seq.shape[2], dtype=torch.int8, device=device)
                    # mask[segs] = 1
                    # tmp_sample_seq = sampled_seq[cur_len + i, 0].clone().detach().to(device).masked_fill(mask == 0, -1e9)
                    # tmp_sample_seq = torch.softmax(tmp_sample_seq, dim=0)
                    tmp_pred.append(torch.argmax(sampled_seq[cur_len + i, 0]).item())
                    tmp_trg.append(trg_rid[cur_len + i, 0].int().item())
                pred.append(tmp_pred)
                trg.append(tmp_trg)
                cur_len += length
            # assert len(pred) == len(trg)

            total_num += sampled_seq.shape[0]

        # logging.info(f'avg segs: {total_segs / total_num}')
        acc, recall, precision, f1 = cal_eval(pred, trg)

        return loss_all, loss_all / total_num, vb_all / total_num, acc, recall, precision, f1


def testing_shortcut(model, iterator, device):
    with torch.no_grad():
        model.eval()

        total_num = 0.0
        pred, trg = [], []
        acc, recall, precision, f1 = 0.0, 0.0, 0.0, 0.0
        for batch in tqdm(iterator, desc='test'):
            traj_cond, trg_rid, trg_onehot, lengths, segs_id, segs_mask, mask = batch2model(batch, model.encoder, device)

            sampled_seq = model.shortcut.inference(batch_size=traj_cond.shape[0], cond=traj_cond, segs_mask=mask)

            cur_len = 0
            for length in lengths.detach().cpu().tolist():
                tmp_pred, tmp_trg = [], []
                for i in range(length):
                    tmp_pred.append(torch.argmax(sampled_seq[cur_len + i, 0]).item())
                    tmp_trg.append(trg_rid[cur_len + i, 0].int().item())
                pred.append(tmp_pred)
                trg.append(tmp_trg)
                cur_len += length
            # assert len(pred) == len(trg)

            total_num += sampled_seq.shape[0]

        # logging.info(f'avg segs: {total_segs / total_num}')
        acc, recall, precision, f1 = cal_eval(pred, trg)

        return acc, recall, precision, f1


def testing(model, iterator, device):
    with torch.no_grad():
        model.eval()

        loss_all, vb_all, total_num = 0.0, 0.0, 0.0
        pred, trg = [], []
        acc, recall, precision, f1 = 0.0, 0.0, 0.0, 0.0
        for batch in tqdm(iterator, desc='test'):
            traj_cond, trg_rid, trg_onehot, lengths, segs_id, segs_mask, mask = batch2model(batch, model.encoder, device)

            sampled_seq = model.diffusion.sample(batch_size=traj_cond.shape[0], cond=traj_cond, segs_mask=mask)

            loss = model.diffusion(trg_onehot, traj_cond, segs_mask=mask)
            # vb = model.diffusion.NLL_cal(trg_onehot, traj_cond, segs_mask=mask)

            # forward = []
            # forward.append(trg_onehot.clone().detach().cpu().numpy())
            # for t in range(200):
            #     def unnormalize_to_zero_to_one(t):
            #         return (t + 1) * 0.5
            #     def normalize_to_neg_one_to_one(img):
            #         return img * 2 - 1
                
            #     tt = torch.tensor(t, device=device).long().reshape((1,))
            #     if t == 0:
            #         s = normalize_to_neg_one_to_one(trg_onehot)
            #         forward.append(unnormalize_to_zero_to_one(model.diffusion.q_sample(x_start=s, t=tt).clone().detach().cpu().numpy()))
            #     if (t + 1) % 25 == 0:
            #         s = normalize_to_neg_one_to_one(trg_onehot)
            #         forward.append(unnormalize_to_zero_to_one(model.diffusion.q_sample(x_start=s, t=tt).clone().detach().cpu().numpy()))

            # pickle.dump(forward, open('forward.pkl', 'wb'))

            # vb_all += vb
            loss_all += loss.item() * traj_cond.shape[0]

            # assert sampled_seq.shape[0] == trg_rid.shape[0]

            cur_len = 0
            for length in lengths.detach().cpu().tolist():
                tmp_pred, tmp_trg = [], []
                for i in range(length):
                    # seg_num = torch.sum(segs_mask[cur_len + i, 0]).item()
                    # segs = segs_id[cur_len + i, 0, :seg_num] - 1
                    # mask = torch.zeros(sampled_seq.shape[2], dtype=torch.int8, device=device)
                    # mask[segs] = 1
                    # tmp_sample_seq = sampled_seq[cur_len + i, 0].clone().detach().to(device).masked_fill(mask == 0, -1e9)
                    # tmp_sample_seq = torch.softmax(tmp_sample_seq, dim=0)
                    # logging.info(f"tmp_sample_seq: {tmp_sample_seq}, {tmp_sample_seq.shape}")
                    tmp_pred.append(torch.argmax(sampled_seq[cur_len + i, 0]).item())
                    tmp_trg.append(trg_rid[cur_len + i, 0].int().item())
                    # logging.info(f'{tmp_pred[-1]}, {tmp_trg[-1]}')
                pred.append(tmp_pred)
                trg.append(tmp_trg)
                cur_len += length
            # assert len(pred) == len(trg)

            # logging.info(f'{pred[0]}, {trg[0]}')

            total_num += sampled_seq.shape[0]

        acc, recall, precision, f1 = cal_eval(pred, trg)

        return loss_all, loss_all / total_num, vb_all / total_num, acc, recall, precision, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--city', type=str, default='porto')
    parser.add_argument('--keep_ratio', type=float, default=0.1)
    parser.add_argument('--hid_dim', type=int, default=256, help='hidden dimension')
    parser.add_argument('--num_units', type=int, default=512, help='denoising net units')
    parser.add_argument('--grid_size', type=int, default=50, help='grid size in int')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=30, help='epochs')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--transformer_layers', type=int, default=2)
    parser.add_argument('--timesteps', type=int, default=2)
    parser.add_argument('--samplingsteps', type=int, default=1)
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'])
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'Euclid'], help='')
    parser.add_argument('--objective', type=str, default='pred_x0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--train_flag', action='store_true', help='flag of training')
    parser.add_argument('--test_flag', action='store_true', help='flag of testing')
    parser.add_argument('--pin_memory', action='store_true', help='whether to set pin_memory=True')
    parser.add_argument('--load_model', type=str, default='', help='load model path')
    parser.add_argument('--search_dist', type=int, default=50)

    global args
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Use GPU: cuda {args.gpu_id}')

    if args.num_workers > 0:
        multiprocessing.set_start_method('spawn')

    seed = args.seed
    init_setup(seed)

    model_save_root = os.path.join(os.getcwd(), 'model', 'shortcut', args.city)
    model_save_path = os.path.join(model_save_root, args.city + '_keep-ratio_' +
                                   str(args.keep_ratio) + '_' + time.strftime("%Y%m%d-%H%M%S"))

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filename=os.path.join(model_save_path, 'log.txt'),
                        filemode='a')

    city = args.city
    if city == 'porto':
        zone_range = [41.1395, -8.6911, 41.1864, -8.5521]
        ts = 15
        utc = 1
    elif city == 'beijing':
        zone_range = [39.7547, 116.1994, 40.0244, 116.5452]
        ts = 60
        utc = 0
    else:
        raise NotImplementedError

    print('Preparing Data...')
    logging.info('Preparing Data...')

    rn_root = os.path.join(os.getcwd(), 'data', args.city, 'roadnetwork')
    rn = RoadNetworkMapFull(rn_root, zone_range=zone_range, unit_length=50)

    parameters = AttrDict()
    args_dict = {
        'device': device,

        'transformer_layers': args.transformer_layers,
        'depth': 2,

        'search_dist': args.search_dist,
        'beta': 15,

        # MBR
        'min_lat': zone_range[0],
        'min_lng': zone_range[1],
        'max_lat': zone_range[2],
        'max_lng': zone_range[3],

        # input data params
        'city': args.city,
        'keep_ratio': args.keep_ratio,
        'grid_size': args.grid_size,
        'time_span': ts,

        # model params
        'hid_dim': args.hid_dim,
        'num_units': args.num_units,
        'id_emb_dim': int(args.hid_dim / 2),
        'dropout': 0.1,
        'id_size': rn.valid_edge_cnt_one,
        'timesteps': args.timesteps,
        'samplingsteps': args.samplingsteps,
        'beta_schedule': args.beta_schedule,
        'objective': args.objective,
        'loss_type': args.loss_type,


        # train config
        'n_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_workers': args.num_workers,

        'utc': utc
    }

    parameters.update(args_dict)

    mbr = MBR(parameters.min_lat, parameters.min_lng, parameters.max_lat, parameters.max_lng)
    logging.info(parameters)

    # traj data
    traj_root = os.path.join(os.getcwd(), 'data', args.city)

    if args.train_flag:
        logging.info('loading train data...')
        train_dataset = MMDataset(rn, traj_root, mbr, parameters, 'train', device)

        logging.info('loading valid data...')
        valid_dataset = MMDataset(rn, traj_root, mbr, parameters, 'valid', device)

        print('training dataset shape: ' + str(len(train_dataset)))
        print('validation dataset shape: ' + str(len(valid_dataset)))
        logging.info('train dataset shape: ' + str(len(train_dataset)))
        logging.info('valid dataset shape: ' + str(len(valid_dataset)))
        
        train_itr = DataLoader(train_dataset, batch_size=parameters.batch_size,
                           shuffle=True, collate_fn=lambda x: collate_fn(x, device),
                           num_workers=parameters.num_workers, pin_memory=args.pin_memory)
        valid_itr = DataLoader(valid_dataset, batch_size=parameters.batch_size,
                           shuffle=True, collate_fn=lambda x: collate_fn(x, device),
                           num_workers=parameters.num_workers, pin_memory=args.pin_memory)

    if args.test_flag:
        logging.info('loading test data...')
        test_dataset = MMDataset(rn, traj_root, mbr, parameters, 'test', device)
        
        print('testing dataset shape: ' + str(len(test_dataset)))
        logging.info('test dataset shape: ' + str(len(test_dataset)))

        test_itr = DataLoader(test_dataset, batch_size=parameters.batch_size,
                            shuffle=False, collate_fn=lambda x: collate_fn(x, device),
                            num_workers=parameters.num_workers, pin_memory=False)

    print('Finish data preparing.')
    logging.info('Finish data preparing.')


    encoder = TrajEncoder(parameters, device).to(device)
    # =========== diffusion ===========
    # denoiser = DenoiseNet(
    #     n_steps=parameters.timesteps,
    #     dim=parameters.id_size - 1,
    #     num_units=parameters.num_units,
    #     condition=True,
    #     cond_dim=2 * parameters.hid_dim
    # ).to(device)
    # diffusion = GaussianDiffusion(
    #     denoiser,
    #     loss_type=parameters.loss_type,
    #     seq_length=parameters.id_size - 1,
    #     timesteps=parameters.timesteps,
    #     sampling_timesteps=parameters.samplingsteps,
    #     objective=parameters.objective,
    #     beta_schedule=parameters.beta_schedule
    # ).to(device)
    # model = ModelAll(encoder, diffusion).to(device)

    # =========== short cut ===========
    dit = DiT(
        parameters.id_size - 1, 
        parameters.num_units, 
        depth=parameters.depth, 
        cond_dim=2 * parameters.hid_dim
    ).to(device)
    shortcut = ShortCut(
        dit,
        infer_steps=parameters.samplingsteps,
        seq_length=parameters.id_size - 1
    ).to(device)
    model = ModelAllShortCut(encoder, shortcut).to(device)
    
    model.apply(init_weights)

    if args.load_model != '':
        model.load_state_dict(torch.load(args.load_model))

    print('model' + str(model))
    logging.info('model' + str(model))

    optimizer = optim.AdamW(model.parameters(), lr=parameters.learning_rate)
    early_stop = 0
    min_loss_val = 1e20
    max_acc_val = 0
    warmup_steps = 5

    for epoch in range(parameters.n_epochs):
        if args.train_flag:
            print(f'epoch: {epoch}')
            logging.info(f'epoch: {epoch}')

            # if epoch < warmup_steps:
            #     for param_group in optimizer.param_groups:
            #         lr = lr_warmup(parameters.learning_rate, warmup_steps, epoch)
            #         param_group["lr"] = lr

            # else:
            #     for param_group in optimizer.param_groups:
            #         lr = parameters.learning_rate - (parameters.learning_rate - 5e-5)*(epoch-warmup_steps)/parameters.n_epochs
            #         param_group["lr"] = lr
        
            lr = parameters.learning_rate

            print(f'training... lr: {lr}')
            logging.info(f'training... lr: {lr}')
            # loss, vb = train(model, train_itr, optimizer, device)
            loss = train_shortcut(model, train_itr, optimizer, parameters.timesteps, device)
            torch.save(model.state_dict(), os.path.join(model_save_path, 'last.pt'))
            print(f'train_loss: {loss}')
            logging.info(f'train_loss: {loss}')

            print('validating...')
            logging.info('validating...')
            # loss_all, loss, vb, acc, recall, precision, f1 = validate(model, valid_itr, device)
            acc, recall, precision, f1 = validate_shortcut(model, valid_itr, device)
            # print(f'valid_loss: {loss}, valid_vb: {vb}')
            print(f'valid_accuracy: {acc}, valid_recall: {recall}, valid_precision: {precision}, valid_f1: {f1}')
            # logging.info(f'valid_loss: {loss}, valid_vb: {vb}')
            logging.info(f'valid_accuracy: {acc}, valid_recall: {recall}, valid_precision: {precision}, valid_f1: {f1}')
            if acc < max_acc_val:
                early_stop += 1
                if early_stop >= 10:
                    print('early stop!')
                    logging.info('early stop!')
                    break
            else:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pt'))
                early_stop = 0

            # min_loss_val = min(loss_all, min_loss_val)
            max_acc_val = max(acc, max_acc_val)
        
        with torch.cuda.device("cuda:{}".format(args.gpu_id)):
            torch.cuda.empty_cache()

    if args.test_flag:
        print('testing...')
        logging.info('testing...')
        # _, loss, vb, acc, recall, precision, f1 = testing(model, test_itr, device)
        acc, recall, precision, f1 = testing_shortcut(model, test_itr, device)
        # print(f'test_loss: {loss}, test_vb: {vb}')
        print(f'test_accuracy: {acc}, test_recall: {recall}, test_precision: {precision}, test_f1: {f1}')
        # logging.info(f'test_loss: {loss}, test_vb: {vb}')
        logging.info(f'test_accuracy: {acc}, test_recall: {recall}, test_precision: {precision}, test_f1: {f1}')


if __name__ == '__main__':
    main()

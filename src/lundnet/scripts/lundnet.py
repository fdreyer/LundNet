# This file is part of LundNet by F. Dreyer and H. Qu

"""
    lundnet.py: the entry point for LundNet.
"""

from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import DataLoader

import tqdm
from functools import partial
import os, time, datetime, argparse, pickle

from lundnet.dgl_dataset import DGLGraphDatasetParticle, DGLGraphDatasetLund, collate_wrapper, collate_wrapper_tree
from sklearn.metrics import roc_curve


def bkg_rejection_at_threshold(signal_eff, background_eff, sig_eff=0.5):
    """Background rejection at a given signal efficiency."""
    return 1 / (1 - background_eff[np.argmin(np.abs(signal_eff - sig_eff)) + 1])


def ROC_area(signal_eff, background_eff):
    """Area under the ROC curve."""
    normal_order = signal_eff.argsort()
    return np.trapz(background_eff[normal_order], signal_eff[normal_order])


def accuracy(preds, labels):
    """Return the accuracy."""
    if labels.ndim == 2:
        labels = labels[:, 1]
    return (preds.argmax(1) == labels).sum().item() / len(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--train-sig', type=str, default='')
    parser.add_argument('--train-bkg', type=str, default='')
    parser.add_argument('--val-sig', type=str, default='')
    parser.add_argument('--val-bkg', type=str, default='')
    parser.add_argument('--test-sig', type=str, default='')
    parser.add_argument('--test-bkg', type=str, default='')
    parser.add_argument('--model', type=str, default='lundnet5', choices=['lundnet5', 'lundnet2',
                                                                          'lundnet3', 'lundnet4',
                                                                          'particlenet', 'particlenet-lite'])
    parser.add_argument('--ln-kt-min', type=float, default=None)
    parser.add_argument('--ln-delta-min', type=float, default=None)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save', type=str, default='')
    parser.add_argument('--name', type=str, default='model')
    parser.add_argument('--test-output', type=str, default='')
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--nev', type=int, default=-1)
    parser.add_argument('--nev-val', type=int, default=-1)
    parser.add_argument('--nev-test', type=int, default=-1)
    parser.add_argument('--start-lr', type=float, default=0.001)
    parser.add_argument('--lr-steps', type=str, default='10,20')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=-1)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if 'lund' in args.model:
        from lundnet.JetTree import JetTree, LundCoordinates
        if args.model == 'lundnet4':
            LundCoordinates.change_dimension(4, ['lnz', 'lnDelta', 'lnKt', 'psi'])
        elif args.model == 'lundnet3':
            LundCoordinates.change_dimension(3, ['lnz', 'lnDelta', 'lnKt'])
        elif args.model == 'lundnet2':
            LundCoordinates.change_dimension(2, ['lnDelta', 'lnKt'])
        kt_min = np.exp(args.ln_kt_min) if (args.ln_kt_min is not None and args.ln_kt_min > -99) else 0
        delta_min = np.exp(args.ln_delta_min) if args.ln_delta_min is not None else 0
        JetTree.change_cuts(kt_min, delta_min)
        print('Using %s, kt_min=%f and delta_min=%f' % (args.model, JetTree.ktmin, JetTree.deltamin))

    if args.demo:
        args.train_sig = 'sample_WW_500GeV.json.gz'
        args.train_bkg = 'sample_QCD_500GeV.json.gz'
        args.val_sig = 'sample_WW_500GeV.json.gz'
        args.val_bkg = 'sample_QCD_500GeV.json.gz'
        args.test_sig = 'sample_WW_500GeV.json.gz'
        args.test_bkg = 'sample_QCD_500GeV.json.gz'

    # training/testing mode
    if args.train_bkg and args.train_sig:
        training_mode = True
    else:
        assert(args.load)
        training_mode = False

    # data format
    DGLGraphDataset = DGLGraphDatasetLund if 'lund' in args.model else DGLGraphDatasetParticle

    # model parameter
    if args.model == 'particlenet':
        from lundnet.ParticleNet import ParticleNet
        Net = ParticleNet
        conv_params = [
            (16, (64, 64, 64)),
            (16, (128, 128, 128)),
            (16, (256, 256, 256)),
        ]
        fc_params = [(256, 0.1)]
        use_fusion = False
        if args.batch_size <= 0:
            args.batch_size = 256
        collate_fn = partial(collate_wrapper, k=conv_params[0][0])
    elif args.model == 'particlenet-lite':
        from lundnet.ParticleNet import ParticleNet
        Net = ParticleNet
        conv_params = [
            (7, (32, 32, 32)),
            (7, (64, 64, 64))
        ]
        fc_params = [(128, 0.1)]
        use_fusion = False
        if args.batch_size <= 0:
            args.batch_size = 1024
        collate_fn = partial(collate_wrapper, k=conv_params[0][0])
    else:
        from lundnet.LundNet import LundNet
        Net = LundNet
        conv_params = [[32, 32], [32, 32], [64, 64], [64, 64], [128, 128], [128, 128]]
        fc_params = [(256, 0.1)]
        use_fusion = True
        if args.batch_size <= 0:
            args.batch_size = 256
        collate_fn = collate_wrapper_tree

    # device
    dev = torch.device(args.device)

    # load data
    if training_mode:
        train_data = DGLGraphDataset(args.train_bkg, args.train_sig, nev=args.nev)
        val_data = DGLGraphDataset(args.val_bkg, args.val_sig, nev=args.nev_val)
        train_loader = DataLoader(train_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                  collate_fn=collate_fn, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                collate_fn=collate_fn, shuffle=False, drop_last=True, pin_memory=True)
        input_dims = train_data.num_features
    else:
        test_data = DGLGraphDataset(args.test_bkg, args.test_sig, nev=args.nev_test)
        test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)
        input_dims = test_data.num_features

    # model
    model = Net(input_dims=input_dims, num_classes=2,
                conv_params=conv_params,
                fc_params=fc_params,
                use_fusion=use_fusion)
    model = model.to(dev)

    def train(model, opt, scheduler, train_loader, dev):
        model.train()

        total_loss = 0
        num_batches = 0
        total_correct = 0
        count = 0
        tic = time.time()
        with tqdm.tqdm(train_loader, ascii=True) as tq:
            for batch in tq:
                label = batch.label
                num_examples = label.shape[0]
                label = label.to(dev).squeeze().long()
                opt.zero_grad()
                logits = model(batch.batch_graph.to(dev), batch.features.to(dev))
                loss = loss_func(logits, label)
                loss.backward()
                opt.step()

                _, preds = logits.max(1)

                num_batches += 1
                count += num_examples
                loss = loss.item()
                correct = (preds == label).sum().item()
                total_loss += loss
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / num_batches),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})
        scheduler.step()

        ts = time.time() - tic
        print('Trained over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(
            count=count, ts=ts, speed=count / ts
        ))

    def evaluate(model, test_loader, dev, return_scores=False, return_time=False):
        model.eval()

        total_correct = 0
        count = 0
        scores = []
        tic = time.time()

        with torch.no_grad():
            with tqdm.tqdm(test_loader, ascii=True) as tq:
                for batch in tq:
                    label = batch.label
                    num_examples = label.shape[0]
                    label = label.to(dev).squeeze().long()
                    logits = model(batch.batch_graph.to(dev), batch.features.to(dev))
                    _, preds = logits.max(1)

                    if return_scores:
                        scores.append(torch.softmax(logits, dim=1).cpu().detach().numpy())

                    correct = (preds == label).sum().item()
                    total_correct += correct
                    count += num_examples

                    tq.set_postfix({
                        'Acc': '%.5f' % (correct / num_examples),
                        'AvgAcc': '%.5f' % (total_correct / count)})

        ts = time.time() - tic
        print('Tested over {count} samples in {ts} secs (avg. speed {speed} samples/s.)'.format(
            count=count, ts=ts, speed=count / ts
        ))
        if return_time:
            return ts

        if return_scores:
            return np.concatenate(scores)
        else:
            return total_correct / count

    if training_mode:
        # loss function
        loss_func = torch.nn.CrossEntropyLoss()

        # optimizer
        opt = torch.optim.Adam(model.parameters(), lr=args.start_lr)

        # learning rate
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_steps, gamma=0.1)

        # training loop
        best_valid_acc = 0
        for epoch in range(args.num_epochs):
            train(model, opt, scheduler, train_loader, dev)

            print('Epoch #%d Validating' % epoch)
            valid_acc = evaluate(model, val_loader, dev)
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if args.save:
                    if args.save and not os.path.exists(args.save):
                        os.makedirs(args.save)
                    torch.save(model.state_dict(), os.path.join(args.save, '%s_state.pt' % args.name))
            print('Current validation acc: %.5f (best: %.5f)' % (valid_acc, best_valid_acc))

    # evaluate model on test dataset
    path = args.save if training_mode else os.path.dirname(args.load)
    name = args.test_output if args.test_output else 'test'
    if path and not os.path.exists(path):
        os.makedirs(path)

    if training_mode:
        del train_data, train_loader, val_data, val_loader
        test_data = DGLGraphDataset(args.test_bkg, args.test_sig, args.nev_test)
        test_loader = DataLoader(test_data, num_workers=args.num_workers, batch_size=args.batch_size,
                                 collate_fn=collate_fn, shuffle=False, drop_last=False, pin_memory=True)

    test_labels = test_data.label.cpu().detach().numpy()
    test_preds = np.zeros((len(test_labels), 2), dtype='float32')

    # load saved model
    model_path = args.save if training_mode else args.load
    if not model_path.endswith('.pt'):
        model_path = os.path.join(model_path, '%s_state.pt' % args.name)
    print('Loading model %s for eval' % model_path)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(args.device)))

    test_preds += evaluate(model, test_loader, dev, return_scores=True)

    info_dict = {'model_name': args.model,
                 'model_params': {'conv_params': conv_params, 'fc_params': fc_params},
                 'lund_ln_kt_min': args.ln_kt_min,
                 'lund_ln_delta_min': args.ln_delta_min,
                 'date': str(datetime.date.today()),
                 'model_path': args.save if training_mode else args.load,
                 'model_name': args.name,
                 'test_sig': args.test_sig,
                 'test_bkg': args.test_bkg}
    if training_mode:
        info_dict.update({'train_sig': args.train_sig,
                          'train_bkg': args.train_bkg})

    fpr, tpr, threshs = roc_curve(test_labels, test_preds[:, 1], pos_label=1)
    # convert into signal and background efficiency
    eff_s = tpr
    eff_b = 1 - fpr
    auc = ROC_area(eff_s, eff_b)

    info_dict['accuracy'] = accuracy(test_preds, test_labels)
    info_dict['auc'] = auc
    info_dict['inv_bkg_at_sig_50'] = bkg_rejection_at_threshold(eff_s, eff_b, 0.5)
    info_dict['inv_bkg_at_sig_30'] = bkg_rejection_at_threshold(eff_s, eff_b, 0.3)

    print(' === Summary ===')
    for k in info_dict:
        print('%s: %s' % (k, info_dict[k]))

    info_file = os.path.join(path, args.name if training_mode else name) + '_INFO.txt'
    with open(info_file, 'w') as f:
        for k in info_dict:
            f.write('%s: %s\n' % (k, info_dict[k]))

    base_name = name.split('.')[0]
    filename = os.path.join(path, base_name)
    with open(filename + '_ROC_data.pickle', 'wb') as f:
        pickle.dump({'signal_eff': eff_s,
                     'background_eff': eff_b,
                     'thresholds': threshs,
                     'description': str(args)}, f)

    print('Saving ROC data for %s' % base_name)

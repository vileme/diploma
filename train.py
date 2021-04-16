import argparse
import json
import random
from pathlib import Path
import pandas as pd
import time
import pickle
import wandb
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.backends import cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from models import UNet16
from loss import LossBinary
from dataset import make_loader
from utils import save_weights, write_event, write_tensorboard, print_model_summay
from validation import validation_binary
from transforms import DualCompose, ImageOnly, Normalize, HorizontalFlip, VerticalFlip
from metrics import AllInOneMeter


def get_split(train_test_split_file='./data/train_test_id.pickle'):
    with open(train_test_split_file, 'rb') as f:
        train_test_id = pickle.load(f)

        train_test_id['total'] = train_test_id[['pigment_network',
                                                'negative_network',
                                                'streaks',
                                                'milia_like_cyst',
                                                'globules']].sum(axis=1)
        valid = train_test_id[train_test_id.Split != 'train'].copy()
        valid['Split'] = 'train'
        train_test_id = pd.concat([train_test_id, valid], axis=0)
    return train_test_id


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--jaccard-weight', type=float, default=1)
    arg('--checkpoint', type=str, default='checkpoint/1_multi_task_unet', help='checkpoint path')
    arg('--train-test-split-file', type=str, default='./data/train_test_id.pickle', help='train test split file path')
    arg('--image-path', type=str, default='data/task2_h5/', help='image path')
    arg('--batch-size', type=int, default=8)
    arg('--n-epochs', type=int, default=100)
    arg('--optimizer', type=str, default='Adam', help='Adam or SGD')
    arg('--lr', type=float, default=0.001)
    arg('--workers', type=int, default=4)
    arg('--model-weight', type=str, default=None)
    arg('--resume-path', type=str, default=None)
    arg('--attribute', type=str, default='all', choices=['pigment_network', 'negative_network',
                                                         'streaks', 'milia_like_cyst',
                                                         'globules', 'all'])
    args = parser.parse_args()

    wandb.init(project="baseline", config=args)

    checkpoint = Path(args.checkpoint)
    checkpoint.mkdir(exist_ok=True, parents=True)

    image_path = args.image_path

    if args.attribute == 'all':
        num_classes = 5
    else:
        num_classes = 1
    args.num_classes = num_classes
    print('--' * 10)
    print(args)
    print('--' * 10)
    checkpoint.joinpath('params.json').write_text(
        json.dumps(vars(args), indent=True, sort_keys=True))

    model = UNet16(num_classes=num_classes, pretrained='vgg')
    wandb.watch(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #    model = nn.DataParallel(model)
    model = nn.DataParallel(model)
    model.to(device)

    if args.model_weight is not None:
        state = torch.load(args.model_weight)
        model.load_state_dict(state['model'])
        print('--' * 10)
        print('Load pretrained model', args.model_weight)
        print('--' * 10)
    print_model_summay(model)

    loss_fn = LossBinary(jaccard_weight=args.jaccard_weight)

    cudnn.benchmark = True

    train_test_id = get_split(args.train_test_split_file)

    print('--' * 10)
    print('num train = {}, num_val = {}'.format((train_test_id['Split'] == 'train').sum(),
                                                (train_test_id['Split'] != 'train').sum()
                                                ))
    print('--' * 10)

    train_transform = DualCompose([
        HorizontalFlip(),
        VerticalFlip(),
        ImageOnly(Normalize())
    ])

    val_transform = DualCompose([
        ImageOnly(Normalize())
    ])

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = make_loader(train_test_id, image_path, args, train=True, shuffle=True,
                               train_test_split_file=args.train_test_split_file,
                               transform=train_transform)
    valid_loader = make_loader(train_test_id, image_path, args, train=False, shuffle=True,
                               train_test_split_file=args.train_test_split_file,
                               transform=val_transform)

    if True:
        print('--' * 10)
        print('check data')
        train_image, train_mask, train_mask_ind = next(iter(train_loader))
        print('train_image.shape', train_image.shape)
        print('train_mask.shape', train_mask.shape)
        print('train_mask_ind.shape', train_mask_ind.shape)
        print('train_image.min', train_image.min().item())
        print('train_image.max', train_image.max().item())
        print('train_mask.min', train_mask.min().item())
        print('train_mask.max', train_mask.max().item())
        print('train_mask_ind.min', train_mask_ind.min().item())
        print('train_mask_ind.max', train_mask_ind.max().item())
        print('--' * 10)

    valid_fn = validation_binary

    if args.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)

    criterion = loss_fn
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)

    previous_valid_loss = 10
    model_path = checkpoint / 'model.pt'
    if args.resume_path is not None and model_path.exists():
        state = torch.load(str(model_path))
        epoch = state['epoch']
        step = state['step']
        model.load_state_dict(state['model'])
        epoch = 1
        step = 0
        try:
            previous_valid_loss = state['valid_loss']
        except:
            previous_valid_loss = 10
        print('--' * 10)
        print('Restored previous model, epoch {}, step {:,}'.format(epoch, step))
        print('--' * 10)
    else:
        epoch = 1
        step = 0

    log = checkpoint.joinpath('train.log').open('at', encoding='utf8')
    writer = SummaryWriter(log_dir=checkpoint)
    meter = AllInOneMeter()
    print('Start training')
    print_model_summay(model)
    previous_valid_jaccard = 0
    for epoch in range(epoch, args.n_epochs + 1):
        model.train()
        random.seed()
        start_time = time.time()
        meter.reset()
        w1 = 1.0
        w2 = 0.5
        w3 = 0.5
        try:
            train_loss = 0
            valid_loss = 0
            for i, (train_image, train_mask, train_mask_ind) in enumerate(train_loader):
                train_image = train_image.permute(0, 3, 1, 2)
                train_mask = train_mask.permute(0, 3, 1, 2)
                train_image = train_image.to(device)
                train_mask = train_mask.to(device).type(
                    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)
                train_mask_ind = train_mask_ind.to(device).type(
                    torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)

                outputs, outputs_mask_ind1, outputs_mask_ind2 = model(train_image)
                train_prob = torch.sigmoid(outputs)
                train_mask_ind_prob1 = torch.sigmoid(outputs_mask_ind1)
                train_mask_ind_prob2 = torch.sigmoid(outputs_mask_ind2)

                loss1 = criterion(outputs, train_mask)
                loss2 = F.binary_cross_entropy_with_logits(outputs_mask_ind1, train_mask_ind)
                loss3 = F.binary_cross_entropy_with_logits(outputs_mask_ind2, train_mask_ind)
                # loss3 = criterion(outputs_mask_ind2, train_mask_ind)
                loss = loss1 * w1 + loss2 * w2 + loss3 * w3
                print(
                    f'epoch={epoch:3d},iter={i:3d}, loss1={loss1.item():.4g}, loss2={loss2.item():.4g}, loss={loss.item():.4g}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                meter.add(train_prob, train_mask, train_mask_ind_prob1, train_mask_ind_prob2, train_mask_ind,
                          loss1.item(), loss2.item(), loss3.item(), loss.item())
            epoch_time = time.time() - start_time
            train_metrics = meter.value()
            train_metrics['epoch_time'] = epoch_time
            train_metrics['image'] = train_image.data
            train_metrics['mask'] = train_mask.data
            train_metrics['prob'] = train_prob.data

            valid_metrics = valid_fn(model, criterion, valid_loader, device, num_classes)
            print(valid_metrics)
            write_event(log, step, epoch=epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
            write_tensorboard(writer, model, epoch, train_metrics=train_metrics, valid_metrics=valid_metrics)
            valid_loss = valid_metrics['loss1']
            valid_jaccard = valid_metrics['jaccard']
            if valid_loss < previous_valid_loss:
                save_weights(model, model_path, epoch + 1, step, train_metrics, valid_metrics)
                previous_valid_loss = valid_loss
                print('Save best model by loss')
            if valid_jaccard > previous_valid_jaccard:
                save_weights(model, model_path, epoch + 1, step, train_metrics, valid_metrics)
                previous_valid_jaccard = valid_jaccard
                print('Save best model by jaccard')
            wandb.log({"loss": valid_metrics["loss"], "loss1": valid_metrics["loss1"],
                       "jaccard_mean": valid_metrics["jaccard"], "jaccard1": valid_metrics["jaccard1"],
                       "jaccard2": valid_metrics["jaccard2"], "jaccard3": valid_metrics["jaccard3"],
                       "jaccard4": valid_metrics["jaccard4"], "jaccard5": valid_metrics["jaccard5"]})
            scheduler.step(valid_metrics['loss1'])

        except KeyboardInterrupt:
            writer.close()
    writer.close()


if __name__ == '__main__':
    main()

import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dataset.CamVid import CamVid
import os
from model.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
from torch.nn import functional as F
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy

def val(args, model, dataloader, csv_path):
    print('start val!')
    label_info = get_label_info(csv_path)
    with torch.no_grad():
        model.eval()
        precision_record = []
        for i, (data, label) in enumerate(dataloader):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # get RGB predict image
            predict = model(data).squeeze()
            predict = reverse_one_hot(predict)
            predict = colour_code_segmentation(np.array(predict.cpu()), label_info)

            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = colour_code_segmentation(np.array(label.cpu()), label_info)
            # compute per pixel accuracy
            precision = compute_global_accuracy(predict, label)
            precision_record.append(precision)
        dice = np.mean(precision_record)
        print('precision per pixel for validation: %.3f' % dice)
        return dice


def train(args, model, optimizer, dataloader_train, dataloader_val, csv_path):
    writer = SummaryWriter()
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i,(data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output, output_sup1, output_sup2 = model(data)
            loss1 = torch.nn.BCEWithLogitsLoss()(output, label)
            loss2 = torch.nn.BCEWithLogitsLoss()(output_sup1, label)
            loss3 = torch.nn.BCEWithLogitsLoss()(output_sup2, label)
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.mkdir(args.save_model_path)
            torch.save(model.module.state_dict(), os.path.join(args.save_model_path, 'epoch_{}.pth'.format(epoch)))
        if epoch % args.validation_step == 0:
            dice = val(args, model, dataloader_val, csv_path)
            writer.add_scalar('precision_val', dice, epoch)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=5, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=640, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='/home/sean/Data/CamVid',help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')

    args = parser.parse_args(params)

    # create dataset and dataloader
    train_path = os.path.join(args.data, 'train')
    train_label_path = os.path.join(args.data, 'train_labels')
    val_path = os.path.join(args.data, 'val')
    val_label_path = os.path.join(args.data, 'val_labels')
    csv_path = os.path.join(args.data, 'class_dict.csv')
    dataset_train = CamVid(train_path, train_label_path, csv_path, scale=(args.crop_height, args.crop_width), mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    dataset_val = CamVid(val_path, val_label_path, csv_path, scale=((args.crop_height, args.crop_width)),  mode='val')
    dataloader_val = DataLoader(
        dataset_val,
        # this has to be 1
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else: #rmsprop
        print('not supported optimizer \n')
        return None
        

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train
    train(args, model, optimizer, dataloader_train, dataloader_val, csv_path)

    # val(args, model, dataloader_val, csv_path)


if __name__ == '__main__':
    params = [
        '--num_epochs', '300',
        '--learning_rate', '0.001',
        '--data', '/home/sean/Data/CamVid',
        '--num_workers', '8',
        '--num_classes', '32',
        '--cuda', '0,1',
        '--batch_size', '6',# 6 for resnet101, 12 for resnet18
        '--save_model_path', './checkpoints_101_adam',
        '--context_path', 'resnet101', # only support resnet18 and resnet101
        '--optimizer', 'adam'
    ]
    main(params)


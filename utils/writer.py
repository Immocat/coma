import os
import time
import torch
import json
from glob import glob


class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Test Loss: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_loss'], info['test_loss'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))

    def load_state_dict(self, model, optimizer, scheduler):
        """
        Set parameters converted from Caffe models authors of VGGFace2 provide.
        See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.

        Arguments:
            model: model
            fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
        """
        fname = self.args.checkpoint
        with open(fname, 'rb') as f:
            weights = torch.load(f, encoding='latin1')

        model_weights = weights['model_state_dict']
        own_state = model.state_dict()
        for name, param in model_weights.items():
            if name in own_state:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
            else:
                raise KeyError('unexpected key "{}" in state_dict'.format(name))


        optimizer_weights = weights['optimizer_state_dict']
        own_state = optimizer.state_dict()
        own_state = optimizer_weights.copy()


        scheduler_weights = weights['scheduler_state_dict']
        own_state = scheduler.state_dict()
        own_state = scheduler_weights.copy()

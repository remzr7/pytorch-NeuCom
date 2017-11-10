import numpy as np
import getopt
import sys
import os
import argparse
import pickle
import visdom

sys.path.insert(0, os.path.join('..', '..'))

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from torch.nn.utils import clip_grad_norm
from neucom.utils import inves, apply_dict
from recurrentController import RecurrentController
from neucom.dnc import DNC

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
    description='PyTorch Differentiable Neural Computer')
parser.add_argument(
    '--input_size', type=int, default=6, help='dimension of input feature')

parser.add_argument(
    '--nhid',
    type=int,
    default=64,
    help='humber of hidden units of the inner nn')

parser.add_argument(
    '--nn_output',
    type=int,
    default=64,
    help='humber of output units of the inner nn')

parser.add_argument('--nlayer', type=int, default=2, help='number of layers')
parser.add_argument(
    '--lr', type=float, default=1e-2, help='initial learning rate')
parser.add_argument(
    '--clip', type=float, default=0.5, help='gradient clipping')

parser.add_argument(
    '--batch_size', type=int, default=2, metavar='N', help='batch size')
parser.add_argument(
    '--mem_size', type=int, default=16, help='memory dimension')
parser.add_argument(
    '--mem_slot', type=int, default=15, help='number of memory slots')
parser.add_argument(
    '--read_heads', type=int, default=1, help='number of read heads')

parser.add_argument(
    '--sequence_max_length',
    type=int,
    default=4,
    metavar='N',
    help='sequence_max_length')
parser.add_argument(
    '--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument(
    '--log-interval',
    type=int,
    default=200,
    metavar='N',
    help='report interval')

parser.add_argument(
    '--iterations',
    type=int,
    default=100000,
    metavar='N',
    help='total number of iteration')
parser.add_argument(
    '--summerize_freq',
    type=int,
    default=100,
    metavar='N',
    help='summerise frequency')
parser.add_argument(
    '--check_freq',
    type=int,
    default=100,
    metavar='N',
    help='check point frequency')

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
if args.cuda:
    print('Using CUDA.')
else:
    print('Using CPU.')

DATASET = 'en'

vis = visdom.Visdom()
loss_avg_log = []
win_loss_avg = "win_loss_avg"
env = "NeuCom"


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def onehot(index, size):
    # print('idx', index, size)
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def load(path):
    return pickle.load(open(path, 'rb'))


def loss_fn(predictions, targets):
    return torch.mean(-1 * F.logsigmoid(predictions) * (targets) - torch.log(
        1 - F.sigmoid(predictions) + 1e-9) * (1 - targets))


def register_nan_checks(model):
    def check_grad(module, grad_input, grad_output):
        # print(module) you can add this to see that the hook is called
        #print(module)
        if any(
                np.all(np.isnan(gi.data.cpu().numpy())) for gi in grad_input
                if gi is not None):
            print('NaN gradient in ' + type(module).__name__)

    model.apply(lambda module: module.register_backward_hook(check_grad))


def generate_data(sample, target_code, word_space_size, cuda=False):
    input_data = np.array(sample[0]['inputs'], dtype=np.float32)
    output_data = np.array(sample[0]['inputs'], dtype=np.float32)
    seq_len = input_data.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_data == target_code)
    output_data[target_mask] = sample[0]['outputs']
    weights_vec[target_mask] = 1.0

    input_data = np.array(
        [onehot(code, word_space_size) for code in input_data])
    output_data = np.array(
        [onehot(code, word_space_size) for code in output_data])
    input_data = np.reshape(input_data, (1, -1, word_space_size))
    output_data = np.reshape(output_data, (1, -1, word_space_size))
    weights_vec = np.reshape(weights_vec, (1, -1, 1))
    input_data = torch.from_numpy(input_data)
    output_data = torch.from_numpy(output_data)
    weights_vec = torch.from_numpy(weights_vec)

    if cuda:
        # cuda
        assert ('no cuda' == 'cuda')

    return Variable(input_data), Variable(output_data), seq_len, Variable(
        weights_vec)


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    ckpts_dir = os.path.join(dirname, 'checkpoints')
    data_dir = os.path.join(dirname, 'data', DATASET)

    llprint("Loading Data ... ")
    lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
    data = load(os.path.join(data_dir, 'train', 'train.pkl'))
    llprint("Done!\n")

    batch_size = 1
    input_size = output_size = len(lexicon_dict)
    mem_slot = args.mem_slot
    mem_size = args.mem_size
    sequence_max_length = args.sequence_max_length
    word_space_size = len(lexicon_dict)
    words_count = 256
    word_size = 64
    read_heads = args.read_heads
    check_freq = args.check_freq
    summerize_freq = args.summerize_freq

    from_checkpoint = None

    learning_rate = args.lr
    momentum = 0.9

    from_checkpoint = None
    iterations = 100000
    start_step = 0

    from_checkpoint = None

    options, _ = getopt.getopt(sys.argv[1:], '',
                               ['checkpoint=', 'iterations='])

    for opt in options:
        if opt[0] == '--checkpoint':
            from_checkpoint = opt[1]
        elif opt[0] == '--iterations':
            iterations = int(opt[1])

    ncomputer = DNC(
        args.nhid,
        args.nn_output,
        args.nlayer,
        RecurrentController,
        input_size,
        output_size,
        mem_slot,
        mem_size,
        read_heads,
        batch_size,
        # use_cuda = False
    )

    register_nan_checks(ncomputer)

    if from_checkpoint is not None:
        ncomputer.load_state_dict(torch.load(from_checkpoint))  # 12)

    last_save_losses = []
    optimizer = optim.RMSprop(ncomputer.parameters(), learning_rate)

    for epoch in range(iterations + 1):
        llprint("\rIteration {ep}/{tot}".format(ep=epoch, tot=iterations))
        optimizer.zero_grad()

        sample = np.random.choice(data, 1)
        input_data, target_output, seq_len, weights = generate_data(
            sample, lexicon_dict['-'], word_space_size)
        # import pdb;pdb.set_trace()

        input_data = input_data.transpose(0, 1).contiguous()
        target_output = target_output.transpose(0, 1).contiguous()

        output, _ = ncomputer(input_data)

        loss = loss_fn(output, target_output)

        loss.backward()

        optimizer.step()
        loss_value = loss.data[0]
        summerize_freq = 10
        summerize = (epoch % summerize_freq == 0)
        take_checkpoint = (epoch != 0) and (epoch % check_freq == 0)
        last_save_losses.append(loss_value)

        if summerize:
            llprint("\n\tAvg. Logistic Loss: %.4f\n" %
                    (np.mean(last_save_losses)))
            loss_avg_log.append([epoch, np.mean(last_save_losses)])
            win_loss_avg = vis.scatter(
                X=np.array(loss_avg_log),
                env=env,
                win=win_loss_avg,
                opts=dict(title="Avg. Logistic Loss"))
            last_save_losses = []

        if take_checkpoint:
            llprint("\nSaving Checkpoint ... "),
            check_ptr = os.path.join(ckpts_dir, 'step_{}.pth'.format(epoch))
            cur_weights = ncomputer.state_dict()
            torch.save(cur_weights, check_ptr)
            llprint("Done!\n")

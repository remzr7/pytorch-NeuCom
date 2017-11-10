from recurrentController import RecurrentController
from neucom.dnc import DNC

import torch
import numpy as np
import pickle
import os
import re
import argparse
import sys

from torch.autograd import Variable

DATA_DIR = './data/en'
CHECKPOINT = 'step_100.pth'

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

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def load(path):
    return pickle.load(open(path, 'rb'))


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[int(index)] = 1.0
    return vec


def prepare_sample(sample, target_code, word_space_size):
    input_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    output_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == target_code)
    output_vec[target_mask] = sample[0]['outputs']
    weights_vec[target_mask] = 1.0

    input_vec = np.array([onehot(code, word_space_size) for code in input_vec])
    output_vec = np.array(
        [onehot(code, word_space_size) for code in output_vec])

    input_vec = np.reshape(input_vec, (1, -1, word_space_size))
    output_vec = np.reshape(output_vec, (1, -1, word_space_size))
    weights_vec = np.reshape(weights_vec, (1, -1, 1))
    input_vec = torch.from_numpy(input_vec)
    output_vec = torch.from_numpy(output_vec)
    weights_vec = torch.from_numpy(weights_vec)

    # if cuda:
        # cuda
        # assert ('no cuda' == 'cuda')

    return Variable(input_vec), Variable(output_vec), seq_len, Variable(
        weights_vec)


ckpts_dir = './checkpoints/'
lexicon_dict = load(DATA_DIR + '/lexicon-dict.pkl')
question_code = lexicon_dict["?"]
target_code = lexicon_dict["-"]
test_files = []
read_heads = args.read_heads
check_freq = args.check_freq

batch_size = 1
input_size = output_size = len(lexicon_dict)
mem_slot = args.mem_slot
mem_size = args.mem_size

for entryname in os.listdir(DATA_DIR + '/test/'):
    entry_path = os.path.join(DATA_DIR + '/test/', entryname)
    if os.path.isfile(entry_path):
        test_files.append(entry_path)
        

ncomputer = ncomputer = DNC(
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

ncomputer.restore(ckpts_dir, CHECKPOINT)

task_results = {}
task_names = {}

for test_file in test_files:
    print('filename', test_file)
    test_data = load(test_file)
    task_regexp = r'qa([0-9]{1,2})_([a-z\-]*)_test.txt.pkl'
    task_filename = os.path.basename(test_file)
    task_match_obj = re.match(task_regexp, task_filename)
    task_number = task_match_obj.group(1)
    task_name = task_match_obj.group(2).replace('-', ' ')
    task_names[task_number] = task_name
    counter = 0
    results = []

    llprint("%s ... %d/%d" % (task_name, counter, len(test_data)))

    for story in test_data:
        astory = np.array(story['inputs'])
        questions_indecies = np.argwhere(astory == question_code)
        questions_indecies = np.reshape(questions_indecies, (-1,))
        target_mask = (astory == target_code)

        desired_answers = np.array(story['outputs'])
        input_vec, _, seq_len, _ = prepare_sample([story], target_code, len(lexicon_dict))

        input_vec = input_vec.transpose(0, 1).contiguous()
        output, _ = ncomputer(input_vec)
        output = np.squeeze(output, axis=0)
        given_answers = np.argmax(output.data.numpy()[target_mask], axis=1)

        answers_cursor = 0
        for question_indx in questions_indecies:
            question_grade = []
            targets_cursor = question_indx + 1
            while targets_cursor < len(astory) and astory[targets_cursor] == target_code:
                question_grade.append(given_answers[answers_cursor] == desired_answers[answers_cursor])
                answers_cursor += 1
                targets_cursor += 1
            results.append(np.prod(question_grade))

        counter += 1
        llprint("\r%s ... %d/%d" % (task_name, counter, len(test_data)))

    error_rate = 1. - np.mean(results)
    task_results[task_number] = error_rate
    llprint("\r%s ... %.3f%% Error Rate.\n" % (task_name, error_rate * 100))

print("\n")
print("%-27s%-27s%s" % ("Task", "Result", "Paper's Mean"))
print("-------------------------------------------------------------------")
paper_means = {
    '1': '9.0±12.6%', '2': '39.2±20.5%', '3': '39.6±16.4%',
    '4': '0.4±0.7%', '5': '1.5±1.0%', '6': '6.9±7.5%', '7': '9.8±7.0%',
    '8': '5.5±5.9%', '9': '7.7±8.3%', '10': '9.6±11.4%', '11':'3.3±5.7%',
    '12': '5.0±6.3%', '13': '3.1±3.6%', '14': '11.0±7.5%', '15': '27.2±20.1%',
    '16': '53.6±1.9%', '17': '32.4±8.0%', '18': '4.2±1.8%', '19': '64.6±37.4%',
    '20': '0.0±0.1%', 'mean': '16.7±7.6%', 'fail': '11.2±5.4'
}
print('tr', task_results)
for k in range(20):
    task_id = str(k + 1)
    task_result = "%.2f%%" % (task_results[task_id] * 100)
    print("%-27s%-27s%s" % (tasks_names[task_id], task_result, paper_means[task_id]))
print("-------------------------------------------------------------------")
all_task_results = [v for _,v in task_results.iteritems()]
results_mean = "%.2f%%" % (np.mean(all_task_results) * 100)
failed_count = "%d" % (np.sum(np.array(all_task_results) > 0.05))

print( "%-27s%-27s%s" % ("Mean Err.", results_mean, paper_means['mean']))
print( "%-27s%-27s%s" % ("Failed (err. > 5%)", failed_count, paper_means['fail']))

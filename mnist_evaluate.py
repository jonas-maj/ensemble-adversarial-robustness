import waitGPU
# waitGPU.wait(utilization=90, available_memory=1000, interval=10)

import examples.problems as pblm
from examples.trainer import *
import setproctitle
import torch.nn as nn
from convex_adversarial import robust_loss, Dense, DenseSequential
import random
import csv
import sys
from collections import Counter

SEED = 0
BATCH_SIZE = 1
VAL_RATIO = 0.2

def get_copy_layer(output_size):
    copy_layer = nn.Linear(output_size, output_size, bias=False)
    copy_layer.weight.data = torch.eye(output_size).cuda()
    return copy_layer

def get_average_layer(output_size, n):
    avg_layer = nn.Linear(output_size, output_size, bias=False)
    avg_layer.weight.data = (1.0 / n) * torch.eye(output_size).cuda()
    return avg_layer

# def averaging_model(models, output_size=10):
#     modules = []
#     modules.append(Dense(nn.Sequential()))
#     for i, model in enumerate(models):
#         modules.append(Dense(*([model] + [None] * i)))
#     modules.append(Dense(*([get_copy_layer(output_size)] * len(models))))
#     modules.append(get_average_layer(output_size, len(models)))

#     avg_model = DenseSequential(*modules)
#     return avg_model

def averaging_model(models, output_size=10):
    modules = []
    modules.append(Dense(nn.Sequential()))
    model_layer_counts = []
    for i, model in enumerate(models):
        model_modules = list(model.modules())[1:]
        model_layer_counts.append(len(model_modules))
        modules.append(Dense(*([nn.Sequential()] + [None] * (len(modules) - 1))))
        for model_module in model_modules:
            modules.append(model_module)
    combine_modules = []
    combine_modules.append(get_copy_layer(output_size))
    for model_layer_count in model_layer_counts[1:]:
        combine_modules += [None] * model_layer_count
        combine_modules.append(get_copy_layer(output_size))

    modules.append(Dense(*combine_modules))
    modules.append(get_average_layer(output_size, len(models)))

    avg_model = DenseSequential(*modules)
    return avg_model

def get_mnist_test_loader():
    train_loader, valid_loader, test_loader = pblm.mnist_loaders(
            batch_size=BATCH_SIZE, path='../data', ratio=VAL_RATIO, seed=SEED)
    return test_loader

def read_models(model_paths):
    models = []
    for i in range(len(model_paths)):
        model = pblm.mnist_model().cuda()
        model.load_state_dict(torch.load(model_paths[i]))
        model.cuda()
        model.eval()
        models.append(model)
    return models

def evaluate_ensemble(model_paths):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    test_loader = get_mnist_test_loader()

    models = read_models(model_paths)
    avg_model = averaging_model(models)
    models.append(avg_model)
    
    rows = list(csv.reader(open('examples/random_indices.csv')))
    idx = set([int(row[0]) for row in rows[1:1001]])

    header = ['eps', 'unanm_norm_err', 'unanm_norm_rej', 'unanm_rob_certs',
            'maj_norm_err', 'maj_norm_rej', 'maj_rob_certs', 'avg_norm_err']  
    header += ['model_' + str(i+1) + '_rob_certs' for i in range(len(models)-1)]
    print(','.join([col for col in header]))

    for k in range(10, 21):
        unanm = {'norm_errs': 0, 'norm_rejs': 0, 'rob_certs': 0}
        maj = {'norm_errs': 0, 'norm_rejs': 0, 'rob_certs': 0}
        avg_norm_errs = 0
        indiv_rob_certs = [0 for _ in range(len(models)-1)]
        eps = k / 100.0
        for i, (X, y) in enumerate(test_loader):
            if i not in idx:
              continue
            X, y = X.cuda(), y.cuda()

            true_class = y.data.item()
            preds = [model(X).data for model in models[:-1]]
            pred_classes = [pred.max(1)[1].item() for pred in preds]
            counter = Counter(pred_classes)
            max_class = -1
            max_count = 0
            for c in counter:
                if counter[c] > max_count:
                    max_count = counter[c]
                    max_class = c

            if max_count == len(preds) and max_class == true_class:
                pass
            elif max_count == len(preds) and max_class != true_class:
                unanm['norm_errs'] += 1
            else:
                unanm['norm_rejs'] += 1

            if max_count >= len(preds) // 2 + 1 and max_class == true_class:
                pass
            elif max_count >= len(preds) // 2 + 1 and max_class != true_class:
                maj['norm_errs'] += 1
            else:
                maj['norm_rejs'] += 1

            preds_avg = models[-1](X).data
            if preds_avg.max(1)[1].item() == true_class:
                pass
            else:
                avg_norm_errs += 1

            robust_errs = []
            for i, model in enumerate(models):
                _, robust_err = robust_loss(model, eps, X, y, bounded_input=False)
                robust_errs.append(int(robust_err))

            if sum(robust_errs) < len(robust_errs):
                unanm['rob_certs'] += 1
            elif sum(robust_errs[:-1]) < len(robust_errs[:-1]) // 2 + 1:
                maj['rob_certs'] += 1

            for i in range(len(indiv_rob_certs)):
               indiv_rob_certs[i] += (1 - robust_errs[i])


        lst = [(k/100.0), unanm['norm_errs'], unanm['norm_rejs'], unanm['rob_certs'], maj['norm_errs'], maj['norm_rejs'], maj['rob_certs'], avg_norm_errs]
        lst += indiv_rob_certs
        print(','.join([str(t) for t in lst]))
                
ensemble_model_paths = {
               'mod2_seed': ['./models/even.pth', './models/odd.pth'],
               'mod2_target': ['./models/even_target.pth', './models/odd_target.pth'],
               'opt2_seed': ['./models/even_opt.pth', './models/odd_opt.pth'],
               'opt2_target': ['./models/even_opt_target.pth', './models/odd_opt_target.pth'],
               'mod5_seed': ['./models/mod5_0.pth', './models/mod5_1.pth', './models/mod5_2.pth', './models/mod5_3.pth', './models/mod5_4.pth'],
               'mod5_target': ['./models/mod5_target_0.pth', './models/mod5_target_1.pth', './models/mod5_target_2.pth', './models/mod5_target_3.pth', './models/mod5_target_4.pth'],
               'opt5_seed': ['./models/opt5_0.pth', './models/opt5_1.pth', './models/opt5_2.pth', './models/opt5_3.pth', './models/opt5_4.pth'],
               'opt5_target': ['./models/opt5_target_0.pth', './models/opt5_target_1.pth', './models/opt5_target_2.pth', './models/opt5_target_3.pth', './models/opt5_target_4.pth'],
               'mod10_seed': ['./models/mod10_0.pth', './models/mod10_1.pth', './models/mod10_2.pth', './models/mod10_3.pth', './models/mod10_4.pth', './models/mod10_5.pth', './models/mod10_6.pth', './models/mod10_7.pth', './models/mod10_8.pth', './models/mod10_9.pth'],
               'mod10_target': ['./models/mod10_target_0.pth', './models/mod10_target_1.pth', './models/mod10_target_2.pth', './models/mod10_target_3.pth', './models/mod10_target_4.pth', './models/mod10_target_5.pth', './models/mod10_target_6.pth', './models/mod10_target_7.pth', './models/mod10_target_8.pth', './models/mod10_target_9.pth'],
              }

def main():
    for ensemble in ensemble_model_paths:
        print(ensemble)
        evaluate_ensemble(ensemble_model_paths[ensemble])

if __name__ == "__main__":
    main()

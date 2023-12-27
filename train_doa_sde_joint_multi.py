#
# A wrapper script that trains the DOAnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import models
import parameters
import time
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from cls_metric import doa_sde_metric

eps = 1e-7


def test_epoch(data_generator, model, sde_loss, criterion, metric_cls, params, device, max_doas):
    nb_train_batches, train_loss = 0, 0.
    model.eval()

    with torch.no_grad():
        for data, target in data_generator.generate():
            # load one batch of data
            target_activity = target[:, :, -params['unique_classes']:].reshape(-1, params['unique_classes'])
            
            if not params['use_two_input']:
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-params['unique_classes']]).to(device).float()
                # process the batch of data based on chosen mode
                activity_binary = None
                
                output_doa, output_sde = model(data)
            else:
                data1, data2, target = torch.tensor(data[0]).to(device).float(), torch.tensor(data[1]).to(device).float(), torch.tensor(target[:, :, :-params['unique_classes']]).to(device).float()
                # process the batch of data based on chosen mode
                activity_binary = None
                output_doa, output_sde = model(data1, data2)
                    
            # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
            max_nb_doas = output_doa.shape[2]//3
            output_doa = output_doa.view(output_doa.shape[0], output_doa.shape[1], 3, max_nb_doas).transpose(-1, -2)
            output_sde = output_sde.view(output_sde.shape[0], output_sde.shape[1], 1, max_nb_doas).transpose(-1, -2)
            
            target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)

            # Compute unit-vectors of predicted DoA
            # (batch, sequence, 3, max_nb_doas) to (batch*sequence, 3, max_nb_doas)
            output_doa, output_sde, target = output_doa.view(-1, output_doa.shape[-2], output_doa.shape[-1]), output_sde.view(-1, output_sde.shape[-2], output_sde.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1])

            target_norm = torch.sqrt(torch.sum(target**2, -1) + 1e-10)
            target_norm = target_norm.unsqueeze(-1)

            #output = output/output_norm.unsqueeze(-1)

            loss_doa = criterion(output_doa, target)
            loss_sde = sde_loss(output_sde, target_norm)
        
            loss = params['branch_weights'][0] * loss_doa + params['branch_weights'][1] * loss_sde

            dist_diff = torch.abs(torch.sub(output_sde, target_norm))
            thres = torch.mul(target_norm, 0.1)
            is_greater_thres = torch.gt(thres, dist_diff)
            
            dist_diff = dist_diff.cpu().detach().numpy()
            is_greater_thres = is_greater_thres.cpu().detach().numpy()
            
            # compute the angular distance matrix to estimate the localization error
            dist_mat_hung = torch.matmul(output_doa.detach(), target.transpose(-1, -2))
            dist_mat_hung = torch.clamp(dist_mat_hung, -1+eps, 1-eps) # the +- eps is critical because the acos computation will become saturated if we have values of -1 and 1
            dist_mat_hung = torch.acos(dist_mat_hung)  # (batch, sequence, max_nb_doas, max_nb_doas)
            dist_mat_hung = dist_mat_hung.cpu().detach().numpy()
            
            metric_cls.partial_compute_metric(dist_mat_hung, dist_diff.squeeze(), is_greater_thres.squeeze(), target_activity, pred_activity=activity_binary, max_doas=max_doas)

            train_loss += loss.item()
            nb_train_batches += 1
            if params['quick_test'] and nb_train_batches == 4:
                break

        train_loss /= nb_train_batches

    return metric_cls, train_loss


def train_epoch(data_generator, optimizer, model, sde_loss, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for data, target in data_generator.generate():
        if not params['one_per_file']:
            target_activity = target[:, :, -params['unique_classes']:].reshape(-1, params['unique_classes'])
            nb_framewise_doas_gt = target_activity.sum(-1)
        else:
            target_activity = target[:, -params['unique_classes']:]
            nb_framewise_doas_gt = target_activity.sum(-1)
        # load one batch of data
        if not params['use_two_input']:
            if not params['one_per_file']:
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :, :-params['unique_classes']]).to(device).float()
            else:
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target[:, :-params['unique_classes']]).to(device).float()
    
            # process the batch of data based on chosen mode
            optimizer.zero_grad()
            output_doa, output_sde = model(data)
            
        else:
            data1, data2, target = torch.tensor(data[0]).to(device).float(), torch.tensor(data[1]).to(device).float(), torch.tensor(target[:, :, :-params['unique_classes']]).to(device).float()
    
            # process the batch of data based on chosen mode
            optimizer.zero_grad()

            output_doa, output_sde = model(data1, data2)

        # (batch, sequence, max_nb_doas*3) to (batch, sequence, 3, max_nb_doas)
        max_nb_doas = output_doa.shape[2]//3
        output_doa = output_doa.view(output_doa.shape[0], output_doa.shape[1], 3, max_nb_doas).transpose(-1, -2)
        output_sde = output_sde.view(output_sde.shape[0], output_sde.shape[1], 1, max_nb_doas).transpose(-1, -2)
        target = target.view(target.shape[0], target.shape[1], 3, max_nb_doas).transpose(-1, -2)


        # Compute unit-vectors of predicted DoA
        # (batch, sequence, 3, max_nb_doas) to (batch*sequence, 3, max_nb_doas)
        output_doa, output_sde, target = output_doa.view(-1, output_doa.shape[-2], output_doa.shape[-1]), output_sde.view(-1, output_sde.shape[-2], output_sde.shape[-1]), target.view(-1, target.shape[-2], target.shape[-1])
        target_norm = torch.sqrt(torch.sum(target**2, -1) + 1e-10)
        target_norm = target_norm.unsqueeze(-1)

        loss_doa = criterion(output_doa, target)
        loss_sde = sde_loss(output_sde, target_norm)

        loss = params['branch_weights'][0] * loss_doa + params['branch_weights'][1] * loss_sde

        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches

    return train_loss


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        test_splits = params['test_splits']
        val_splits = params['val_splits']
        train_splits = params['train_splits']

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_{}_split{}'.format(
            task_id, job_id, params['dataset'], params['mode'], split
        )
        unique_name = os.path.join(params['model_dir'], unique_name)
        model_name = '{}_model.h5'.format(unique_name)
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False
        )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train.get_data_sizes()
        if not params['use_two_input']:
            model = models.CRNN_Multi(data_in, data_out, params).to(device)
        else:
            if params['rot_cnn']:
                model = models.CRNN_Two_Multi(data_in, data_out, params).to(device)
            else:
                model = models.CRNN_Rot(data_in, data_out, params).to(device)

        print('---------------- DOA-net -------------------')
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size']))
        print(model)

        # start training
        best_val_epoch = -1
        best_doa, best_dist, best_acc, best_joint = 180, 100, 0, 100
        patience_cnt = 0

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        tr_loss_list = np.zeros(nb_epoch)
        val_loss_list = np.zeros(nb_epoch)

        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        criterion = torch.nn.MSELoss()
        sde_loss = torch.nn.MSELoss()

        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss = train_epoch(data_gen_train, optimizer, model, sde_loss, criterion, params, device)
            train_time = time.time() - start_time
            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------
            start_time = time.time()
            val_metric = doa_sde_metric() #if params['one_per_file'] else doa_metric()
            val_metric, val_loss = test_epoch(data_gen_val, model, sde_loss, criterion, val_metric, params, device, params['unique_classes'])
            val_hung_loss, dist_err, dist_acc = val_metric.get_results()
            val_time = time.time() - start_time
            joint_score = np.mean([val_hung_loss/180., 1.-dist_acc])
            # Save model if loss is good
            if joint_score < best_joint:
                patience_cnt = 0
                best_val_epoch, best_doa, best_dist, best_acc, best_joint = epoch_cnt, val_hung_loss, dist_err, dist_acc, joint_score
                torch.save(model.state_dict(), model_name)

            # Print stats and plot scores
            print(
                'epoch: {}, time: {:0.2f}/{:0.2f}, '
                'train_loss: {:0.2f}, val_loss: {:0.2f}, '
                'DOA error/Distance error/Distance acc: {:0.3f}/{}, '
                'best_val_epoch: {} {}'.format(
                    epoch_cnt, train_time, val_time,
                    train_loss,
                    val_loss,
                    val_hung_loss, '{:0.2f}/{:0.2f}'.format(dist_err, dist_acc),
                    best_val_epoch, '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_doa, best_dist, best_acc, best_joint))
            )

            tr_loss_list[epoch_cnt], val_loss_list[epoch_cnt] = train_loss, val_loss

            patience_cnt += 1
            if patience_cnt > params['patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False
        )

        test_metric = doa_sde_metric()
        test_metric, test_loss = test_epoch(data_gen_test, model, sde_loss, criterion, test_metric, params, device, params['unique_classes'])

        test_hung_loss, test_dist_err, test_dist_acc = test_metric.get_results()

        print(
            'test_loss: {:0.2f}, DOA error/Distance rrror/Distance acc: {:0.3f}/{}'.format(
                test_loss,
                test_hung_loss, '{:0.2f}/{:0.2f}'.format(test_dist_err, test_dist_acc))
        )



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)


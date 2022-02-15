from __future__ import print_function
import pickle
import datetime
import glob
import os
import argparse
from scipy import io
import math
import time
from sklearn.preprocessing import MinMaxScaler as Scaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import haste_pytorch as haste
import sys
from os.path import dirname
sys.path.append(dirname(__file__))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)*0.1

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        '''Zero out the gradients by the inner optimizer'''
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''
        self.n_current_steps += 1
        #print('n_current_steps:', self.n_current_steps)
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, _checkpoint):
        self._optimizer.load_state_dict(_checkpoint)

    def load_n_current_step(self, _n_current_step):
        self.n_current_steps = _n_current_step

    def state_n_current_step(self):
        return self.n_current_steps

def weights_init(m):
    """pytorch의 network를 초기화 하기위한 함수f

    :param Object m: 초기화를 하기 위한 network instance

    :return: None
    """
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(0.0, batchnorm_var)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.normal_(0.0, batchnorm_var)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
    elif isinstance(m, nn.ConvTranspose2d):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0]  # number of rows
        fan_in = size[1]  # number of columns
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)

    elif isinstance(m, nn.LSTMCell):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
                
def init_hidden(bsz, numlayers, hidden_size):

    h_t_enc = []
    c_t_enc = []

    for ii in range(numlayers):
        h_t_enc.append(torch.zeros(1, bsz, hidden_size,
                       requires_grad=False).cuda())
        c_t_enc.append(torch.zeros(1, bsz, hidden_size,
                       requires_grad=False).cuda())

    return h_t_enc, c_t_enc

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class Sequence2(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_size, net_dim, numlayers, numlayers_nn, zoneout_prob, dropout_prob):
        super(Sequence2, self).__init__()

        enccell_list = []
        enccell_list.append(haste.LayerNormLSTM(input_size=input_dim, hidden_size=hidden_size,
                            zoneout=zoneout_prob, dropout=dropout_prob, return_state_sequence=True))  # forward

        for idcell in range(1, int(numlayers)):
            enccell_list.append(haste.LayerNormLSTM(input_size=hidden_size, hidden_size=hidden_size,
                                zoneout=zoneout_prob, dropout=dropout_prob, return_state_sequence=True))

        fclayer_list = []
        fclayer_list.append(nn.Linear((hidden_size), output_dim, bias=True))

        self.enccell_list = nn.ModuleList(enccell_list)
        self.fclayer_list = nn.ModuleList(fclayer_list)

    def forward(self, inputs, h_t_enc, c_t_enc):
        
        y,   (h_t_enc[0], c_t_enc[0]) = self.enccell_list[0](inputs, (h_t_enc[0], c_t_enc[0]))
        
        for ii in range(1, len(self.enccell_list)):
            y,   (h_t_enc[ii], c_t_enc[ii]) = self.enccell_list[ii](h_t_enc[ii-1][0,:,:,:], (h_t_enc[ii], c_t_enc[ii]))
        
        nn_lstm_out = self.h_t_enc_layernorm(h_t_enc[-1][0,-1,:,:])
                
        for iiii in range(0, len(self.fclayer_list)-1):
            nn_lstm_out = self.fclayer_list[iiii](nn_lstm_out)
            nn_lstm_out = F.relu(self.fclayer_norm_list[iiii](nn_lstm_out))
            
        nn_lstm_out = self.fclayer_list[-1](nn_lstm_out)

        for ii in range(len(self.enccell_list)):
            h_t_enc[ii] = h_t_enc[ii][:,-1,:,:]
            c_t_enc[ii] = c_t_enc[ii][:,-1,:,:]
            
        return nn_lstm_out, h_t_enc, c_t_enc


class lnlstm_module:

    def __init__(self, label, seq_len=168, bsz=4, hidden_size=128, train_size=0.7, numlayers=4, numlayers_nn=2, zoneout_prob=0.1, dropout_prob=0.05, net_dim=128, num_epochs=1000):

        self.label = label
        self.num_epochs = num_epochs
        self.bsz = bsz
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.numlayers = numlayers
        self.numlayers_nn = numlayers_nn
        self.zoneout_prob = zoneout_prob
        self.dropout_prob = dropout_prob
        self.batchnorm_var = 2e-2
        self.net_dim = net_dim
        self.train_size = train_size

    def load_partial_data(self):
        excel_list = [
            'data_1st_period.xlsx',
            'data_2nd_period.xlsx',
        ]
        path = '/data/Research/dataset/'

        train_in_list=[]
        train_out_list=[]
        ScalerData_in = pd.DataFrame()
        ScalerData_out = pd.DataFrame()
        self.input_columns=['날짜', '광량','외부온도','내부온도', '내부습도']

        self.output_columns = ['내부온도', '내부습도']

        for file in excel_list:
            tmptrain_in = pd.read_excel(path+file).filter(items=self.input_columns)
            tmptrain_out = pd.read_excel(path+file).filter(items=self.output_columns)

            tmptrain_in['날짜'] = tmptrain_in['날짜'].apply(pd.to_datetime)
            tmptrain_in['time_index'] = ""
            tmptrain_in['time_index'] = (
            ((tmptrain_in['날짜']-tmptrain_in['날짜'].min())/np.timedelta64(1, 'h')) % 24.0)/24.0  # 24 h --> 0 ~ 1
            for column in tmptrain_in:
                tmptrain_in[column] = tmptrain_in[column].interpolate()

            for column in tmptrain_out:
                tmptrain_out[column] = tmptrain_out[column].interpolate()
            train_in = tmptrain_in.iloc[:, 1:]
            train_out = tmptrain_out.iloc[:, :]

            ScalerData_in  = ScalerData_in .append(train_in )
            ScalerData_out = ScalerData_out.append(train_out)

            train_in_list.append(train_in)
            train_out_list.append(train_out)

        self.input_dim = train_in_list[0].shape[1]
        self.output_dim = train_out_list[0].shape[1]

        self.Scaler_in = Scaler()
        self.Scaler_in.fit(ScalerData_in)
        self.Scaler_out = Scaler()
        self.Scaler_out.fit(ScalerData_out)

        for i in range(len(excel_list)):
            train_in_list[i] = self.Scaler_in.transform(train_in_list[i])
            train_out_list[i] = self.Scaler_out.transform(train_out_list[i])

        return train_in_list, train_out_list
    def load_test_data(self):
        tmptest_data = pd.read_excel('/data/Research/dataset/test_data.xlsx')
        
        tmptest_in = tmptest_data.filter(items=self.input_columns)
        tmptest_out = tmptest_data.filter(items=self.output_columns)

        tmptest_in['날짜'] = tmptest_in['날짜'].apply(pd.to_datetime)
        tmptest_in['time_index'] = ""
        tmptest_in['time_index'] = (
            ((tmptest_in['날짜']-tmptest_in['날짜'].min())/np.timedelta64(1, 'h')) % 24.0)/24.0  # 24 h --> 0 ~ 1

        for column in tmptest_in:
            tmptest_in[column] = tmptest_in[column].interpolate()

        for column in tmptest_out:
            tmptest_out[column] = tmptest_out[column].interpolate()

        test_in = tmptest_in.iloc[:, 1:]
        test_out = tmptest_out.iloc[:, :]

    def train(self, input_seq, output_seq, epoch):
        self.seq.train()

        inx = np.zeros(0)
        file_inx = np.zeros(0)

        for i in range(len(input_seq)):
          
            tmp = np.arange(0, input_seq[i].shape[0]-(self.seq_len))
            inx = np.concatenate((inx,tmp), axis=0)
            file_inx = np.concatenate((file_inx, np.ones(tmp.shape[0])*i), axis=0)

        acc_loss = np.zeros(len(file_inx)//self.bsz)
        s = np.arange(file_inx.shape[0])
        np.random.shuffle(s)
        inx = inx[s]
        file_inx = file_inx[s]
        num_iters = (len(file_inx)//self.bsz)
  
        for i in range(num_iters):

            h_t_enc, c_t_enc = init_hidden(self.bsz, self.numlayers, self.hidden_size)
            
            self.optimizer.zero_grad()

            inputs = np.zeros((self.seq_len, self.bsz, self.input_dim))
            targets = np.zeros((self.seq_len, self.bsz, self.output_dim))
            acc_outputs = torch.zeros(self.seq_len, self.bsz, self.output_dim).cuda()

            
            for ii in range(self.bsz):
                inputs[:, ii, :]  = input_seq[int(file_inx[i*self.bsz+ii])][int(inx[i*self.bsz+ii]):int(inx[i*self.bsz+ii])+self.seq_len,:]
                targets[:, ii, :] = output_seq[int(file_inx[i*self.bsz+ii])][int(inx[i*self.bsz+ii])+1:int(inx[i*self.bsz+ii])+self.seq_len+1,:]
            
            inputs = torch.FloatTensor((inputs)).cuda()
            targets = torch.FloatTensor((targets)).cuda()

            for ii in range(self.seq_len):
                tmp = inputs[ii, :, :].unsqueeze(0).clone()
                acc_outputs[ii, :, :], h_t_enc, c_t_enc = self.seq(tmp, h_t_enc, c_t_enc)

            loss = self.train_criterion(acc_outputs, targets)

            acc_loss[i] = loss.data.cpu().numpy()

            print('{} of {} train_loss {:.2e}'.format(i+1, num_iters, acc_loss[i]))

            if torch.isfinite(loss):
                loss.backward()
                grads = [x.grad for x in self.seq.parameters()]
                if not torch.isnan(torch.sum(grads[0])):
                    torch.nn.utils.clip_grad_norm_(
                        self.seq.parameters(), 0.1)  #
                    self.optimizer.step_and_update_lr()
            else:
                print('loss is infinite')
                exit(1)

        return acc_loss

    def load_model(self, epoch):
        self.seq = Sequence2(self.input_dim, self.output_dim, self.hidden_size, self.net_dim, self.numlayers, self.numlayers_nn, self.zoneout_prob, self.dropout_prob)
        self.seq.cuda()

        self.optimizer = ScheduledOptim(torch.optim.Adam(self.seq.parameters(), betas=(0.9, 0.98), eps=1e-09), self.hidden_size, 1)
        self.train_criterion = nn.MSELoss()
        self.eval_criterion = nn.MSELoss(reduction='none')

        self.seq = torch.nn.DataParallel(self.seq, device_ids=[0])

        if epoch > 0:
            tmp = glob.glob("data_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}/model_epoch_{}_*.pth".format(
                dataset, bsz, seq_len, hidden_size, numlayers, net_dim, numlayers_nn, zoneout_prob, dropout_prob, epoch))
            if len(tmp) == 1:
                check_point = torch.load(tmp[0])
                self.seq.load_state_dict(check_point['state_dict'])
                if hasattr(self, 'optimizer'):
                    self.optimizer.load_n_current_step(
                        check_point['n_current_steps'])
                    self.optimizer.load_state_dict(check_point['optimizer'])
            elif len(tmp) > 1:
                print("there is more than two pth files for {} epoch".format(epoch))
                sys.exit(1)
            elif len(tmp) == 0:
                print("there is no pth file for {} epoch".format(epoch))
                sys.exit(1)

    def evaluate(self, input_seq, output_seq):

        self.seq.eval()

        acc_loss = np.zeros((len(input_seq)-self.seq_len, self.output_dim))
        acc_output = np.zeros((len(input_seq)-self.seq_len, self.output_dim))
        acc_target = np.zeros((len(input_seq)-self.seq_len, self.output_dim))

        num_iters = len(acc_loss)//self.bsz

        idx = np.arange(len(input_seq)-self.seq_len)
        
        for i in range(num_iters-2):

            h_t_enc, c_t_enc = init_hidden(self.bsz, self.numlayers, self.hidden_size)

            inputs = np.zeros((self.seq_len, self.bsz, self.input_dim))
            targets = np.zeros((self.bsz, self.output_dim))
            outputs = np.zeros((self.bsz, self.output_dim))

            for ii in range(self.bsz):
                inputs[:, ii, :] = input_seq[idx[i*self.bsz+ii]:idx[i*self.bsz+ii]+self.seq_len, :]
                targets[ii, :] = output_seq[idx[i*self.bsz+ii]+self.seq_len+1, :]

            with torch.no_grad():

                inputs = torch.FloatTensor((inputs)).cuda()
                targets = torch.FloatTensor((targets)).cuda()
                outputs = torch.FloatTensor((outputs)).cuda()

                for ii in range(self.seq_len):
                    outputs, h_t_enc, c_t_enc = self.seq(
                        inputs[ii, :, :].unsqueeze(0), h_t_enc, c_t_enc)

                loss = self.eval_criterion(outputs, targets)

            acc_loss[i*self.bsz:(i+1)*self.bsz, :] = loss.data.cpu().numpy()

            acc_output[i*self.bsz:(i+1)*self.bsz,:] = outputs.data.cpu().numpy()

            acc_target[i*self.bsz:(i+1)*self.bsz,:] = targets.data.cpu().numpy()

            print('{} of {} eval_loss {:.2e}'.format(i+1, num_iters, loss.data.cpu().numpy().mean()))

        if len(acc_loss) % self.bsz != 0:
            i = i+1
            bsz = (len(acc_loss) % self.bsz) 

            h_t_enc, c_t_enc = init_hidden(bsz, self.numlayers, self.hidden_size)

            inputs = np.zeros((self.seq_len, bsz, self.input_dim))
            targets = np.zeros((bsz, self.output_dim))
            outputs = np.zeros((bsz, self.output_dim))

            for ii in range(bsz):
                inputs[:, ii, :] = input_seq[idx[i*self.bsz+ii]:idx[i*self.bsz+ii]+self.seq_len, :]
                targets[ii, :] = output_seq[idx[i*self.bsz+ii]+self.seq_len+1, :]

            with torch.no_grad():

                inputs = torch.FloatTensor((inputs)).cuda()
                targets = torch.FloatTensor((targets)).cuda()
                outputs = torch.FloatTensor((outputs)).cuda()

                for ii in range(self.seq_len):
                    outputs, h_t_enc, c_t_enc = self.seq(inputs[ii, :, :].unsqueeze(0), h_t_enc, c_t_enc)

                loss = self.eval_criterion(outputs, targets)

            acc_loss[i*bsz:(i+1)*bsz, :] = loss.data.cpu().numpy()

            acc_output[i*bsz:(i+1)*bsz, :] = outputs.data.cpu().numpy()

            acc_target[i*bsz:(i+1)*bsz, :] = targets.data.cpu().numpy()

            print('{} of {} eval_loss {:.2e}'.format(i+1, num_iters+1, loss.data.cpu().numpy().mean()))

        return acc_output, acc_target, acc_loss

    def checkpoint0(self, epoch, train_predict, train_target, train_loss):
        count_t = time.time()
        model_out_dir = "label_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}".format(
            self.label, self.bsz, self.seq_len, self.hidden_size, self.numlayers, self.net_dim, self.numlayers_nn, self.zoneout_prob, self.dropout_prob)

        if not os.path.isdir(model_out_dir):
            os.mkdir(model_out_dir)

        model_out_path = "label_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}/model_epoch_{}_{}.pth".format(
            self.label, self.bsz, self.seq_len, self.hidden_size, self.numlayers, self.net_dim, self.numlayers_nn, self.zoneout_prob, self.dropout_prob, epoch, count_t)
        torch.save({
            'epoch': epoch,
            'state_dict': self.seq.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'n_current_steps': self.optimizer.state_n_current_step()
        }, model_out_path)

        result_out_path = "label_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}/train_epoch_{}_{}.mat".format(
            self.label, self.bsz, self.seq_len, self.hidden_size, self.numlayers, self.net_dim, self.numlayers_nn, self.zoneout_prob, self.dropout_prob, epoch, count_t)
        data = {'train_loss': train_loss,
                'train_predict': self.Scaler_out.inverse_transform(train_predict),
                'train_target': self.Scaler_out.inverse_transform(train_target)
                }
        io.savemat(result_out_path, data)

        print("Checkpoint saved to {} \n".format(model_out_path))

    def checkpoint1(self, epoch, eval_predict, eval_target, eval_loss):

        count_t = time.time()
        model_out_dir = "label_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}".format(
            self.label, self.bsz, self.seq_len, self.hidden_size, self.numlayers, self.net_dim, self.numlayers_nn, self.zoneout_prob, self.dropout_prob)

        if not os.path.isdir(model_out_dir):
            os.mkdir(model_out_dir)

        result_out_path = "label_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}/eval_epoch_{}_{}.mat".format(
            self.label, self.bsz, self.seq_len, self.hidden_size, self.numlayers, self.net_dim, self.numlayers_nn, self.zoneout_prob, self.dropout_prob, epoch, count_t)
        data = {'eval_loss': eval_loss,
                'eval_predict': self.Scaler_out.inverse_transform(eval_predict),
                'eval_target': self.Scaler_out.inverse_transform(eval_target)
                }
        io.savemat(result_out_path, data)

        print("Checkpoint saved to {} \n".format(result_out_path))

    def checkpoint2(self, epoch, test_predict, test_target, test_loss):

        count_t = time.time()
        model_out_dir = "label_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}".format(
            self.label, self.bsz, self.seq_len, self.hidden_size, self.numlayers, self.net_dim, self.numlayers_nn, self.zoneout_prob, self.dropout_prob)

        if not os.path.isdir(model_out_dir):
            os.mkdir(model_out_dir)

        result_out_path = "data_{}_batch_{}_seqlen_{}_hidden_{}_numlayers_{}_netdim_{}_numlayersnn_{}_dropout_{}_{}/test_epoch_{}_{}.mat".format(
            self.label, self.bsz, self.seq_len, self.hidden_size, self.numlayers, self.net_dim, self.numlayers_nn, self.zoneout_prob, self.dropout_prob, epoch, count_t)
        data = {'test_loss': test_loss,
                'test_predict': self.Scaler_out.inverse_transform(test_predict),
                'test_target': self.Scaler_out.inverse_transform(test_target)}
        io.savemat(result_out_path, data)

        print("Checkpoint saved to {} \n".format(result_out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='resume training')
    parser.add_argument('--epoch', type=int, default=0,
                        help="the value of epoch means which traning epoch you want to employ to resume (default:0)")
    args = parser.parse_args()

    num_epochs = 100
    seq_len = 144
    bsz = 64
    train_size = 1.0
    hidden_size = 32
    net_dim = 1
    numlayers = 24
    numlayers_nn = 0 # obsolete
    zoneout_prob = 0.1
    dropout_prob = 0.1

    label = 'label_1'

    lnlstm_module = lnlstm_module(label, seq_len, bsz, hidden_size, train_size, numlayers, numlayers_nn, zoneout_prob, dropout_prob, net_dim, num_epochs)

    train_in, train_out = lnlstm_module.load_partial_data()
    test_in, test_out = lnlstm_module.load_test_data()

    lnlstm_module.load_model(args.epoch)

    for epoch in range(args.epoch+1, num_epochs + 1):

        print('start train epoch {} of {}'.format(epoch, num_epochs))

        train_loss = lnlstm_module.train(train_in, train_out, epoch)

        if epoch % 50 == 0:

            for i in range(len(train_in)):
                train_predict, train_target, train_loss = lnlstm_module.evaluate(train_in[i], train_out[i])
                lnlstm_module.checkpoint0(epoch, train_predict, train_target, train_loss)

        #eval_predict, eval_target, eval_loss = lnlstm_module.evaluate(eval_in, eval_out)
        #lnlstm_module.checkpoint1(epoch, eval_predict, eval_target, eval_loss)

            test_predict, test_target, test_loss = lnlstm_module.evaluate(test_in, test_out)
            lnlstm_module.checkpoint2(epoch, test_predict, test_target, test_loss)

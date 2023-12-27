#
# The SELDnet architecture
#

import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, mask=None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.div(torch.matmul(Q, K.permute(0, 1, 3, 2)), np.sqrt(self.head_dim))

        #energy = [batch size, n heads, query len, key len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x


class AttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, key_channels):
        super(AttentionLayer, self).__init__()
        self.conv_Q = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_K = nn.Conv1d(in_channels, key_channels, kernel_size=1, bias=False)
        self.conv_V = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        Q = self.conv_Q(x)
        K = self.conv_K(x)
        V = self.conv_V(x)
        A = Q.permute(0, 2, 1).matmul(K).softmax(2)
        x = A.matmul(V.permute(0, 2, 1)).permute(0, 2, 1)
        return x

    def __repr__(self):
        return self._get_name() + \
            '(in_channels={}, out_channels={}, key_channels={})'.format(
            self.conv_Q.in_channels,
            self.conv_V.out_channels,
            self.conv_K.out_channels
            )


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x

class ConvBlockLittle(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(1, 1), padding=(1, 1)):

        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding)

        self.bn = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.relu_(self.bn(self.conv(x)))
        return x

        
class CRNN(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.conv_block_list = torch.nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'][conv_cnt-1] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt'][conv_cnt],
                        kernel_size=params['kernel_size']
                        )
                        )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                    )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                    )

        if params['nb_rnn_layers']:
            self.in_gru_size = int(params['nb_cnn2d_filt'][-1] * int(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
#            self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        # Branch for activity detection 
        self.fnn_act_list = torch.nn.ModuleList()
        if self.use_hnet and self.use_activity_out:
            for fc_cnt in range(params['nb_fnn_act_layers']):
                self.fnn_act_list.append(
                    torch.nn.Linear(params['fnn_act_size'] if fc_cnt else params['rnn_size'] , params['fnn_act_size'], bias=True)
                )
            self.fnn_act_list.append(
                torch.nn.Linear(params['fnn_act_size'] if params['nb_fnn_act_layers'] else params['rnn_size'], params['unique_classes'], bias=True)
            )


    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            # out - batch x hidden x seq
            x = torch.tanh(x)

        for fnn_cnt in range(len(self.fnn_list)-1):
            x = torch.relu_(self.fnn_list[fnn_cnt](x))
        doa = torch.tanh(self.fnn_list[-1](x))
        '''(batch_size, time_steps, label_dim)'''

        return doa
        
class CRNN_Multi(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.conv_block_list = torch.nn.ModuleList()
        self.use_binary = params['use_binary_label']
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = int(params['nb_cnn2d_filt'] * int(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
#            self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        # Branch for SDE 
        self.fnn_sde_list = torch.nn.ModuleList()
        for fc_cnt in range(params['nb_fnn_sde_layers']):
            self.fnn_sde_list.append(
                torch.nn.Linear(params['fnn_sde_size'] if fc_cnt else params['rnn_size'] , params['fnn_sde_size'], bias=True)
                )
        self.fnn_sde_list.append(
            torch.nn.Linear(params['fnn_sde_size'] if params['nb_fnn_sde_layers'] else params['rnn_size'], params['unique_classes'], bias=True)
        )


    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            x = self.attn.forward(x, x, x)
            # out - batch x hidden x seq
            x = torch.tanh(x)

        x_rnn = x
        for fnn_cnt in range(len(self.fnn_list)-1):
            x = torch.relu_(self.fnn_list[fnn_cnt](x))
        doa = torch.tanh(self.fnn_list[-1](x))
        '''(batch_size, time_steps, label_dim)'''

        for fnn_cnt in range(len(self.fnn_sde_list)-1):
            x_rnn = torch.relu_(self.fnn_sde_list[fnn_cnt](x_rnn))
        if self.use_binary:
            sde = torch.sigmoid(self.fnn_sde_list[-1](x_rnn))
        else:
            sde = torch.tanh(self.fnn_sde_list[-1](x_rnn))

        return doa, sde
        
class CRNN_Two(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.conv_block_list = torch.nn.ModuleList()
        self.conv_rot_list = torch.nn.ModuleList()
        self.merge_mode = params['merge_mode']
        self._use_rot = params['use_rot']
        self._use_trans = params['use_trans']
        self._use_rot_trans = params['use_rot_trans']
        if params['use_rot']:
            self._rot_feat_nb = 4
        elif params['use_trans']:
            self._rot_feat_nb = 3
        elif params['use_rot_trans']:
            self._rot_feat_nb = 7
            
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                       ConvBlock(
                           in_channels=params['nb_cnn2d_filt'][conv_cnt-1] if conv_cnt else in_feat_shape[0][1],
                           out_channels=params['nb_cnn2d_filt'][conv_cnt],
                           kernel_size=params['kernel_size']
                           )
                       )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                    )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                    )
                
        if params['use_two_input']:
            for cnv_id in range(2):
                self.conv_rot_list.append(
                    ConvBlock(
                        in_channels=params['rot_filt'] if cnv_id else in_feat_shape[1][1],
                        out_channels=params['rot_filt']
                    )
                )

        if params['nb_rnn_layers']:
            if self.merge_mode == 'concat':
                self.in_gru_size = int(params['nb_cnn2d_filt'][-1] * (in_feat_shape[0][-1] / np.prod(params['f_pool_size']))+self._rot_feat_nb*params['rot_filt'])
            else:
                self.in_gru_size = int(params['nb_cnn2d_filt'][-1] * (in_feat_shape[0][-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
#            self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        # Branch for activity detection 
        self.fnn_act_list = torch.nn.ModuleList()
        if self.use_hnet and self.use_activity_out:
            for fc_cnt in range(params['nb_fnn_act_layers']):
                self.fnn_act_list.append(
                    torch.nn.Linear(params['fnn_act_size'] if fc_cnt else params['rnn_size'] , params['fnn_act_size'], bias=True)
                )
            self.fnn_act_list.append(
                torch.nn.Linear(params['fnn_act_size'] if params['nb_fnn_act_layers'] else params['rnn_size'], params['unique_classes'], bias=True)
            )


    def forward(self, x, y):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        for conv_cnt in range(len(self.conv_rot_list)):
            y = self.conv_rot_list[conv_cnt](y)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        y = y.transpose(1, 2).contiguous()
        y = y.view(y.shape[0], y.shape[1], -1).contiguous()

        if self.merge_mode == 'concat':
            z = torch.cat((x,y),2)
        elif self.merge_mode == 'additive':
            z = torch.add(x,y)
        elif self.merge_mode == 'multiplicative':
            z = torch.mul(x,y)
        
        (z, _) = self.gru(z)
        z = torch.tanh(z)
        z = z[:, :, z.shape[-1]//2:] * z[:, :, :z.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            z = self.attn.forward(z, z, z)
            # out - batch x hidden x seq
            z = torch.tanh(z)

        for fnn_cnt in range(len(self.fnn_list)-1):
            z = torch.relu_(self.fnn_list[fnn_cnt](z))
        doa = self.fnn_list[-1](z)
        '''(batch_size, time_steps, label_dim)'''

        return doa

class CRNN_Two_Multi(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.conv_block_list = torch.nn.ModuleList()
        self.conv_rot_list = torch.nn.ModuleList()
        self.merge_mode = params['merge_mode']
        self._use_rot = params['use_rot']
        self._use_trans = params['use_trans']
        self._use_rot_trans = params['use_rot_trans']
        self.use_binary = params['use_binary_label']
        if params['use_rot']:
            self._rot_feat_nb = 4
        elif params['use_trans']:
            self._rot_feat_nb = 3
        elif params['use_rot_trans']:
            self._rot_feat_nb = 7
            
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'][conv_cnt-1] if conv_cnt else in_feat_shape[0][1],
                        out_channels=params['nb_cnn2d_filt'][conv_cnt]
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )
        if params['use_two_input']:
            for cnv_id in range(2):
                self.conv_rot_list.append(
                    ConvBlock(
                        in_channels=params['rot_filt'] if cnv_id else in_feat_shape[1][1],
                        out_channels=params['rot_filt']
                    )
                )

        if params['nb_rnn_layers']:
            if self.merge_mode == 'concat':
                self.in_gru_size = int(params['nb_cnn2d_filt'][-1] * (in_feat_shape[0][-1] / np.prod(params['f_pool_size']))+self._rot_feat_nb*params['rot_filt'])
            else:
                self.in_gru_size = int(params['nb_cnn2d_filt'][-1] * (in_feat_shape[0][-1] / np.prod(params['f_pool_size'])))
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
#            self.attn = AttentionLayer(params['rnn_size'], params['rnn_size'], params['rnn_size'])
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        # Branch for SDE 
        self.fnn_sde_list = torch.nn.ModuleList()
        for fc_cnt in range(params['nb_fnn_sde_layers']):
            self.fnn_sde_list.append(
                torch.nn.Linear(params['fnn_sde_size'] if fc_cnt else params['rnn_size'] , params['fnn_sde_size'], bias=True)
                )
        self.fnn_sde_list.append(
            torch.nn.Linear(params['fnn_sde_size'] if params['nb_fnn_sde_layers'] else params['rnn_size'], params['unique_classes'], bias=True)
        )


    def forward(self, x, y):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''
        
        for conv_cnt in range(len(self.conv_rot_list)):
            y = self.conv_rot_list[conv_cnt](y)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        y = y.transpose(1, 2).contiguous()
        y = y.view(y.shape[0], y.shape[1], -1).contiguous()
        if self.merge_mode == 'concat':
            z = torch.cat((x,y),2)
        elif self.merge_mode == 'additive':
            z = torch.add(x,y)
        elif self.merge_mode == 'multiplicative':
            z = torch.mul(x,y)
        
        (z, _) = self.gru(z)
        z = torch.tanh(z)
        z = z[:, :, z.shape[-1]//2:] * z[:, :, :z.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            z = self.attn.forward(z, z, z)
            # out - batch x hidden x seq
            z = torch.tanh(z)

        z_rnn = z
        for fnn_cnt in range(len(self.fnn_list)-1):
            z = torch.relu_(self.fnn_list[fnn_cnt](z))
        doa = self.fnn_list[-1](z)
        '''(batch_size, time_steps, label_dim)'''


        for fnn_cnt in range(len(self.fnn_sde_list)-1):
            z_rnn = torch.relu_(self.fnn_sde_list[fnn_cnt](z_rnn))
            
        if self.use_binary:
            sde = torch.sigmoid(self.fnn_sde_list[-1](z_rnn))
        else:
            sde = torch.relu_(self.fnn_sde_list[-1](z_rnn))

        return doa, sde
        
        
class CRNN_Rot(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params):
        super().__init__()
        self.conv_block_list = torch.nn.ModuleList()
        if len(params['f_pool_size']):
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[0][1],
                        out_channels=params['nb_cnn2d_filt']
                    )
                )
                self.conv_block_list.append(
                    torch.nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt]))
                )
                self.conv_block_list.append(
                    torch.nn.Dropout2d(p=params['dropout_rate'])
                )

        if params['nb_rnn_layers']:
            self.in_gru_size = int(params['nb_cnn2d_filt'] * (in_feat_shape[0][-1] / np.prod(params['f_pool_size']))+4)
            self.gru = torch.nn.GRU(input_size=self.in_gru_size, hidden_size=params['rnn_size'],
                                    num_layers=params['nb_rnn_layers'], batch_first=True,
                                    dropout=params['dropout_rate'], bidirectional=True)
        self.attn = None
        if params['self_attn']:
            self.attn = MultiHeadAttentionLayer(params['rnn_size'], params['nb_heads'], params['dropout_rate'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_rnn_layers'] and params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(
                    torch.nn.Linear(params['fnn_size'] if fc_cnt else params['rnn_size'] , params['fnn_size'], bias=True)
                )
        self.fnn_list.append(
            torch.nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else params['rnn_size'], out_shape[-1], bias=True)
        )

        # Branch for activity detection 
        self.fnn_act_list = torch.nn.ModuleList()
        if self.use_hnet and self.use_activity_out:
            for fc_cnt in range(params['nb_fnn_act_layers']):
                self.fnn_act_list.append(
                    torch.nn.Linear(params['fnn_act_size'] if fc_cnt else params['rnn_size'] , params['fnn_act_size'], bias=True)
                )
            self.fnn_act_list.append(
                torch.nn.Linear(params['fnn_act_size'] if params['nb_fnn_act_layers'] else params['rnn_size'], params['unique_classes'], bias=True)
            )


    def forward(self, x, y):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        ''' (batch_size, time_steps, feature_maps):'''

        z = torch.cat((x,y),2)
        
        (z, _) = self.gru(z)
        z = torch.tanh(z)
        z = z[:, :, z.shape[-1]//2:] * z[:, :, :z.shape[-1]//2]
        '''(batch_size, time_steps, feature_maps)'''
        if self.attn is not None:
            z = self.attn.forward(z, z, z)
            # out - batch x hidden x seq
            z = torch.tanh(z)

        for fnn_cnt in range(len(self.fnn_list)-1):
            z = torch.relu_(self.fnn_list[fnn_cnt](z))
        doa = torch.tanh(self.fnn_list[-1](z))
        '''(batch_size, time_steps, label_dim)'''

        return doa
    
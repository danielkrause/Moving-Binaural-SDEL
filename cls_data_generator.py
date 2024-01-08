#
# Data generator for training the SELDnet
#

import os
import numpy as np
import cls_feature_class
from collections import deque
import random


class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        self._per_file = params['per_file']
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._batch_size = params['batch_size']
        self._feature_seq_len = params['feature_sequence_length']
        self._label_seq_len = params['label_sequence_length']
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length
        self._nb_classes = self._feat_cls.get_nb_classes()
        self._approach = params['approach']
        if self._approach == 'DOA_class' or self._approach == 'Joint_class':
            self._fnn_dnn = True
            self._nb_mel_bins = self._feat_cls.get_nb_mel_bins() + 1
        else:
            self._fnn_dnn = False
            self._nb_mel_bins = self._feat_cls.get_nb_mel_bins()
        self._get_filenames_list_and_feat_label_sizes()
        self._one_per_file = params['one_per_file']

        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len
        self._label_batch_seq_len = self._batch_size*self._label_seq_len
        self._circ_buf_feat = None
        self._circ_buf_label = None
        
        self._two_input = params['use_two_input']
        self._use_cnn = params['rot_cnn']
        self._use_rot = params['use_rot']
        self._use_trans = params['use_trans']
        self._use_rot_trans = params['use_rot_trans']


        if self._per_file:
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor((len(self._filenames_list) * self._nb_frames_file /
                                               float(self._feature_batch_seq_len))))

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        if not self._two_input:
            if not self._fnn_dnn:
                feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
            else:
                feat_shape = (self._batch_size, self._feature_seq_len, self._nb_mel_bins)
        else:
            if self._use_rot:
                nb_rot_feat = 4
                minus_ch = 1
            elif self._use_trans:
                nb_rot_feat = 3
                minus_ch = 1
            else:
                nb_rot_feat = 7
                minus_ch = 2
                    
            feat_shape = [(self._batch_size, self._nb_ch-minus_ch, self._feature_seq_len, self._nb_mel_bins),(self._batch_size, 1, self._feature_seq_len, nb_rot_feat)]

        if self._is_eval:
            label_shape = None
        else:
            if not self._one_per_file:
                label_shape = [self._batch_size, self._label_seq_len, self._nb_classes*3]
            else:
                label_shape = [self._batch_size, self._nb_classes*3]
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        for filename in os.listdir(self._label_dir):
            fil_nb = int(filename.split('.')[0][8:])
            split_nb = None
            rnd_files = np.load('rnd_files.npy')
            rnd_first = rnd_files[:500]
            rnd_second = rnd_files[500:1000]
            rnd_third = rnd_files[1000:1500]
            rnd_fourth = rnd_files[1500:2000]
            rnd_fifth = rnd_files[2000:2500]
            if fil_nb in rnd_first:
                split_nb = 1
            elif fil_nb in rnd_second:
                split_nb = 2
            elif fil_nb in rnd_third:
                split_nb = 3
            elif fil_nb in rnd_fourth:
                split_nb = 4
            elif fil_nb in rnd_fifth:
                split_nb = 5
            feat_filename = '{}{}.npy'.format('binaural',filename.split('.')[0][8:])
            if split_nb in self._splits: # check which split the file belongs to
                self._filenames_list.append(feat_filename)

        temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[0]))
        self._nb_frames_file = temp_feat.shape[0]
        self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins
        
        lab_filename = '{}{}.npy'.format('metadata',self._filenames_list[0].split('.')[0][8:])
                
        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, lab_filename))
            self._label_len = temp_label.shape[-1]
            self._doa_len = (self._label_len - self._nb_classes)//self._nb_classes

        if self._per_file:
            self._batch_size = int(np.ceil(temp_feat.shape[0]/float(self._feature_seq_len)))

        return

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """
        if self._shuffle:
            random.shuffle(self._filenames_list)

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                # Split to sequences
                
                if not self._two_input:
                    feat = self._split_in_seqs(feat, self._feature_seq_len)
                    feat = np.transpose(feat, (0, 2, 1, 3))
                    
                    if self._fnn_dnn:
                        feat = np.squeeze(feat)
                    yield feat
                else:
                    if self._use_cnn:
                        feat = self._split_in_seqs(feat, self._feature_seq_len)
                        feat = np.transpose(feat, (0, 2, 1, 3))
                        feat_spec = feat[:, :4, :, :]
                        feat_rot = feat[:, 4, :, :4]
    
                        yield [feat_spec, np.expand_dims(feat_rot,1)]
                    else:
                        feat = self._split_in_seqs(feat, self._feature_seq_len)
                        feat_spec = feat[:, :, :4, :]
                        feat_rot = feat[:, :, 4, :4]
                        feat_spec = np.transpose(feat_spec, (0, 2, 1, 3))
    
                        yield [feat_spec, feat_rot]

        else:
            for i in range(self._nb_total_batches):

                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while len(self._circ_buf_feat) < self._feature_batch_seq_len:
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))
                    lab_filename = '{}{}.npy'.format('metadata',self._filenames_list[file_cnt].split('.')[0][8:])
                    temp_label = np.load(os.path.join(self._label_dir, lab_filename))

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)

                    file_cnt = file_cnt + 1
                
                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                label = np.zeros((self._label_batch_seq_len, self._label_len))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                for j in range(self._label_batch_seq_len):
                    a=self._circ_buf_label.popleft()
                    label[j, :] = a
                    
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))
                # Split to sequences
                # if not self._use_trans and not self._use_rot_trans:
                #     label_act = label[:, 4]
                #     label_dist = label[:, 3]
                #     label = label[:, :4]
                #     label[:, 0] = label[:, 0]*label_dist
                #     label[:, 1] = label[:, 1]*label_dist
                #     label[:, 2] = label[:, 2]*label_dist
                    
                #     label[:, 3] = label_act

                    
                label = self._split_in_seqs(label, self._label_seq_len)
                
                if not self._two_input:
                    feat = self._split_in_seqs(feat, self._feature_seq_len)
                    feat = np.transpose(feat, (0, 2, 1, 3))
                    if self._one_per_file:
                        label = label[:, 5, :]
                    if self._fnn_dnn:
                        feat = np.squeeze(feat)
                    yield feat, label
                else:
                    if self._use_cnn:
                        if self._use_trans:
                            feat = self._split_in_seqs(feat, self._feature_seq_len)
                            feat = np.transpose(feat, (0, 2, 1, 3))
                            feat_spec = feat[:, :4, :, :]
                            feat_rot = feat[:, 4, :, :3]
                        elif self._use_rot_trans:
                            feat = self._split_in_seqs(feat, self._feature_seq_len)
                            feat = np.transpose(feat, (0, 2, 1, 3))
                            feat_spec = feat[:, :4, :, :]
                            feat_rot = feat[:, 4, :, :4]
                            feat_trans = feat[:, 5, :, :3] 
                            feat_rot = np.concatenate((feat_rot, feat_trans), axis=-1)
                        else:
                            feat = self._split_in_seqs(feat, self._feature_seq_len)
                            feat = np.transpose(feat, (0, 2, 1, 3))
                            feat_spec = feat[:, :4, :, :]
                            feat_rot = feat[:, 4, :, :4]
    
                        yield [feat_spec, np.expand_dims(feat_rot,1)], label
                    else:
                        feat = self._split_in_seqs(feat, self._feature_seq_len)
                        feat_spec = feat[:, :, :4, :]
                        feat_rot = feat[:, :, 4, :4]
                        feat_spec = np.transpose(feat_spec, (0, 2, 1, 3))
    
                        yield [feat_spec, feat_rot], label
                        
                

    def _split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict)

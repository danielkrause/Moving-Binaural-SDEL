# Contains routines for labels creation, features extraction and normalization
#


import os
import numpy as np
import scipy.io.wavfile as wav
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plot
import librosa
plot.switch_backend('agg')
import math


def nCr(n, r):
    return math.factorial(n) // math.factorial(r) // math.factorial(n-r)


class FeatureClass:
    def __init__(self, params, is_eval=False):
        """

        :param params: parameters dictionary
        :param is_eval: if True, does not load dataset labels.
        """

        # Input directories
        self._feat_label_dir = params['feat_label_dir']
        self._dataset_dir = params['dataset_dir']
        if is_eval:
            self._dataset_combination = '{}_{}'.format(params['dataset'], 'eval')
        else:
            self._dataset_combination = '{}_{}'.format(params['dataset'], 'dev')
            
        self._aud_dir = os.path.join(self._dataset_dir, self._dataset_combination)

        self._desc_dir = None if is_eval else os.path.join(self._dataset_dir, 'metadata_dev')

        # Output directories
        self._label_dir = None
        self._feat_dir = None
        self._feat_dir_norm = None

        # Local parameters
        self._is_eval = is_eval

        self._fs = params['fs']
        self._hop_len_s = params['hop_len_s']
        self._hop_len = int(self._fs * self._hop_len_s)

        self._label_hop_len_s = params['label_hop_len_s']
        self._label_hop_len = int(self._fs * self._label_hop_len_s)
        self._label_frame_res = self._fs / float(self._label_hop_len)
        self._nb_label_frames_1s = int(self._label_frame_res)

        self._win_len = 2 * self._hop_len
        self._nfft = self._next_greater_power_of_2(self._win_len)
        self._nb_mel_bins = params['nb_mel_bins']
        self._mel_wts = librosa.filters.mel(sr=self._fs, n_fft=self._nfft, n_mels=self._nb_mel_bins).T

        self._dataset = params['dataset']
        self._eps = 1e-8
        self._nb_channels = 2

        self._use_rot = params['use_rot']
        self._use_trans = params['use_trans']
        self._use_rot_trans = params['use_rot_trans']
        self._approach = params['approach']
        self._feat = params['feat']

        # Sound event classes dictionary
        self._nb_unique_classes = params['unique_classes']
        self._audio_max_len_samples = params['max_audio_len_s'] * self._fs

        self._max_feat_frames = int(np.ceil(self._audio_max_len_samples / float(self._hop_len))) - 1
        self._max_label_frames = 999 #TODO: change this hardcoded value

    def _load_audio(self, audio_path):
        fs, audio = wav.read(audio_path)
        audio = audio[:, :self._nb_channels] / 32768.0 + self._eps
        if audio.shape[0] < self._audio_max_len_samples:
            zero_pad = np.random.rand(self._audio_max_len_samples - audio.shape[0], audio.shape[1])*self._eps
            audio = np.vstack((audio, zero_pad))
        elif audio.shape[0] > self._audio_max_len_samples:
            audio = audio[:self._audio_max_len_samples, :]
        return audio, fs

    # INPUT FEATURES
    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def _spectrogram(self, audio_input):
        _nb_ch = audio_input.shape[1]
        nb_bins = self._nfft // 2
        spectra = np.zeros((self._max_feat_frames, nb_bins + 1, _nb_ch), dtype=complex)
        for ch_cnt in range(_nb_ch):
            stft_ch = librosa.core.stft(np.asfortranarray(audio_input[:, ch_cnt]), n_fft=self._nfft, hop_length=self._hop_len,
                                        win_length=self._win_len, window='hann')
            spectra[:, :, ch_cnt] = stft_ch[:, :self._max_feat_frames].T + self._eps
        return spectra

    def _get_mel_spectrogram(self, linear_spectra):
        mel_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, linear_spectra.shape[-1]))
        for ch_cnt in range(linear_spectra.shape[-1]):
            mag_spectra = np.abs(linear_spectra[:, :, ch_cnt])**2
            mel_spectra = np.dot(mag_spectra, self._mel_wts)
            log_mel_spectra = librosa.power_to_db(mel_spectra)
            mel_feat[:, :, ch_cnt] = log_mel_spectra
        mel_feat = mel_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))
        return mel_feat

    def _get_foa_intensity_vectors(self, linear_spectra):
        W = linear_spectra[:, :, 0]
        I = np.real(np.conj(W)[:, :, np.newaxis] * linear_spectra[:, :, 1:])
        E = self._eps + (np.abs(W)**2 + ((np.abs(linear_spectra[:, :, 1:])**2).sum(-1))/3.0 )
        
        I_norm = I/E[:, :, np.newaxis]
        I_norm_mel = np.transpose(np.dot(np.transpose(I_norm, (0,2,1)), self._mel_wts), (0,2,1))
        foa_iv = I_norm_mel.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], self._nb_mel_bins * 3))
        if np.isnan(foa_iv).any():
            print('Feature extraction is generating nan outputs')
            exit()
        return foa_iv

    def _get_gcc(self, linear_spectra):
        gcc_channels = nCr(linear_spectra.shape[-1], 2)
        gcc_feat = np.zeros((linear_spectra.shape[0], self._nb_mel_bins, gcc_channels))
        cnt = 0
        for m in range(linear_spectra.shape[-1]):
            for n in range(m+1, linear_spectra.shape[-1]):
                R = np.conj(linear_spectra[:, :, m]) * linear_spectra[:, :, n]
                cc = np.fft.irfft(np.exp(1.j*np.angle(R)))
                cc = np.concatenate((cc[:, -self._nb_mel_bins//2:], cc[:, :self._nb_mel_bins//2]), axis=-1)
                gcc_feat[:, :, cnt] = cc
                cnt += 1
        return gcc_feat.transpose((0, 2, 1)).reshape((linear_spectra.shape[0], -1))

    def _get_spectrogram_for_file(self, audio_filename):
        audio_in, fs = self._load_audio(os.path.join(self._aud_dir, audio_filename))
        audio_spec = self._spectrogram(audio_in)
        return audio_spec

    def _get_rotation_features(self, lab_filename):
        lab_path = os.path.join(self._desc_dir, lab_filename)
        _fid = open(lab_path, 'r')
        
        rot_features = np.zeros((self._max_label_frames, self._nb_mel_bins))
        
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            
            rot_features[_frame_ind-1, :] = np.tile(np.array([float(_words[4]),float(_words[5]),float(_words[6]),float(_words[7])]),64)
        
        return rot_features

    def _get_translation_features(self, lab_filename):
        lab_path = os.path.join(self._desc_dir, lab_filename)
        _fid = open(lab_path, 'r')
        
        trans1_features = np.zeros((self._max_label_frames, self._nb_mel_bins))
        
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            translation = np.tile(np.array([float(_words[1]),float(_words[2])]),128)
            #translation = np.append(translation,translation[0])
            trans1_features[_frame_ind-1, :] = translation
        
        trans_features = np.zeros_like(trans1_features)
        nb_feat = np.shape(trans_features)[1]
        for feat in range(nb_feat):
            trans_features[:,feat] = trans1_features[:, feat] - trans1_features[0, feat]
            
        return trans_features
    
    def _get_mean_binaural_magnitude_spectrum(self, linear_spectra):
        return np.sqrt((np.abs(linear_spectra[:, :, 0])**2 + np.abs(linear_spectra[:, :, 1])**2)/2.)

    def _get_ILD(self, linear_spectra):
        return np.abs(np.divide(np.abs(linear_spectra[:, :, 0]), np.abs(linear_spectra[:, :, 1]+self._eps)))
    
    def _get_IPDx(self, linear_spectra):
        return np.cos(np.angle(linear_spectra[:, :, 0]) - np.angle(linear_spectra[:, :, 1]))

    def _get_IPDy(self, linear_spectra):
        return np.sin(np.angle(linear_spectra[:, :, 0]) - np.angle(linear_spectra[:, :, 1]))

    # OUTPUT LABELS
    def get_labels_for_file(self, _desc_file):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        x_label = np.zeros((self._max_label_frames, self._nb_unique_classes))
        y_label = np.zeros((self._max_label_frames, self._nb_unique_classes))
        z_label = np.ones((self._max_label_frames, self._nb_unique_classes))

        nb_classes = np.zeros((self._max_label_frames, self._nb_unique_classes))
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self._max_label_frames:
                for active_event in range(len(active_event_list)):
                    nb_classes[frame_ind, active_event] = 1
                    x_label[frame_ind-1, active_event] = active_event_list[active_event][0]
                    y_label[frame_ind-1, active_event] = active_event_list[active_event][1]
                    z_label[frame_ind-1, active_event] = active_event_list[active_event][2]
        label_mat = np.concatenate((x_label, y_label, z_label, nb_classes), axis=1)
        return label_mat

    def get_binary_label(self, _desc_file):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        # If not using Hungarian net use a deafult DOA, which is a unit vector. We are choosing (x, y, z) = (0, 0, 1)
        x_label = np.zeros((self._max_label_frames, self._nb_unique_classes))
        y_label = np.zeros((self._max_label_frames, self._nb_unique_classes))
        z_label = np.ones((self._max_label_frames, self._nb_unique_classes))
        dists = np.zeros((self._max_label_frames, self._nb_unique_classes))

        nb_classes = np.zeros((self._max_label_frames, self._nb_unique_classes))
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self._max_label_frames:
                for active_event in range(len(active_event_list)):
                    nb_classes[frame_ind, active_event] = 1
                    x_label[frame_ind-1, active_event] = active_event_list[active_event][0]
                    y_label[frame_ind-1, active_event] = active_event_list[active_event][1]
                    z_label[frame_ind-1, active_event] = active_event_list[active_event][2]
                    dist = np.sqrt(x_label[frame_ind-1, active_event]**2 + y_label[frame_ind-1, active_event]**2 + z_label[frame_ind-1, active_event]**2)
                    if dist < 5.:
                        dists[frame_ind-1, active_event] = 1
        label_mat = np.concatenate((x_label, y_label, z_label, dists, nb_classes), axis=1)
        return label_mat

    def get_class_approach(self, _desc_file):
        """
        Reads description file and returns classification based SED labels and regression based DOA labels

        :param _desc_file: metadata description file
        :return: label_mat: of dimension [nb_frames, 3*max_classes], max_classes each for x, y, z axis,
        """

        # If using Hungarian net set default DOA value to a fixed value greater than 1 for all axis. We are choosing a fixed value of 10
        x_label = np.zeros((self._max_label_frames, self._nb_unique_classes))
        y_label = np.zeros((self._max_label_frames, self._nb_unique_classes))
        z_label = np.ones((self._max_label_frames, self._nb_unique_classes))
        dists = np.zeros((self._max_label_frames, 6))

        nb_classes = np.zeros((self._max_label_frames, self._nb_unique_classes))
        for frame_ind, active_event_list in _desc_file.items():
            if frame_ind < self._max_label_frames:
                for active_event in range(len(active_event_list)):
                    nb_classes[frame_ind, active_event] = 1
                    x_label[frame_ind-1, active_event] = active_event_list[active_event][0]
                    y_label[frame_ind-1, active_event] = active_event_list[active_event][1]
                    z_label[frame_ind-1, active_event] = active_event_list[active_event][2]
                    dist = np.sqrt(x_label[frame_ind-1, active_event]**2 + y_label[frame_ind-1, active_event]**2 + z_label[frame_ind-1, active_event]**2)
                    if dist < 2.:
                        dists[frame_ind-1, 0] = 1
                    elif dist < 3.:
                        dists[frame_ind-1, 1] = 1
                    elif dist < 4.:
                        dists[frame_ind-1, 2] = 1
                    elif dist < 5.:
                        dists[frame_ind-1, 3] = 1
                    elif dist < 7.:
                        dists[frame_ind-1, 4] = 1
                    else:
                        dists[frame_ind-1, 5] = 1
                        
        label_mat = np.concatenate((x_label, y_label, z_label, dists, nb_classes), axis=1)
        return label_mat
    
    # ------------------------------- EXTRACT FEATURE AND PREPROCESS IT -------------------------------
    def extract_all_feature(self):
        # setting up folders
        self._feat_dir = self.get_unnormalized_feat_dir()
        create_folder(self._feat_dir)

        # extraction starts
        print('Extracting spectrogram:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tfeat_dir {}'.format(
            self._aud_dir, self._desc_dir, self._feat_dir))
#        for file_cnt, file_name in enumerate(os.listdir(self._aud_dir)):
#            print(file_name)
        for file_cnt, file_name in enumerate(os.listdir(self._aud_dir)):
            if not file_name == '.DS_Store':
                wav_filename = '{}.wav'.format(file_name.split('.')[0])
                lab_filename = '{}{}.csv'.format('metadata',file_name.split('.')[0][8:])
                spect = self._get_spectrogram_for_file(wav_filename)
                
                spect1 = spect[:, 1:, :]

                feat = None

                # extract mean binaural magnitud spectrum (MBMS)
                mbms = self._get_mean_binaural_magnitude_spectrum(spect1)

                # extract inter-level channel difference
                ild = self._get_ILD(spect1)

                # extract inter-channel phase differences (IPD)
                ipdx = self._get_IPDx(spect1)
                ipdy = self._get_IPDy(spect1)
                gcc = self._get_gcc(spect1)
                
                # extract gcc
                #gcc = self._get_gcc(spect)

                # features concatenation
                if self._use_rot:
                    rot_inf = self._get_rotation_features(lab_filename)
                    feat = np.concatenate((mbms, ild, ipdx, ipdy, rot_inf), axis=-1)
                elif self._use_trans:
                    trans_inf = self._get_translation_features(lab_filename)
                    feat = np.concatenate((mbms, ild, ipdx, ipdy, trans_inf), axis=-1)
                elif self._use_rot_trans:
                    rot_inf = self._get_rotation_features(lab_filename)
                    trans_inf = self._get_translation_features(lab_filename)
                    feat = np.concatenate((mbms, ild, ipdx, ipdy, rot_inf, trans_inf), axis=-1)
                else:
                    if self._feat == 'spec':
                        feat = mbms
                    elif self._feat == 'logmel':
                        feat = self._get_mel_spectrogram(spect)
                    elif self._feat == 'two_specs':
                        feat = np.concatenate((np.abs(spect1[:,:,0]),np.abs(spect1[:,:,1])), axis=-1)
                    elif self._feat == 'spec_phase':
                        feat = np.concatenate((np.abs(spect1[:,:,0]),np.abs(spect1[:,:,1]),np.angle(spect1[:,:,0]), np.angle(spect1[:,:,1])), axis=-1)
                    elif self._feat == 'spec_ild':
                        feat = np.concatenate((mbms, ild), axis=-1)
                    elif self._feat == 'spec_sin_cos':
                        feat = np.concatenate((mbms, ipdx, ipdy), axis=-1)
                    elif self._feat == 'spec_gcc':
                        feat = np.concatenate((mbms, gcc), axis=-1)
                    elif self._feat == 'spec_ild_gcc':
                        feat = np.concatenate((mbms, ild, gcc), axis=-1)
                    elif self._feat == 'spec_ild_sin_cos':
                        feat = np.concatenate((mbms, ild, ipdx, ipdy), axis=-1)
                    elif self._feat == 'ild_gcc':
                        feat = np.concatenate((ild, gcc), axis=-1)
                    elif self._feat == 'ild_sin_cos':
                        feat = np.concatenate((ild, ipdx, ipdy), axis=-1)
                    elif self._feat == 'sin_cos':
                        feat = np.concatenate((ipdx, ipdy), axis=-1)
                        
                if feat is not None:
                    print('{}: {}, {}'.format(file_cnt, file_name, feat.shape))
                    np.save(os.path.join(self._feat_dir, '{}.npy'.format(wav_filename.split('.')[0])), feat)

    def preprocess_features(self):
        # Setting up folders and filenames
        self._feat_dir = self.get_unnormalized_feat_dir()
        self._feat_dir_norm = self.get_normalized_feat_dir()
        create_folder(self._feat_dir_norm)
        normalized_features_wts_file = self.get_normalized_wts_file()
        spec_scaler = None

        # pre-processing starts
        if self._is_eval:
            spec_scaler = joblib.load(normalized_features_wts_file)
            print('Normalized_features_wts_file: {}. Loaded.'.format(normalized_features_wts_file))

        else:
            print('Estimating weights for normalizing feature files:')
            print('\t\tfeat_dir: {}'.format(self._feat_dir))

            spec_scaler = preprocessing.StandardScaler()
            for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
                print('{}: {}'.format(file_cnt, file_name))
                feat_file = np.load(os.path.join(self._feat_dir, file_name))
                spec_scaler.partial_fit(feat_file)
                del feat_file
            joblib.dump(
                spec_scaler,
                normalized_features_wts_file
            )
            print('Normalized_features_wts_file: {}. Saved.'.format(normalized_features_wts_file))

        print('Normalizing feature files:')
        print('\t\tfeat_dir_norm {}'.format(self._feat_dir_norm))
        for file_cnt, file_name in enumerate(os.listdir(self._feat_dir)):
            print('{}: {}'.format(file_cnt, file_name))
            feat_file = np.load(os.path.join(self._feat_dir, file_name))
            feat_file = spec_scaler.transform(feat_file)
            np.save(
                os.path.join(self._feat_dir_norm, file_name),
                feat_file
            )
            del feat_file

        print('normalized files written to {}'.format(self._feat_dir_norm))

    # ------------------------------- EXTRACT LABELS AND PREPROCESS IT -------------------------------
    def extract_all_labels(self):
        self._label_dir = self.get_label_dir()

        print('Extracting labels:')
        print('\t\taud_dir {}\n\t\tdesc_dir {}\n\t\tlabel_dir {}'.format(
            self._aud_dir, self._desc_dir, self._label_dir))
        create_folder(self._label_dir)

        for file_cnt, file_name in enumerate(os.listdir(self._desc_dir)):
            wav_filename = '{}.wav'.format(file_name.split('.')[0])
            desc_file = self.load_output_format_file(os.path.join(self._desc_dir, file_name))
            #desc_file = self.convert_output_format_polar_to_cartesian(desc_file_polar)
            if self._use_binary_label:
                label_mat = self.get_binary_label(desc_file)
            else:
                label_mat = self.get_labels_for_file(desc_file)
            print('{}: {}, {}'.format(file_cnt, file_name, label_mat.shape))
            np.save(os.path.join(self._label_dir, '{}.npy'.format(wav_filename.split('.')[0])), label_mat)

    # -------------------------------  DCASE OUTPUT  FORMAT FUNCTIONS -------------------------------
    def load_output_format_file(self, _output_format_file):
        """
        Loads DCASE output format csv file and returns it in dictionary format

        :param _output_format_file: DCASE output format CSV
        :return: _output_dict: dictionary
        """
        _output_dict = {}
        _fid = open(_output_format_file, 'r')
        # next(_fid)
        for _line in _fid:
            _words = _line.strip().split(',')
            _frame_ind = int(_words[0])
            if _frame_ind not in _output_dict:
                _output_dict[_frame_ind] = []
            _output_dict[_frame_ind].append([float(_words[1]), float(_words[2]), float(_words[3])])
        _fid.close()
        return _output_dict

    def write_output_format_file(self, _output_format_file, _output_format_dict):
        """
        Writes DCASE output format csv file, given output format dictionary

        :param _output_format_file:
        :param _output_format_dict:
        :return:
        """
        _fid = open(_output_format_file, 'w')
        # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
        for _frame_ind in _output_format_dict.keys():
            for _value in _output_format_dict[_frame_ind]:
                # Write Cartesian format output. Since baseline does not estimate track count we use a fixed value.
                _fid.write('{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3])))
        _fid.close()

    def segment_labels(self, _pred_dict, _max_frames):
        '''
            Collects class-wise sound event location information in segments of length 1s from reference dataset
        :param _pred_dict: Dictionary containing frame-wise sound event time and location information. Output of SELD method
        :param _max_frames: Total number of frames in the recording
        :return: Dictionary containing class-wise sound event location information in each segment of audio
                dictionary_name[segment-index][class-index] = list(frame-cnt-within-segment, azimuth, elevation)
        '''
        nb_blocks = int(np.ceil(_max_frames/float(self._nb_label_frames_1s)))
        output_dict = {x: {} for x in range(nb_blocks)}
        for frame_cnt in range(0, _max_frames, self._nb_label_frames_1s):

            # Collect class-wise information for each block
            # [class][frame] = <list of doa values>
            # Data structure supports multi-instance occurence of same class
            block_cnt = frame_cnt // self._nb_label_frames_1s
            loc_dict = {}
            for audio_frame in range(frame_cnt, frame_cnt+self._nb_label_frames_1s):
                if audio_frame not in _pred_dict:
                    continue
                for value in _pred_dict[audio_frame]:
                    if value[0] not in loc_dict:
                        loc_dict[value[0]] = {}

                    block_frame = audio_frame - frame_cnt
                    if block_frame not in loc_dict[value[0]]:
                        loc_dict[value[0]][block_frame] = []
                    loc_dict[value[0]][block_frame].append(value[1:])

            # Update the block wise details collected above in a global structure
            for class_cnt in loc_dict:
                if class_cnt not in output_dict[block_cnt]:
                    output_dict[block_cnt][class_cnt] = []

                keys = [k for k in loc_dict[class_cnt]]
                values = [loc_dict[class_cnt][k] for k in loc_dict[class_cnt]]

                output_dict[block_cnt][class_cnt].append([keys, values])

        return output_dict

    def regression_label_format_to_output_format(self, _sed_labels, _doa_labels):
        """
        Converts the sed (classification) and doa labels predicted in regression format to dcase output format.

        :param _sed_labels: SED labels matrix [nb_frames, nb_classes]
        :param _doa_labels: DOA labels matrix [nb_frames, 2*nb_classes] or [nb_frames, 3*nb_classes]
        :return: _output_dict: returns a dict containing dcase output format
        """

        _nb_classes = self._nb_unique_classes
        _is_polar = _doa_labels.shape[-1] == 2*_nb_classes
        _azi_labels, _ele_labels = None, None
        _x, _y, _z = None, None, None
        if _is_polar:
            _azi_labels = _doa_labels[:, :_nb_classes]
            _ele_labels = _doa_labels[:, _nb_classes:]
        else:
            _x = _doa_labels[:, :_nb_classes]
            _y = _doa_labels[:, _nb_classes:2*_nb_classes]
            _z = _doa_labels[:, 2*_nb_classes:]

        _output_dict = {}
        for _frame_ind in range(_sed_labels.shape[0]):
            _tmp_ind = np.where(_sed_labels[_frame_ind, :])
            if len(_tmp_ind[0]):
                _output_dict[_frame_ind] = []
                for _tmp_class in _tmp_ind[0]:
                    if _is_polar:
                        _output_dict[_frame_ind].append([_tmp_class, _azi_labels[_frame_ind, _tmp_class], _ele_labels[_frame_ind, _tmp_class]])
                    else:
                        _output_dict[_frame_ind].append([_tmp_class, _x[_frame_ind, _tmp_class], _y[_frame_ind, _tmp_class], _z[_frame_ind, _tmp_class]])
        return _output_dict

    def convert_output_format_polar_to_cartesian(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:

                    ele_rad = tmp_val[3]*np.pi/180.
                    azi_rad = tmp_val[2]*np.pi/180

                    tmp_label = np.cos(ele_rad)
                    x = np.cos(azi_rad) * tmp_label
                    y = np.sin(azi_rad) * tmp_label
                    z = np.sin(ele_rad)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], x, y, z])
        return out_dict

    def convert_output_format_cartesian_to_polar(self, in_dict):
        out_dict = {}
        for frame_cnt in in_dict.keys():
            if frame_cnt not in out_dict:
                out_dict[frame_cnt] = []
                for tmp_val in in_dict[frame_cnt]:
                    x, y, z = tmp_val[2], tmp_val[3], tmp_val[4]

                    # in degrees
                    azimuth = np.arctan2(y, x) * 180 / np.pi
                    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180 / np.pi
                    r = np.sqrt(x**2 + y**2 + z**2)
                    out_dict[frame_cnt].append([tmp_val[0], tmp_val[1], azimuth, elevation])
        return out_dict

    # ------------------------------- Misc public functions -------------------------------

    def get_normalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_norm'.format(self._dataset_combination)
        )

    def get_unnormalized_feat_dir(self):
        return os.path.join(
            self._feat_label_dir,
            '{}'.format(self._dataset_combination)
        )

    def get_label_dir(self):
        if self._is_eval:
            return None
        else:
            if not self._approach == 'classification':
                return os.path.join(
                    self._feat_label_dir, '{}_label'.format(self._dataset_combination)
                    )
            else:
                return os.path.join(
                    self._feat_label_dir, '{}_label_class'.format(self._dataset_combination)
                    )

    def get_normalized_wts_file(self):
        return os.path.join(
            self._feat_label_dir,
            '{}_wts'.format(self._dataset)
        )

    def get_nb_channels(self):
        return self._nb_channels

    def get_nb_classes(self):
        return self._nb_unique_classes

    def nb_frames_1s(self):
        return self._nb_label_frames_1s

    def get_hop_len_sec(self):
        return self._hop_len_s

    def get_nb_frames(self):
        return self._max_label_frames

    def get_nb_mel_bins(self):
        return self._nb_mel_bins


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)

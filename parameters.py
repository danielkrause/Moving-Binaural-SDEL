# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
import numpy as np

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=False,     # To do quick test. Trains/test on small subset of dataset, and # of epochs

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='/scratch/asignal/krauseda/BinRot/1src/rotation/',

        # OUTPUT PATH
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='/scratch/asignal/krauseda/BinRot/1src/rotation/feat_label/',  # Directory to dump extracted features and labels

        model_dir='models/',   # Dumps the trained models and training curves in this folder
        dcase_output=True,     # If true, dumps the results recording-wise in 'dcase_dir' path.
                               # Set this true after you have finalized your model, save the output, and submit
        dcase_dir='results/',  # Dumps the recording-wise network output in this folder

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='binaural',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.01,
        label_hop_len_s=0.01,
        max_audio_len_s=10.,
        nb_mel_bins=256,

        # DNN MODEL PARAMETERS
        label_sequence_length=250,    # Feature sequence length
        batch_size=128,              # Batch size
        dropout_rate=0.,             # Dropout rate, constant for all layers
                   # Number of CNN nodes, constant for each layer
        f_pool_size=[2, 2, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=128,        # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_heads=4,

        nb_fnn_layers=2,
        fnn_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_fnn_sde_layers=2,
        fnn_sde_size=128,             # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=200,              # Train for maximum epochs
        lr=5e-4,
        branch_weights=[1, 10.],
        feat='all',
        per_file=False,
        unique_classes = 1,
        use_two_input = False,
        use_rot = True,
        use_trans = False,
        use_rot_trans = False,
        use_static_label=False,
        use_binary_label = False,
        use_three_labels = False,
        approach = 'regression',
        rot_cnn = False,
        merge_mode=None,
        kernel_size = (3, 3),
        nb_cnn2d_filt=[32, 64, 128],
        test_splits = [1, 2, 3, 4, 5],
        val_splits = [2, 3, 4, 5, 1],
        train_splits = [[3, 4, 5], [4, 5, 1], [5, 1, 2], [1, 2, 3], [2, 3, 4]],
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

########JOINT DOA AND SDE
######## SINGLE TASK
    elif argv == '850': # rotation scenario
        params['dataset_dir']='/scratch/asignal/krauseda/BinRot/1src/rotation_azi/'
        params['feat_label_dir']='/scratch/asignal/krauseda/BinRot/1src/rotation_azi/feat_label/'
        params['unique_classes'] = 1 # maximum number of overlapping sound events
        params['use_rot'] = True
        params['use_two_input'] = True
        params['rot_filt'] = 128
        params['rot_cnn'] = True
        params['f_pool_size'] = [4, 4, 4]
        params['merge_mode'] = 'concat'
        params['lr'] = 0.001/20.
        
        
    elif argv == '851': # static scenario
        params['dataset_dir']='/scratch/asignal/krauseda/BinRot/1src/static/'
        params['feat_label_dir']='/scratch/asignal/krauseda/BinRot/1src/static/feat_label/'
        params['unique_classes'] = 1 # maximum number of overlapping sound events
        params['use_rot'] = False
        params['use_two_input'] = False
        params['rot_filt'] = 128
        params['rot_cnn'] = False
        params['f_pool_size'] = [4, 4, 4]
        params['merge_mode'] = 'concat'
        params['lr'] = 0.001/20.
        
    elif argv == '852': # walking scenario 
        params['dataset_dir']='/scratch/asignal/krauseda/BinRot/1src_walk/rotation_azi/'
        params['feat_label_dir']='/scratch/asignal/krauseda/BinRot/1src_walk/rotation_azi/feat_label/'
        params['unique_classes'] = 1 # maximum number of overlapping sound events
        params['use_rot'] = True
        params['use_two_input'] = True
        params['rot_filt'] = 128
        params['rot_cnn'] = True
        params['f_pool_size'] = [4, 4, 4]
        params['merge_mode'] = 'concat'
        params['lr'] = 0.001/20. 
        

        
###### MULTITASK REGRESSION
 
    elif argv == '860': # rotation scenario
        params['dataset_dir']='/scratch/asignal/krauseda/BinRot/1src/rotation_azi/'
        params['feat_label_dir']='/scratch/asignal/krauseda/BinRot/1src/rotation_azi/feat_label/'
        params['unique_classes'] = 1 # maximum number of overlapping sound events
        params['use_rot'] = True
        params['use_two_input'] = True
        params['rot_filt'] = 128
        params['rot_cnn'] = True
        params['f_pool_size'] = [4, 4, 4]
        params['merge_mode'] = 'concat'
        params['lr'] = 0.001/20.
        params['branch_weights'] = [3., 1.]
        params['quick_test'] = True

    elif argv == '861':  # static scenario
        params['dataset_dir']='/scratch/asignal/krauseda/BinRot/1src/static/'
        params['feat_label_dir']='/scratch/asignal/krauseda/BinRot/1src/static/feat_label/'
        params['unique_classes'] = 1 # maximum number of overlapping sound events
        params['use_rot'] = False
        params['use_two_input'] = False
        params['rot_filt'] = 128
        params['rot_cnn'] = False
        params['f_pool_size'] = [4, 4, 4]
        params['merge_mode'] = 'concat'
        params['lr'] = 0.001/20.
        params['branch_weights'] = [3., 1.]
        
    elif argv == '862':  # walking scenario
        params['dataset_dir']='/scratch/asignal/krauseda/BinRot/1src_walk/rotation_azi/'
        params['feat_label_dir']='/scratch/asignal/krauseda/BinRot/1src_walk/rotation_azi/feat_label/'
        params['unique_classes'] = 1 # maximum number of overlapping sound events
        params['use_rot'] = True
        params['use_two_input'] = True
        params['rot_filt'] = 128
        params['rot_cnn'] = True
        params['f_pool_size'] = [4, 4, 4]
        params['merge_mode'] = 'concat'
        params['lr'] = 0.001/20.
        params['branch_weights'] = [3., 1.]
        
    
    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True
        params['epochs_per_fit'] = 1

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = np.ones_like(params['f_pool_size'])#[feature_label_resolution, 1, 1]     # CNN time pooling
    params['patience'] = 20#0.25*int(params['nb_epochs'])     # Stop training if patience is reached
    
    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params

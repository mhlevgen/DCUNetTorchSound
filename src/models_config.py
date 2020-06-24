SAMPLE_RATE = 16000
LOAD_SAMPLE_RATE = 48000
N_FFT = SAMPLE_RATE * 64 // 1000
HOP_LENGTH = SAMPLE_RATE * 16 // 1000
AUDIO_LEN = 49152
TARGET_SNR_LIST_TRAIN = [15, 10, 5, 0]
TARGET_SNR_LIST_TEST = [17.5, 12.5, 7.5, 2.5]

base_dict = {'SAMPLE_RATE': SAMPLE_RATE,
             'LOAD_SAMPLE_RATE': LOAD_SAMPLE_RATE,
             'N_FFT': N_FFT,
             'HOP_LENGTH': HOP_LENGTH,
             'AUDIO_LEN': AUDIO_LEN,
             'TARGET_SNR_LIST_TRAIN': TARGET_SNR_LIST_TRAIN,
             'TARGET_SNR_LIST_TEST': TARGET_SNR_LIST_TEST,
             'BATCH_SIZE': 40
             }


def get_model_config(model_features, encoder_depth):
    global size_dict
    if model_features == 32 and encoder_depth == 5:
        size_dict = {'enc_channels': [1,
                                      model_features,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2],
                     'enc_kernel_sizes': [(7, 5), (7, 5), (5, 3), (5, 3), (5, 3)],
                     'enc_strides': [(2, 2), (2, 2), (2, 2), (2, 2), (2, 1)],
                     'dec_channels': [0,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features,
                                      1],
                     'dec_kernel_sizes': [(5, 3), (5, 3), (5, 3), (7, 5), (7, 5)],
                     'dec_strides': [(2, 1), (2, 2), (2, 2), (2, 2), (2, 2)],
                     'dec_paddings': [(2, 1), (2, 1), (2, 1), (3, 2), (3, 2)]
                     }
    if model_features == 32 and encoder_depth == 8:
        size_dict = {'enc_channels': [1,
                                      model_features,
                                      model_features,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2],
                     'enc_kernel_sizes': [(7, 5), (7, 5), (7, 5), (5, 3),
                                          (5, 3), (5, 3), (5, 3), (5, 3)],
                     'enc_strides': [(2, 2), (2, 1), (2, 2), (2, 1),
                                     (2, 2), (2, 1), (2, 2), (2, 1)],
                     'dec_channels': [0,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features,
                                      model_features,
                                      1],
                     'dec_kernel_sizes': [(5, 3), (5, 3), (5, 3), (5, 3),
                                          (5, 3), (7, 5), (7, 5), (7, 5)],
                     'dec_strides': [(2, 1), (2, 2), (2, 1), (2, 2),
                                     (2, 1), (2, 2), (2, 1), (2, 2)],
                     'dec_paddings': [(2, 1), (2, 1), (2, 1), (2, 1),
                                      (2, 1), (3, 2), (3, 2), (3, 2)]
                     }
    if model_features == 45 and encoder_depth == 10:
        size_dict = {'enc_channels': [1,
                                      model_features,
                                      model_features,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      128],
                     'enc_kernel_sizes': [(7, 1), (1, 7), (7, 5), (7, 5), (5, 3),
                                          (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)],
                     'enc_strides': [(1, 1), (1, 1), (2, 2), (2, 1), (2, 2),
                                     (2, 1), (2, 2), (2, 1), (2, 2), (2, 1)],
                     'dec_channels': [0,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      1],
                     'dec_kernel_sizes': [(5, 3), (5, 3), (5, 3), (5, 3), (5, 3),
                                          (5, 3), (7, 5), (7, 5), (1, 7), (7, 1)],
                     'dec_strides': [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1),
                                     (2, 2), (2, 1), (2, 2), (1, 1), (1, 1)],
                     'dec_paddings': [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1),
                                      (2, 1), (3, 2), (3, 2), (0, 3), (3, 0)]
                     }

    if model_features == 32 and encoder_depth == 10:
        size_dict = {'enc_channels': [1,
                                      model_features,
                                      model_features,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      90],
                     'enc_kernel_sizes': [(7, 1), (1, 7), (7, 5), (7, 5), (5, 3),
                                          (5, 3), (5, 3), (5, 3), (5, 3), (5, 3)],
                     'enc_strides': [(1, 1), (1, 1), (2, 2), (2, 1), (2, 2),
                                     (2, 1), (2, 2), (2, 1), (2, 2), (2, 1)],
                     'dec_channels': [0,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features * 2,
                                      model_features,
                                      model_features,
                                      1],
                     'dec_kernel_sizes': [(5, 3), (5, 3), (5, 3), (5, 3), (5, 3),
                                          (5, 3), (7, 5), (7, 5), (1, 7), (7, 1)],
                     'dec_strides': [(2, 1), (2, 2), (2, 1), (2, 2), (2, 1),
                                     (2, 2), (2, 1), (2, 2), (1, 1), (1, 1)],
                     'dec_paddings': [(2, 1), (2, 1), (2, 1), (2, 1), (2, 1),
                                      (2, 1), (3, 2), (3, 2), (0, 3), (3, 0)]
                     }
    return size_dict

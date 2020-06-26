from collections import namedtuple

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
             'BATCH_SIZE': 10
             }


def get_model_config(model_features, encoder_depth):
    fields = ('in_channels',
              'out_channels',
              'kernel_size',
              'stride',
              'padding')
    layer_features = namedtuple('layer_features', fields)
    if model_features == 32 and encoder_depth == 5:
        encoder_dict = {'encoder_0': layer_features(1,
                                                    model_features,
                                                    (7, 5), (2, 2), None),
                        'encoder_1': layer_features(model_features,
                                                    model_features * 2,
                                                    (7, 5), (2, 2), None),
                        'encoder_2': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_3': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_4': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None)
                        }

        decoder_dict = {'decoder_0': layer_features(encoder_dict['encoder_4'].out_channels,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_1': layer_features((encoder_dict['encoder_3'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_2': layer_features((encoder_dict['encoder_2'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_3': layer_features((encoder_dict['encoder_1'].out_channels +
                                                     model_features * 2),
                                                    model_features,
                                                    (7, 5), (2, 2), (3, 2)),
                        'decoder_4': layer_features((encoder_dict['encoder_0'].out_channels +
                                                     model_features),
                                                    1,
                                                    (7, 5), (2, 2), (3, 2))
                        }
        return {'encoder': encoder_dict, 'decoder': decoder_dict}

    if model_features == 32 and encoder_depth == 8:
        encoder_dict = {'encoder_0': layer_features(1,
                                                    model_features,
                                                    (7, 5), (2, 2), None),
                        'encoder_1': layer_features(model_features,
                                                    model_features,
                                                    (7, 5), (2, 1), None),
                        'encoder_2': layer_features(model_features,
                                                    model_features * 2,
                                                    (7, 5), (2, 2), None),
                        'encoder_3': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None),
                        'encoder_4': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_5': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None),
                        'encoder_6': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_7': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None)
                        }

        decoder_dict = {'decoder_0': layer_features(encoder_dict['encoder_7'].out_channels,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_1': layer_features((encoder_dict['encoder_6'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_2': layer_features((encoder_dict['encoder_5'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_3': layer_features((encoder_dict['encoder_4'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_4': layer_features((encoder_dict['encoder_3'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_5': layer_features((encoder_dict['encoder_2'].out_channels +
                                                     model_features * 2),
                                                    model_features,
                                                    (7, 5), (2, 2), (3, 2)),
                        'decoder_6': layer_features((encoder_dict['encoder_1'].out_channels +
                                                     model_features),
                                                    model_features,
                                                    (7, 5), (2, 1), (3, 2)),
                        'decoder_7': layer_features((encoder_dict['encoder_0'].out_channels +
                                                     model_features),
                                                    1,
                                                    (7, 5), (2, 2), (3, 2))
                        }
        return {'encoder': encoder_dict, 'decoder': decoder_dict}
    if model_features == 45 and encoder_depth == 10:
        encoder_dict = {'encoder_0': layer_features(1,
                                                    model_features,
                                                    (7, 1), (1, 1), None),
                        'encoder_1': layer_features(model_features,
                                                    model_features,
                                                    (1, 7), (1, 1), None),
                        'encoder_2': layer_features(model_features,
                                                    model_features * 2,
                                                    (7, 5), (2, 2), None),
                        'encoder_3': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (7, 5), (2, 1), None),
                        'encoder_4': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_5': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None),
                        'encoder_6': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_7': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None),
                        'encoder_8': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_9': layer_features(model_features * 2,
                                                    128,
                                                    (5, 3), (2, 1), None)
                        }

        decoder_dict = {'decoder_0': layer_features(encoder_dict['encoder_9'].out_channels,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_1': layer_features((encoder_dict['encoder_8'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_2': layer_features((encoder_dict['encoder_7'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_3': layer_features((encoder_dict['encoder_6'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_4': layer_features((encoder_dict['encoder_5'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_5': layer_features((encoder_dict['encoder_4'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_6': layer_features((encoder_dict['encoder_3'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (7, 5), (2, 1), (3, 2)),
                        'decoder_7': layer_features((encoder_dict['encoder_2'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (7, 5), (2, 2), (3, 2)),
                        'decoder_8': layer_features((encoder_dict['encoder_1'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (1, 7), (1, 1), (0, 3)),
                        'decoder_9': layer_features((encoder_dict['encoder_0'].out_channels +
                                                     model_features * 2),
                                                    1,
                                                    (7, 1), (1, 1), (3, 0))
                        }
        return {'encoder': encoder_dict, 'decoder': decoder_dict}

    if model_features == 32 and encoder_depth == 10:
        encoder_dict = {'encoder_0': layer_features(1,
                                                    model_features,
                                                    (7, 1), (1, 1), None),
                        'encoder_1': layer_features(model_features,
                                                    model_features,
                                                    (1, 7), (1, 1), None),
                        'encoder_2': layer_features(model_features,
                                                    model_features * 2,
                                                    (7, 5), (2, 2), None),
                        'encoder_3': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (7, 5), (2, 1), None),
                        'encoder_4': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_5': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None),
                        'encoder_6': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_7': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), None),
                        'encoder_8': layer_features(model_features * 2,
                                                    model_features * 2,
                                                    (5, 3), (2, 2), None),
                        'encoder_9': layer_features(model_features * 2,
                                                    90,
                                                    (5, 3), (2, 1), None)
                        }

        decoder_dict = {'decoder_0': layer_features(encoder_dict['encoder_9'].out_channels,
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_1': layer_features((encoder_dict['encoder_8'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_2': layer_features((encoder_dict['encoder_7'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_3': layer_features((encoder_dict['encoder_6'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_4': layer_features((encoder_dict['encoder_5'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 1), (2, 1)),
                        'decoder_5': layer_features((encoder_dict['encoder_4'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (5, 3), (2, 2), (2, 1)),
                        'decoder_6': layer_features((encoder_dict['encoder_3'].out_channels +
                                                     model_features * 2),
                                                    model_features * 2,
                                                    (7, 5), (2, 1), (3, 2)),
                        'decoder_7': layer_features((encoder_dict['encoder_2'].out_channels +
                                                     model_features * 2),
                                                    model_features,
                                                    (7, 5), (2, 2), (3, 2)),
                        'decoder_8': layer_features((encoder_dict['encoder_1'].out_channels +
                                                     model_features),
                                                    model_features,
                                                    (1, 7), (1, 1), (0, 3)),
                        'decoder_9': layer_features((encoder_dict['encoder_0'].out_channels +
                                                     model_features),
                                                    1,
                                                    (7, 1), (1, 1), (3, 0))
                        }
        return {'encoder': encoder_dict, 'decoder': decoder_dict}

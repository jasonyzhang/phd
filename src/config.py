import os.path as osp
import sys

from absl import flags


curr_path = osp.dirname(osp.abspath(__file__))
model_dir = osp.join(curr_path, '..', 'models')
if not osp.exists(model_dir):
    print('Fix path to models/')
    import ipdb
    ipdb.set_trace()
SMPL_MODEL_PATH = osp.join(model_dir,
                           'neutral_smpl_with_cocoplustoesankles_reg.pkl')
SMPL_FACE_PATH = osp.join(curr_path, '../src/tf_smpl', 'smpl_faces.npy')

# Default pred-trained model path for the demo.
PRETRAINED_MODEL = osp.join(model_dir, 'model.ckpt-667589')

flags.DEFINE_string('smpl_model_path', SMPL_MODEL_PATH,
                    'path to the neutral smpl model')
flags.DEFINE_string('smpl_face_path', SMPL_FACE_PATH,
                    'path to smpl mesh faces (for easy rendering)')

# Model details

flags.DEFINE_string('load_path', None, 'path to trained model dir')
flags.DEFINE_integer('batch_size', 8, 'Size of mini-batch.')
flags.DEFINE_integer('num_conv_layers', 3, '# of layers for convolutional')
flags.DEFINE_boolean('use_delta_from_pred', True,
                     'If True, initialize delta regressor from pred.')
flags.DEFINE_boolean('pad_edges', False, 'If True, edge pad, else zero pad.')
flags.DEFINE_bool('use_optcam', True,
                  'If True, kp reprojection uses optimal camera.')
flags.DEFINE_integer('num_kps', 25, 'Number of keypoints.')


# For training.
flags.DEFINE_string('data_dir', None, 'Where tfrecords are saved')
flags.DEFINE_string('model_dir', None,
                    'Where model will be saved -- filled automatically')
flags.DEFINE_list('datasets', ['h36m', 'penn_action', 'insta_variety'],
                  'datasets to use for training')
flags.DEFINE_list('mocap_datasets', ['CMU', 'H3.6', 'jointLim'],
                  'datasets to use for adversarial prior training')
flags.DEFINE_list('pretrained_model_path', [PRETRAINED_MODEL],
                  'if not None, fine-tunes from this ckpt')
flags.DEFINE_string('image_encoder_model_type', 'resnet',
                    'Specifies which image encoder to use')
flags.DEFINE_string('temporal_encoder_type', 'AZ_FC2GN',
                    'Specifies which network to use for temporal encoding')
flags.DEFINE_integer('img_size', 224,
                     'Input image size to the network after preprocessing')
flags.DEFINE_integer('num_stage', 3, '# of times to iterate IEF regressor')
flags.DEFINE_integer('max_iteration', 5000000, '# of max iteration to train')
flags.DEFINE_integer('log_img_count', 10,
                     'Number of images in sequence to visualize')
flags.DEFINE_integer('log_img_step', 5000,
                     'How often to visualize img during training')

# Random seed
flags.DEFINE_integer('seed', 1, 'Graph-level random seed')


def get_config():
    config = flags.FLAGS
    config(sys.argv)
    return config

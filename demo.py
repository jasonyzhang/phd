"""
Runs PHD on Penn Action video.
"""

from glob import glob
import os
import os.path as osp
import warnings

from absl import flags
import numpy as np
from scipy.io import loadmat
import tensorflow as tf

from src.config import get_config
from src.evaluation.run_video import (
    process_image,
    process_videos,
    run_predictions,
)
from src.evaluation.tester_pred import TesterPred
from src.extract_tracks import (
    compute_tracks,
    get_labels_poseflow,
)
from src.renderer import VisRenderer
from src.util.smooth_bbox import get_smooth_bbox_params

flags.DEFINE_string('dataset', '',
                    'Dataset to use. Leave blank if using PoseFlow to extract'
                    ' tracks. Otherwise can set to "penn_action".')
flags.DEFINE_string('vid_id', '0001', 'Video id number if using Penn Action.')
flags.DEFINE_string('vid_path', 'data/0504.mp4',
                    'Path to filename if using PoseFlow for tracks')

flags.DEFINE_integer('ar_length', 25, 'Number of steps into future to predict.')
flags.DEFINE_integer('start_frame', 0, 'First frame of conditioning.')
flags.DEFINE_integer('skip_rate', None,
                     'If set, will be used for choosing subsequences.')
flags.DEFINE_integer('fps', 5, 'Frames per second in rendered video.')
flags.DEFINE_integer('degrees', '60', 'Angle for rotated viewpoint.')
flags.DEFINE_string('mesh_color', 'blue', 'Color of mesh.')

flags.DEFINE_string('out_dir', 'demo_output', 'Where to save final PHD videos.')
flags.DEFINE_string('penn_dir', 'data/Penn_Action',
                    'Directory where Penn Action is saved.')

NUM_CONDITION = 15

# Hides some TF warnings.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def load_penn_video(penn_dir, vid_id):
    im_paths = sorted(glob(osp.join(penn_dir, 'frames', vid_id, '*.jpg')))
    labels = loadmat(osp.join(penn_dir, 'labels', '{}.mat'.format(vid_id)))
    if np.all(labels['train'] == 1):
        print('Warning: {} is a training sequence!'.format(vid_id))
    kps = np.dstack((labels['x'], labels['y'], labels['visibility']))
    return im_paths, kps


def load_poseflow_video(vid_path, out_dir):
    track_json, im_dir = compute_tracks(vid_path=vid_path, out_dir=out_dir)
    im_paths = sorted(glob(osp.join(im_dir, '*.png')))
    kps = get_labels_poseflow(
        json_path=track_json,
        num_frames=len(im_paths),
        min_kp_count=NUM_CONDITION,
    )
    return im_paths, kps


def main(model):
    # Keypoints are only used to compute the bounding box around human tracks.
    # They are not fed into the model. Keypoint format is [x, y, vis]. Keypoint
    # order doesn't matter.
    if config.dataset == '':
        im_paths, kps = load_poseflow_video(config.vid_path, config.out_dir)
        vis_thresh = 0.1
    elif config.dataset == 'penn_action':
        im_paths, kps = load_penn_video(config.penn_dir, config.vid_id)
        vis_thresh = 0.5
    else:
        raise Exception('Dataset {} not recognized'.format(config.dataset))
    bbox_params_smooth, s, e = get_smooth_bbox_params(kps, vis_thresh)
    images = []
    min_f = max(s, 0)
    max_f = min(e, len(kps))
    for i in range(min_f, max_f):
        images.append(process_image(
            im_path=im_paths[i],
            bbox_param=bbox_params_smooth[i]
        ))
    all_images, vid_paths = process_videos(
        config=config,
        images=images,
        T=(NUM_CONDITION + config.ar_length),
        suffix='AR{}'.format(config.ar_length),
    )
    if not osp.exists(config.out_dir):
        os.mkdir(config.out_dir)
    renderer = VisRenderer(img_size=224)
    for i in range(0, len(all_images), config.batch_size):
        run_predictions(
            config=config,
            renderer=renderer,
            model=model,
            images=all_images[i : i + config.batch_size],
            vid_paths=vid_paths[i : i + config.batch_size],
            num_condition=NUM_CONDITION,
        )


if __name__ == '__main__':
    config = get_config()
    if config.skip_rate is None:
        setattr(config, 'batch_size', 1)
    model = TesterPred(
        config,
        sequence_length=(NUM_CONDITION + config.ar_length),
        resnet_path='models/hmr_noS5.ckpt-642561',
    )
    main(model)

import os
import os.path as osp

import deepdish as dd
import numpy as np
import tensorflow as tf

from src.models import (
    batch_pred_omega,
    get_image_encoder,
    get_prediction_model,
    get_temporal_encoder,
)
from src.omega import (
    OmegasPred,
)
from src.tf_smpl.batch_smpl import SMPL


class TesterPred(object):

    def __init__(self, config, sequence_length, resnet_path='', sess=None,
                 precomputed_phi=False):
        self.config = config
        self.load_path = config.load_path
        tf.set_random_seed(config.seed)

        self.num_conv_layers = 3
        self.fov = self.num_conv_layers * 4 + 1
        self.sequence_length = sequence_length
        self.use_delta_from_pred = config.use_delta_from_pred
        self.use_optcam = config.use_optcam
        self.precomputed_phi = precomputed_phi

        # Config + path
        if not config.load_path:
            raise Exception(
                'You need to specify `load_path` to load a pretrained model'
            )
        if not osp.exists(config.load_path + '.index'):
            print('{} doesnt exist'.format(config.load_path))
            import ipdb
            ipdb.set_trace()

        # Data
        self.batch_size = config.batch_size
        self.img_size = config.img_size
        self.E_var = []
        self.pad_edges = config.pad_edges

        self.smpl_model_path = config.smpl_model_path
        self.use_hmr_ief = False

        self.num_output = 85

        if precomputed_phi:
            input_size = (self.batch_size, self.sequence_length, 2048)
        else:
            input_size = (self.batch_size, self.sequence_length,
                          self.img_size, self.img_size, 3)
        self.images_pl = tf.placeholder(tf.float32, shape=input_size)

        strip_size = (self.batch_size, self.fov, 2048)
        self.movie_strips_pl = tf.placeholder(tf.float32, shape=strip_size)

        # Model Spec
        self.f_image_enc = get_image_encoder()
        self.f_temporal_enc = get_temporal_encoder()
        self.f_prediction_ar = get_prediction_model()

        self.smpl = SMPL(self.smpl_model_path)
        self.omegas_movie_strip = self.make_omega_pred()
        self.omegas_pred = self.make_omega_pred(use_optcam=True)

        # HMR Model Params
        self.num_stage = 3
        self.total_params = 85

        self.load_mean_omega()
        self.build_temporal_encoder_model()
        self.build_auto_regressive_model()
        self.update_E_vars()

        if sess is None:
            options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=options))
        else:
            self.sess = sess

        # Load data.
        self.prepare(resnet_path)

    def make_omega_pred(self, use_optcam=False):
        return OmegasPred(
            config=self.config,
            smpl=self.smpl,
            use_optcam=use_optcam,
            vis_max_batch=self.batch_size,
            is_training=False,
        )

    def update_E_vars(self):
        trainable_vars = tf.contrib.framework.get_variables()
        trainable_vars_e = [var for var in trainable_vars
                            if var.name[:2] != 'D_']
        self.E_var.extend(trainable_vars_e)

    def load_mean_omega(self):
        # Initialize scale at 0.9
        mean_path = os.path.join(os.path.dirname(self.smpl_model_path),
                                 'neutral_smpl_meanwjoints.h5')
        mean_vals = dd.io.load(mean_path)

        mean_cams = [0.9, 0, 0]
        # 72D
        mean_pose = mean_vals['pose']
        mean_pose[:3] = 0.
        mean_pose[0] = np.pi
        # 10D
        mean_shape = mean_vals['shape']

        mean_vals = np.hstack((mean_cams, mean_pose, mean_shape))
        # Needs to be 1 x 85
        mean_vals = np.expand_dims(mean_vals, 0)
        self.mean_var = tf.Variable(
            mean_vals,
            name='mean_param',
            dtype=tf.float32,
            trainable=True
        )
        mean_cams = self.mean_var[0, :3]
        mean_pose = self.mean_var[0, 3:3+72]
        mean_shape = self.mean_var[0, 3+72:]

        self.mean_vars = [mean_cams, mean_pose, mean_shape]
        mean_cams = tf.tile(tf.reshape(mean_cams, (1,-1)),
                            (self.batch_size, 1))
        mean_shape = tf.tile(tf.reshape(mean_shape, (1,-1)),
                             (self.batch_size, 1))
        mean_pose = tf.tile(tf.reshape(mean_pose, (1, 24, 3)),
                            (self.batch_size, 1, 1))
        _, mean_joints3d, mean_poses_rot = self.smpl(
            mean_shape, mean_pose, get_skin=True)

        # Starting point for IEF.
        self.theta_mean = tf.concat((
                mean_cams,
                tf.reshape(mean_pose, (-1, 72)),
                mean_shape
        ), axis=1)

    def prepare(self, resnet_path=''):
        """
        Restores variables from checkpoint.

        Args:
            resnet_path (str): Optional path to load resnet weights.
        """
        if resnet_path and not self.precomputed_phi:
            print('Restoring resnet vars from', resnet_path)
            resnet_vars = []
            e_vars = []
            for var in self.E_var:
                if 'resnet' in var.name:
                    resnet_vars.append(var)
                else:
                    e_vars.append(var)
            resnet_saver = tf.train.Saver(resnet_vars)
            resnet_saver.restore(self.sess, resnet_path)
        else:
            e_vars = self.E_var
        print('Restoring checkpoint ', self.load_path)

        saver = tf.train.Saver(e_vars)
        saver.restore(self.sess, self.load_path)
        self.sess.run(self.mean_vars)

    def build_temporal_encoder_model(self):
        B, T = self.batch_size, self.sequence_length
        if self.precomputed_phi:
            print('loading pre-computed phi!')
            self.img_feat_full = self.images_pl
        else:
            print('Getting all image features...')
            I_t = tf.reshape(
                self.images_pl,
                (B * T, self.img_size, self.img_size, 3)
            )
            img_feat, phi_var_scope = self.f_image_enc(
                I_t,
                is_training=False,
                reuse=False,
            )
            self.img_feat_full = tf.reshape(img_feat, (B, T, -1))

        omega_mean = tf.tile(self.theta_mean, (self.sequence_length, 1))

        # At training time, we only use first 40. Want to make sure GN
        # statistics are right.
        self.movie_strips_cond = self.f_temporal_enc(
            net=self.img_feat_full[:, :40],
            num_conv_layers=self.num_conv_layers,
            prefix='',
            reuse=None,
        )

        self.movie_strips = self.f_temporal_enc(
            net=self.img_feat_full,
            num_conv_layers=self.num_conv_layers,
            prefix='',
            reuse=True,
        )

        omega_movie_strip, _ = batch_pred_omega(
            input_features=self.movie_strips,
            batch_size=B,
            sequence_length=T,
            num_output=self.num_output,
            is_training=False,
            omega_mean=omega_mean,
            scope='single_view_ief',
            use_delta_from_pred=self.use_delta_from_pred,
            use_optcam=self.use_optcam,
        )

        self.omegas_movie_strip.append_batched(omega_movie_strip)
        self.omegas_movie_strip.compute_smpl()

    def build_auto_regressive_model(self):
        omega_mean = tf.tile(self.theta_mean, (self.fov, 1))
        input_movie_strips = self.movie_strips_pl  # B x 13 x 2048
        movie_strip_pred = self.f_prediction_ar(
            net=input_movie_strips,
            num_conv_layers=self.num_conv_layers,
            prefix='pred_',
            reuse=None,
        )
        omega_pred, _ = batch_pred_omega(
            input_features=movie_strip_pred,
            batch_size=self.batch_size,
            is_training=False,
            num_output=self.num_output,
            omega_mean=omega_mean,
            sequence_length=self.fov,
            scope='single_view_ief',
            predict_delta_keys=(),
            use_delta_from_pred=self.use_delta_from_pred,
            use_optcam=self.use_optcam,
        )
        # Only want the last entry.
        self.movie_strip_pred = movie_strip_pred[:, -1:]
        self.omegas_pred.append_batched(omega_pred[:, -1:])
        self.omegas_pred.compute_smpl()

    def make_fetch_dict(self, omegas, suffix=''):
        return {
            # Predictions.
            'cams' + suffix: omegas.get_cams(),
            'joints' + suffix: omegas.get_joints(),
            'kps' + suffix: omegas.get_kps(),
            'poses' + suffix: omegas.get_poses_rot(),
            'shapes' + suffix: omegas.get_shapes(),
            'verts' + suffix: omegas.get_verts(),
            'omegas' + suffix: omegas.get_raw(),
        }

    def predict_movie_strips(self, images, get_smpl=False):
        """
        Converts images to movie strip representation. Number of images should
        be equal to sequence_length. If precomputed_phi, images should be phis.

        Args:
            images (B x (2*fov-1) x H x W x 3) or (B x (2*fov-1) x 2048).
            get_smpl (bool): If True, returns all the smpl stuff.

        Returns:
            Movie strips (B x (2*fov-1) x 2048).
        """
        feed_dict = {
            self.images_pl: images,
        }
        fetch_dict = {
            'movie_strips': self.movie_strips,
            'movie_strips_cond': self.movie_strips_cond,
        }
        if get_smpl:
            fetch_dict.update(self.make_fetch_dict(self.omegas_movie_strip))
        return self.sess.run(fetch_dict, feed_dict)

    def predict_auto_regressive(self, movie_strips):
        """
        Predicts the next time step in an auto-regressive manner.

        Args:
            movie_strips (B x fov x 2048).

        Returns:

        """
        feed_dict = {
            self.movie_strips_pl: movie_strips,
        }

        fetch_dict = {
            'movie_strip': self.movie_strip_pred,
        }
        fetch_dict.update(self.make_fetch_dict(self.omegas_pred))
        result = self.sess.run(fetch_dict, feed_dict)
        return result

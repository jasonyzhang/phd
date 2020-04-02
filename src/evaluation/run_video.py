import os

import cv2
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from tqdm import tqdm

import src.util.render_utils as vis_util
from src.util.common import resize_img
from src.util.video import VideoWriter

IMG_SIZE = 224


def get_output_path_name(config, vid_id, suffix='', inds=()):
    """
    Returns the output video's path.

    Args:
        config: Configuration.
        vid_id (str): Id of video.
        suffix (str).
        inds (tuple): Indices of start and end of conditioning.

    Returns:
        str: Output video path name.
    """
    if inds:
        suffix += '_' + '-'.join(map(str, inds))
    suffix += '_fps{}'.format(config.fps)
    if config.dataset:
        output_name = '{dataset}-{vid}_{suf}.mp4'.format(
            dataset=config.dataset,
            vid=vid_id,
            suf=suffix,
        )
    else:
        output_name = '{vid}_{suf}.mp4'.format(
            vid=os.path.basename(config.vid_path).split('.')[0],
            suf=suffix,
        )

    return output_name


def process_videos(config, images, T, suffix=''):
    """

    Args:
        config: Configuration.
        images (list): list of images.
        T (int): Sequence length (conditioning length + AR length).

    Returns:
        all_images: List of list of images, each corresponding to a subseq
        all_vid_paths: Video names corresponding to each subseq.
    """
    start_frame = config.start_frame
    skip_rate = config.skip_rate
    n = len(images)
    if skip_rate is None:
        if start_frame + T > n:
            # In case we want to run past edge of video.
            images.extend([np.zeros((IMG_SIZE, IMG_SIZE, 3))] * T)
            n += T
        starts = [start_frame]
    else:
        starts = np.arange(start_frame, n - T, skip_rate, dtype=int)
    all_images, all_vid_paths = [], []
    for start in starts:
        end = start + T
        if n < end:
            print('Too short!')
        ims = images[start:end]
        vid_path = get_output_path_name(
            config=config,
            vid_id=config.vid_id.replace('mp4', ''),
            suffix=suffix,
            inds=(start, end),
        )
        all_images.append(ims)
        all_vid_paths.append(os.path.join(config.out_dir, vid_path))
    while len(all_images) % config.batch_size != 0:
        # Pad with garbage so that we fill up the batch.
        all_images.append(np.zeros((T, 224, 224, 3)))
        all_vid_paths.append('')
    return all_images, all_vid_paths


def process_image(im_path, bbox_param):
    """
    Processes an image, producing 224x224 crop.
    Args:
        im_path (str).
        bbox_param (3,): [cx, cy, scale].

    Returns:
        image
    """
    image = imread(im_path)
    center = bbox_param[:2]
    scale = bbox_param[2]

    # Pre-process image to [-1, 1]
    image = ((image / 255.) - 0.5) * 2
    image_scaled, scale_factors = resize_img(image, scale)
    center_scaled = np.round(center * scale_factors).astype(np.int)

    # Make sure there is enough space to crop 224x224.
    image_padded = np.pad(
        array=image_scaled,
        pad_width=((IMG_SIZE,), (IMG_SIZE,), (0,)),
        mode='edge'
    )
    height, width = image_padded.shape[:2]
    center_scaled += IMG_SIZE

    # Crop 224x224 around the center.
    margin = IMG_SIZE // 2

    start_pt = (center_scaled - margin).astype(int)
    end_pt = (center_scaled + margin).astype(int)
    end_pt[0] = min(end_pt[0], width)
    end_pt[1] = min(end_pt[1], height)
    image_scaled = image_padded[start_pt[1]:end_pt[1],
                                start_pt[0]:end_pt[0], :]
    return image_scaled


def run_predictions(config, renderer, model, images, vid_paths, num_condition):
    """

    Args:
        config: Configuration.
        renderer (VisRenderer).
        model (TesterPred).
        images (ndarray): B x T x H x W x 3.
        vid_paths (B).
        ar_length (int): Number of times to run auto-regressive prediction.
        num_condition (int): Condition length.
    """
    fov = model.fov
    ar_length = config.ar_length
    images = np.array(images)
    preds_gt = model.predict_movie_strips(images, get_smpl=True)
    movie_strips = preds_gt['movie_strips_cond']
    movie_strips = movie_strips[:, num_condition - fov: num_condition]
    verts_gt = preds_gt['verts'][:, -ar_length - fov:]  # B x (ar+f) x 6980 x 3
    verts = []
    for _ in tqdm(range(ar_length)):
        preds = model.predict_auto_regressive(movie_strips[:, -fov:])
        movie_strips = np.concatenate((
            movie_strips,
            preds['movie_strip'],  # B x 1 x 2048
        ), axis=1)
        verts.append(np.squeeze(preds['verts'], axis=1))

    verts = np.array(verts)  # ar x B x 6980 x 3!

    for i in range(len(images)):
        if vid_paths[i] == '':
            continue
        render_results(
            config=config,
            renderer=renderer,
            fov=fov,
            vid_path=vid_paths[i],
            images=images[i][-model.fov - ar_length:],
            verts=verts[:, i],
            verts_gt=verts_gt[i],
        )


def render_results(config, renderer, fov, vid_path, images, verts, verts_gt):
    """

    Args:
        config
        renderer (VisRenderer).
        fov (int).
        images ((f+ar) x H x W x 3).
        verts (ar x 6980 x 3): Predicted vertices from auto-regressive model.
        verts_gt ((f+ar) x 6980 x 3): Predicted vertices from real movie strips.
    """
    print('Rendering', vid_path)
    writer = VideoWriter(output_path=vid_path, fps=config.fps)
    images = (images + 1) * 0.5
    for i, im in tqdm(enumerate(images)):
        if im.shape[0] != IMG_SIZE:
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
        im = vis_util.draw_text(im, {'T': i - fov + 1})
        if i < fov:
            vert = verts_gt[i]
            color = 'yellow'
            im = vis_util.add_alpha(im)
        else:
            vert = verts[i - fov]
            color = config.mesh_color
            im = vis_util.add_alpha(im, 0.7)
        mesh = renderer(
            verts=vert,
            color_name=color,
            alpha=True,
            cam=np.array([0.7, 0, 0]),
        ) / 255.
        rot = renderer.rotated(
            verts=vert,
            deg=config.degrees,
            color_name=color,
            alpha=True,
            cam=np.array([0.7, 0, 0]),
        ) / 255.
        combined = np.hstack((im, mesh, rot))
        writer.add_image(combined)
    writer.make_video()
    writer.close()

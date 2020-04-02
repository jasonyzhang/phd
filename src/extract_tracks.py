"""
Given a directory of videos, extracts 2D pose tracklet using AlphaPose/PoseFlow
for each video.
Make sure you have installed AlphaPose in src/external.
This script is basically a wrapper around AlphaPose/PoseFlow:
1. Split the video into individual frames since PoseFlow requires that format
2. Run AlphaPose on the produced directory with frames
3. Run PoseFlow on the AlphaPose output.
Therefore, if at any point this script fails, please look into each system cmd
that's printed prior to running them. Make sure you can run those commands on
their own.
"""
import json
import os
import os.path as osp
import re
import subprocess
from glob import glob

import numpy as np


def dump_frames(vid_path, out_dir):
    """
    Extracts all frames from the video at vid_path and saves them inside of
    out_dir.
    """
    if len(glob(osp.join(out_dir, '*.png'))) > 0:
        print('Image frames already exist!')
        return

    print('{} Writing frames to file'.format(vid_path))

    cmd = [
        'ffmpeg',
        '-i', vid_path,
        '-start_number', '0',
        '{temp_dir}/frame%08d.png'.format(temp_dir=out_dir),
    ]
    print(' '.join(cmd))
    subprocess.call(cmd)


def run_alphapose(img_dir, out_dir):
    if osp.exists(osp.join(out_dir, 'alphapose-results.json')):
        print('Alpha Pose already run!')
        return

    print('----------')
    print('Computing per-frame results with AlphaPose')

    # Ex:
    # python3 demo.py --indir data/0502/ --outdir data/0502/alphapose --sp  \
    #       --cfg pretrained_models/256x192_res50_lr1e-3_1x.yaml  \
    #       --checkpoint pretrained_models/fast_res50_256x192.pth
    cmd = [
        'python', 'scripts/demo_inference.py',
        '--indir', img_dir,
        '--outdir', out_dir,
        '--sp',  # Needed to avoid multi-processing issues.
        # Update thees if you used a different model from the Model Zoo.
        '--cfg', 'pretrained_models/256x192_res50_lr1e-3_1x.yaml',
        '--checkpoint', 'pretrained_models/fast_res50_256x192.pth',
        # '--save_img',  # Uncomment this if you want to visualize poses.
    ]

    print('Running: {}'.format(' '.join(cmd)))
    curr_dir = os.getcwd()
    os.chdir('src/external/AlphaPose')
    ret = subprocess.call(cmd)
    if ret != 0:
        print('Issue running alphapose. Please make sure you can run the above '
              'command from the commandline.')
        exit(ret)
    os.chdir(curr_dir)
    print('AlphaPose successfully ran!')
    print('----------')


def run_poseflow(img_dir, out_dir):
    alphapose_json = osp.join(out_dir, 'alphapose', 'alphapose-results.json')
    out_json = osp.join(out_dir, 'poseflow', 'poseflow-results-tracked.json')
    if osp.exists(out_json):
        print('PoseFlow already run!')
        return out_json

    print('Computing tracking with PoseFlow')

    # Ex:
    # python PoseFlow/tracker-general.py --imgdir data/0502/   \
    #   --in_json demo_output/0502/alphapose/alphapose-results.json  \
    #   --out_json demo_output/0502/poseflow/poseflow-results-tracked.json  \
    #   --visdir demo_output/0502/poseflow/
    cmd = [
        'python', 'PoseFlow/tracker-general.py',
        '--imgdir', img_dir,
        '--in_json', alphapose_json,
        '--out_json', out_json,
        # '--visdir', out_dir,  # Uncomment this to visualize PoseFlow tracks.
    ]

    print('Running: {}'.format(' '.join(cmd)))
    curr_dir = os.getcwd()
    os.chdir('src/external/AlphaPose')
    ret = subprocess.call(cmd)
    if ret != 0:
        print('Issue running PoseFlow. Please make sure you can run the above '
              'command from the commandline.')
        exit(ret)
    os.chdir(curr_dir)
    print('PoseFlow successfully ran!')
    print('----------')
    return out_json


def compute_tracks(vid_path, out_dir):
    """
    This script basically:
    1. Extracts individual frames from mp4 since PoseFlow requires per frame
       images to be written.
    2. Call AlphaPose on these frames.
    3. Call PoseFlow on the output of 2.
    """
    vid_name = osp.basename(vid_path).split('.')[0]

    # Where to save all intermediate outputs in.
    vid_dir = osp.abspath(osp.join(out_dir, vid_name))
    img_dir = osp.abspath(osp.join(vid_dir, 'video_frames'))
    alphapose_dir = osp.abspath(osp.join(vid_dir, 'alphapose'))
    poseflow_dir = osp.abspath(osp.join(vid_dir, 'poseflow'))

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(alphapose_dir, exist_ok=True)
    os.makedirs(poseflow_dir, exist_ok=True)

    dump_frames(vid_path, img_dir)
    run_alphapose(img_dir, alphapose_dir)
    track_json = run_poseflow(img_dir, vid_dir)

    return track_json, img_dir


def get_labels_poseflow(json_path, num_frames, min_kp_count=15):
    """
    Returns the poses for each person tracklet.
    Each pose has dimension num_kp x 3 (x,y,vis) if the person is visible in the
    current frame. Otherwise, the pose will be None.
    Args:
        json_path (str): Path to the json output from AlphaPose/PoseTrack.
        num_frames (int): Number of frames.
        min_kp_count (int): Minimum threshold length for a tracklet.
    Returns:
        List of length num_people. Each element in the list is another list of
        length num_frames containing the poses for each person.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    if len(data.keys()) != num_frames:
        print('Not all frames have people detected in it.')
        frame_ids = [int(re.findall(r'\d+', img_name)[0])
                     for img_name in sorted(data.keys())]
        if frame_ids[0] != 0:
            print('PoseFlow did not find people in the first frame.')
            exit(1)

    all_kps_dict = {}
    all_kps_count = {}
    for i, key in enumerate(sorted(data.keys())):
        # People who are visible in this frame.
        track_ids = []
        for person in data[key]:
            kps = np.array(person['keypoints']).reshape(-1, 3)
            idx = int(person['idx'])
            if idx not in all_kps_dict.keys():
                # If this is the first time, fill up until now with None
                all_kps_dict[idx] = [None] * i
                all_kps_count[idx] = 0
            # Save these kps.
            all_kps_dict[idx].append(kps)
            track_ids.append(idx)
            all_kps_count[idx] += 1
        # If any person seen in the past is missing in this frame, add None.
        for idx in set(all_kps_dict.keys()).difference(track_ids):
            all_kps_dict[idx].append(None)

    all_kps_list = []
    all_counts_list = []
    for k in all_kps_dict:
        if all_kps_count[k] >= min_kp_count:
            all_kps_list.append(all_kps_dict[k])
            all_counts_list.append(all_kps_count[k])

    # Sort it by the length so longest is first:
    sort_idx = np.argsort(all_counts_list)[::-1]
    all_kps_list_sorted = []
    for sort_id in sort_idx:
        all_kps_list_sorted.append(all_kps_list[sort_id])
    print("Number of detected tracks:", len(all_kps_list_sorted))
    return all_kps_list_sorted[0]  # Just take the first track.

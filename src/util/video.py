import os
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm


def images_to_video(output_path, images, fps):
    writer = VideoWriter(output_path, fps)
    writer.add_images(images)
    writer.make_video()
    writer.close()


def sizeof_fmt(num, suffix='B'):
    """
    Returns the filesize as human readable string.

    https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-
        readable-version-of-file-size
    """
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return '%3.1f%s%s' % (num, unit, suffix)
        num /= 1024.0
    return '%.1f%s%s' % (num, 'Yi', suffix)


def get_dir_size(dirname):
    """
    Returns the size of the contents of a directory. (Doesn't include subdirs.)
    """
    size = 0
    for fname in os.listdir(dirname):
        fname = os.path.join(dirname, fname)
        if os.path.isfile(fname):
            size += os.path.getsize(fname)
    return size


class VideoWriter(object):

    def __init__(self, output_path, fps, temp_dir=None):
        self.output_path = output_path
        self.fps = fps
        self.temp_dir = temp_dir
        self.current_index = 0
        self.img_shape = None
        self.frame_string = 'frame{:08}.jpg'

    def add_images(self, images_list, show_pbar=False):
        """
        Adds a list of images to temporary directory.

        Args:
            images_list (iterable): List of images (HxWx3).
            show_pbar (bool): If True, displays a progress bar.

        Returns:
            list: filenames of saved images.
        """
        filenames = []
        if show_pbar:
            images_list = tqdm(images_list)
        for image in images_list:
            filenames.append(self.add_image(image))
        return filenames

    def add_image(self, image):
        """
        Saves image to file.

        Args:
            image (HxWx3).

        Returns:
            str: filename.
        """
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        if self.img_shape is None:
            self.img_shape = image.shape
        assert self.img_shape == image.shape
        filename = self.get_filename(self.current_index)
        plt.imsave(fname=filename, arr=image)
        self.current_index += 1
        return filename

    def get_frame(self, index):
        """
        Read image from file.

        Args:
            index (int).

        Returns:
            Array (HxWx3).
        """
        filename = self.get_filename(index)
        return plt.imread(fname=filename)

    def get_filename(self, index):
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        return os.path.join(self.temp_dir, self.frame_string.format(index))

    def make_video(self):
        cmd = ('ffmpeg -y -threads 16 -r {fps} '
               '-i {temp_dir}/frame%08d.jpg -profile:v baseline -level 3.0 '
               '-c:v libx264 -pix_fmt yuv420p -an -vf '
               '"scale=trunc(iw/2)*2:trunc(ih/2)*2" {output_path}'.format(
            fps=self.fps, temp_dir=self.temp_dir, output_path=self.output_path
        ))
        print(cmd)
        try:
            subprocess.call(cmd, shell=True)
        except OSError as e:
            import ipdb; ipdb.set_trace()
            print('OSError')

    def close(self):
        """
        Clears the temp_dir.
        """
        print('Removing {} which contains {}.'.format(
            self.temp_dir,
            self.get_temp_dir_size())
        )
        shutil.rmtree(self.temp_dir)
        self.temp_dir = None

    def get_temp_dir_size(self):
        """
        Returns the size of the temp dir.
        """
        return sizeof_fmt(get_dir_size(self.temp_dir))


class VideoReader(object):

    def __init__(self, video_path, temp_dir=None):
        self.video_path = video_path
        self.temp_dir = temp_dir
        self.frame_string = 'frame{:08}.jpg'

    def read(self):
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        cmd = ('ffmpeg -i {video_path} -start_number 0 '
               '{temp_dir}/frame%08d.jpg'.format(
            temp_dir=self.temp_dir,
            video_path=self.video_path
        ))
        print(cmd)
        subprocess.call(cmd, shell=True)
        self.num_frames = len(os.listdir(self.temp_dir))

    def get_filename(self, index):
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        return os.path.join(self.temp_dir, self.frame_string.format(index))

    def get_image(self, index):
        return plt.imread(self.get_filename(index))

    def get_images(self):
        i = 0
        fname = self.get_filename(i)
        while os.path.exists(fname):
            yield plt.imread(self.get_filename(i))
            i += 1
            fname = self.get_filename(i)

    def close(self):
        """
        Clears the temp_dir.
        """
        print('Removing {} which contains {}.'.format(
            self.temp_dir,
            self.get_temp_dir_size())
        )
        shutil.rmtree(self.temp_dir)
        self.temp_dir = None

    def get_temp_dir_size(self):
        """
        Returns the size of the temp dir.
        """
        return sizeof_fmt(get_dir_size(self.temp_dir))

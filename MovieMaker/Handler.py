from abc import ABC, abstractclassmethod
import subprocess as sp
import numpy as np
import imageio as io

class Handler(ABC):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def open(self, *args, **kwargs):
        pass
    
    @abstractclassmethod
    def write(self, *args, **kwargs):
        pass

    @abstractclassmethod
    def close(self, *args, **kwargs):
        pass


class hPipe(Handler):
    def __init__(self, filename, shape=None, fps=30, codec='libx264', quality=26, FFMPEG_DIR=r"C:\ffmpeg\bin"):
        super().__init__()
        self.FFMPEG_DIR = FFMPEG_DIR
        self.FFMPEG_EXE = FFMPEG_DIR+"\\ffmpeg.exe"

        ### Handler ###
        self.h = None

        ### Settings ###
        self.filename = filename
        self.shape = shape
        self.fps = fps
        self.codec = codec
        self.crf = quality

    def open(self):
        command = [self.FFMPEG_EXE,
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec','rawvideo',
            '-s', '{:d}x{:d}'.format(self.shape[1], self.shape[0]), # size of one frame
            '-pix_fmt', 'rgb24',
            '-r', '{}'.format(self.fps), # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            '-vcodec', self.codec,
            '-crf', str(self.crf),
            self.filename]

        self.h = sp.Popen(command, 
                        stdout=sp.PIPE, 
                        stdin=sp.PIPE, 
                        bufsize=10**8)

    def write(self, im):
        if type(im) == np.ndarray:
            im = im.tobytes()

        self.h.stdin.write(im)

    def close(self):
        if self.h:
            self.h.stdin.close()

            self.h.communicate()

class hImageIo(Handler):
    def __init__(self, filename, shape=None, fps=30, codec='libx264', quality=26, FFMPEG_DIR=r"C:\ffmpeg\bin"):
        super().__init__()
        self.FFMPEG_DIR = FFMPEG_DIR
        self.FFMPEG_EXE = FFMPEG_DIR+"\\ffmpeg.exe"

        ### Handler ###
        self.h = None

        ### Settings ###
        self.filename = filename
        self.shape = shape
        self.fps = fps
        self.codec = codec
        self.crf = quality

    def open(self):
        self.h = io.get_writer(self.filename, 
                fps=self.fps,
                codec=self.codec)

    def write(self, im):
        self.h.append_data(im)

    def close(self):
        if self.h:
            self.h.close()
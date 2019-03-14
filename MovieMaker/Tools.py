import subprocess as sp
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import matplotlib.pyplot as plt

def writeTextOnImage(im, 
                     text, 
                     position=(0,0), 
                     font_face='arial.ttf', 
                     font_color=(255,255,255), 
                     font_size=18, 
                     return_ndarray='auto'):
    
    converted = False
    
    # If image is numpy array, convert to PIL Image
    if type(im) == np.ndarray:
        im = Image.fromarray(im)
        converted = True
        
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_face, font_size)
    draw.text(position, text, font=font)
        
    # If numpy array is desired
    if return_ndarray and return_ndarray != 'auto':
        return np.array(im)
    
    # If input image was numpy array, return numpy array
    elif return_ndarray and converted:
        return np.array(im)

    # If image was PIL image, return PIL image
    else:
        return im
    
def writeCenteredTextOnImage(im,
                             text, 
                             wrap_width=None,
                             pad=10,
                             font_face='arial.ttf', 
                             font_color=(255,255,255), 
                             font_size=18, 
                             return_ndarray='auto'):
    
    converted = False
    
    # If image is numpy array, convert to PIL Image
    if type(im) == np.ndarray:
        im = Image.fromarray(im)
        converted = True
    
    # Check if text should be automatically wrapped
    if wrap_width is None:
        if type(text) == str:
            lines = [text]
            
        elif type(text) == tuple or type(text) == list:
            lines = text
            
        else:
            raise Exception('text must be either a string, a tuple or a list')
        
    # Auto wrap text
    else:
        lines = textwrap.wrap(text, width=wrap_width)

    # Get height and width of the image to align text
    H, W = im.height, im.width

    # Create draw unterface
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype(font_face, font_size)

    # Compute line dimensions
    line_dimensions = np.array([draw.textsize(line, font=font) for line in lines])

    # Compute height of first line
    line_h = H//2-(line_dimensions.mean(0)[1])*(1+line_dimensions.shape[0]/2)
  
    for i, line in enumerate(lines):
        # Draw text on current line, width centered
        draw.text(((W - line_dimensions[i,0]) / 2, line_h), # compute position
                  line, 
                  fill=font_color, 
                  font=font)
        
        # Increase line_h with padding
        line_h += line_dimensions.mean(0)[1] + pad
        
    # If numpy array is desired
    if return_ndarray and return_ndarray != 'auto':
        return np.array(im)
    
    # If input image was numpy array, return numpy array
    elif return_ndarray and converted:
        return np.array(im)

    # If image was PIL image, return PIL image
    else:
        return im
    

class Title:
    def __init__(self, 
                 title,
                 fadeIn=1, 
                 fadeOut=1, 
                 stay=3, 
                 fps=30, 
                 background_color=(0, 0, 0), 
                 font_face='arial.ttf',
                 font_color=(255, 255, 255), 
                 font_size=50,
                 wrap_width=None,
                 shape=(400, 400)):
        
        self.title = title
        self.fadeIn = fadeIn
        self.fadeOut = fadeOut
        self.stay = stay
        self.fps = fps
        self.background_color = background_color
        self.font_face = font_face
        self.font_color = font_color
        self.font_size = font_size
        self.wrap_width = wrap_width

        if type(shape) == tuple:
            self.shape = shape if len(shape) == 2 else shape[:2]
        
        if type(shape) == MovieMaker:
            self.shape = shape.shape[:2][::-1]
        
        self.total_frames = (fadeIn+fadeOut+stay)*fps
        
        self.slideshow = None

    def create(self):
        im = Image.new('RGB', self.shape, self.background_color)
        im = writeCenteredTextOnImage(im=im, 
                                      text=self.title, 
                                      wrap_width=self.wrap_width, 
                                      font_face=self.font_face, 
                                      font_color=self.font_color,
                                      font_size=self.font_size,
                                      return_ndarray=True)
        
        slideshow = np.zeros((self.total_frames, ) + im.shape, dtype=np.uint8)
        
        c = 0
        
        # fadeIn
        for i in range(self.fadeIn*self.fps):
            slideshow[c] = im * (i / (self.fadeIn*self.fps))
            c += 1
            
        for _ in range(self.stay*self.fps):
            slideshow[c] = im
            c += 1
            
        for i in range(self.fadeOut*self.fps):
            slideshow[c] = im * (1. - i / (self.fadeOut*self.fps))
            c += 1
            
        self.slideshow = slideshow
        
        return self.slideshow
        

class MovieMaker:
    def __init__(self, filename, fps=30, codec='libx264', quality=26, shape=None, FFMPEG_DIR=r"C:\ffmpeg\bin"):
        self.FFMPEG_DIR = FFMPEG_DIR
        self.FFMPEG_EXE = FFMPEG_DIR+"\\ffmpeg.exe"
        
        assert (quality >= 0 and quality <= 63), "Quality min: 0, max: 63"
        
        self.pipe = None
        self.shape = shape
        self.filename = filename
        self.fps = fps
        self.codec = codec
        self.crf = quality
        self.cache = []
        
    def clearCache(self):
        self.cache = []
        return True
    
    def _get_image_from_figure(self, fig, autoclose=True):
        fig.canvas.draw()
        rgb = fig.canvas.tostring_rgb()
        shape = fig.canvas.get_width_height()[::-1] + (3,)
        
        if self.shape is None:
            self.shape = shape
            
        elif self.shape != shape:
            raise ValueError('Shape from figure ({:d}x{:d}) is not as expected ({:d}x{:d})'.format(*shape, *self.shape))
        
        if autoclose:
            plt.close(fig)
            
        return rgb
    
    def addFigureToCache(self, fig, autoclose=True):
        self.cache.append(self._get_image_from_figure(fig, autoclose))

    def addImageToCache(self, im):
        self.cache.append(im.tobytes())    
    
    def writeFigure(self, fig, autoclose=True):
        im = self._get_image_from_figure(fig, autoclose)

        if self.pipe == None:
            self.openPipe()
        
        self.pipe.stdin.write(im)
        
    def writeImage(self, im):
        assert im.shape[-1] == 3, "Image is not RGB ({})".format(im.shape)
        
        b = im.tobytes()
        
        if self.pipe == None:
            self.openPipe()
            
        self.pipe.stdin.write(b)
        
    def addTitle(self, slideshow, to_beginning=True):
        if self.shape is None:
            self.shape = slideshow.shape[1:3]
        
        cache = []
        
        for s in slideshow:
            cache.append(s.tobytes())
            
        if to_beginning:
            self.cache = cache + self.cache
            
        else:
            self.cache.extend(cache)
        
    def writeCache(self, close_pipe=True):
        if self.pipe == None:
            self.openPipe()
            
        for c in self.cache:
            self.pipe.stdin.write(c)
            
        if close_pipe:
            self.close()
        
    def close(self):
        if self.pipe:
            self.pipe.stdin.close()
            self.pipe = None
        
    def openPipe(self):
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

        self.pipe = sp.Popen(command, 
                        stdout=sp.PIPE, 
                        stdin=sp.PIPE, 
                        bufsize=10**8)

    def cacheToArray(self):
        arr = []

        for e in self.cache:
            np
            arr.append(np.frombuffer(e, dtype=np.uint8).reshape(self.shape))

        return np.array(arr, dtype=np.uint8)

    def convertMovie2GIF(self, filename=None, skip=None, duration=None, fps=10):
        if filename is None:
            filename = self.filename

        command = [self.FFMPEG_EXE,
                '-y'] # (optional) overwrite output file if it exists]
            
        if skip:
            command += ['-ss', skip]

        if duration:
            command += ['-t', duration]

        command += ['-i', filename, # The imput comes from a pipe
                '-vf', 'fps={},scale=-1:-1:flags=lanczos,palettegen'.format(fps),
                'palette.png']

        pipe = sp.Popen(command, 

                        stdout=sp.PIPE, #, 
                        stdin=sp.PIPE) 
                        #bufsize=10**8)

        command = [self.FFMPEG_EXE,
                '-y'] # (optional) overwrite output file if it exists]
            
        if skip:
            command += ['-ss', skip]

        if duration:
            command += ['-t', duration]

        command += ['-i', filename, # The imput comes from a pipe
                '-i', 'palette.png', 
                '-filter_complex', 'fps={},scale=-1:-1:flags=lanczos [x]; [x][1:v] paletteuse'.format(fps),
                filename.replace('mp4','gif')]

        print(" ".join(command))

        pipe = sp.Popen(command, 
                        # shell=True,
                        # stdin=sp.PIPE,
                        stdout=sp.PIPE,
                        bufsize=10**8) 


if __name__ == '__main__':
    mm = MovieMaker(r"C:\Users\kistas\PycharmProjects\MovieMaker\test.mp4")
    # x0 = np.linspace(0, 4*np.pi, 1000)

    # for i in range(0, x0.size, 10):
    #     fig = plt.figure(figsize=(12,7), facecolor='white')
        
    #     plt.plot(x0[:i], np.sin(2 * x0[:i]))
    #     plt.xlim([0, x0.max()])
    #     plt.ylim([-1.2, 1.2])
        
    #     mm.addFigureToCache(fig)

    # t = Title("My fancy sine plot", shape=mm)

    # mm.addTitle(t.create())

    # mm.writeCache()
    mm.convertMovie2GIF()

import subprocess as sp
from PIL import Image, ImageDraw, ImageFont
import textwrap
import numpy as np
import matplotlib.pyplot as plt
import imageio as io
from Handler import hPipe, hImageIo

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
    def __init__(self, filename, handler='pipe', fps=30, codec='libx264', quality=26, shape=None, FFMPEG_DIR=r"C:\ffmpeg\bin"):
        self.FFMPEG_DIR = FFMPEG_DIR
        self.FFMPEG_EXE = FFMPEG_DIR+"\\ffmpeg.exe"
        
        assert (quality >= 0 and quality <= 63), "Quality min: 0, max: 63"

        ##### Settings #####
        self.filename = filename
        self.shape = shape
        self.fps = fps
        self.codec = codec
        self.crf = quality

        ##### Cache #####
        self.cache = [] # now numpy arrays are cached, not strings

        self.handler = handler
        self.h = None

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
            
        # Returns numpy array
        return np.frombuffer(rgb, dtype=np.uint8).reshape(shape)
    
    def addFigureToCache(self, fig, autoclose=True):
        self.cache.append(self._get_image_from_figure(fig, autoclose))

    def addImageToCache(self, im):
        self.cache.append(im)   

    def initHandler(self):
        if self.handler == 'io' or self.handler == 'imageio':
            self.h = hImageIo(self.filename)
            
        else:
            print(self.shape)
            self.h = hPipe(self.filename, 
                            shape=self.shape, 
                            fps=self.fps, 
                            codec=self.codec, 
                            quality=self.crf, 
                            FFMPEG_DIR=self.FFMPEG_DIR)

        self.h.open()
         
    
    def writeFigure(self, fig, autoclose=True):
        im = self._get_image_from_figure(fig, autoclose)

        if self.h is None:
            self.initHandler()

        self.h.write(im)
        
    def writeImage(self, im):
        assert im.shape[-1] == 3, "Image is not RGB ({})".format(im.shape)
        
        if self.h is None:
            self.initHandler()

        self.h.write(im)
        
    def addTitle(self, slideshow, to_beginning=True):
        if self.shape is None:
            self.shape = slideshow.shape[1:3]
        
        cache = []
        
        for s in slideshow:
            cache.append(s)
            
        if to_beginning:
            self.cache = cache + self.cache
            
        else:
            self.cache.extend(cache)
        
    def writeCache(self, close_pipe=True, subrectangles=True):
        if self.h is None:
            self.initHandler()

        for c in self.cache:
            self.h.write(c)

        self.h.close()
        
    def cacheToArray(self):
        arr = []

        for e in self.cache:
            np
            arr.append(np.frombuffer(e, dtype=np.uint8).reshape(self.shape))

        return np.array(arr, dtype=np.uint8)

    def toGIF(self, gif_filename=None, method='pipe'):
        if gif_filename is None:
            gif_filename = self.filename.replace('mp4','gif')

        movie2GIF(self.filename, gif_filename, method=method, fps=self.fps)

def movie2GIF(src, out, method='pipe', skip=None, duration=None, fps=10, subrectangles=True, FFMPEG_EXE=r"C:\ffmpeg\bin\ffmpeg.exe"):
    
    if method == 'io':
        frames = io.mimread(src)
        io.mimwrite(out, frames, fps=fps, subrectangles=subrectangles)

    else:
        command = [FFMPEG_EXE,
                '-y'] # (optional) overwrite output file if it exists]
            
        if skip:
            command += ['-ss', skip]

        if duration:
            command += ['-t', duration]

        command += ['-i', src, # The imput comes from a pipe
                '-vf', 'fps={},scale=-1:-1:flags=lanczos,palettegen'.format(fps),
                'palette.png']

        pipe = sp.Popen(command, 
                        stdout=sp.PIPE, #, 
                        stdin=sp.PIPE) 
                        #bufsize=10**8)

        pipe.stdin.close()
        pipe.communicate()

        command = [FFMPEG_EXE,
                '-y'] # (optional) overwrite output file if it exists]
            
        if skip:
            command += ['-ss', skip]

        if duration:
            command += ['-t', duration]

        command += ['-i', src, # The imput comes from a pipe
                '-i', 'palette.png', 
                '-filter_complex', 'fps={},scale=-1:-1:flags=lanczos [x]; [x][1:v] paletteuse'.format(fps),
                out]

        pipe = sp.Popen(command, 
                        stdout=sp.PIPE,
                        bufsize=10**8) 


if __name__ == '__main__':
    mm = MovieMaker(r"C:\Users\kistas\PycharmProjects\MovieMaker\test.mp4")
    x0 = np.linspace(0, 4*np.pi, 1000)

    for i in range(0, x0.size, 10):
        fig = plt.figure(figsize=(12,7), facecolor='white')
        
        plt.plot(x0[:i], np.sin(2 * x0[:i]))
        plt.xlim([0, x0.max()])
        plt.ylim([-1.2, 1.2])
        
        mm.addFigureToCache(fig)

    t = Title("My fancy sine plot", shape=mm)

    mm.addTitle(t.create())

    mm.writeCache()
    mm.toGIF()

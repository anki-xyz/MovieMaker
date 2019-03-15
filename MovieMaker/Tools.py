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
    def __init__(self, 
                 filename, 
                 handler='pipe', 
                 fps=30, 
                 codec='libx264', 
                 quality=26, 
                 shape=None, 
                 FFMPEG_DIR=r"C:\ffmpeg\bin"):
        """MovieMaker class that creates from single frames a movie
        
        Parameters
        ----------
        filename : str
            Path to the file that should be created
        handler : str, optional
            defines the handler ('pipe', 'io') (the default is 'pipe')
        fps : int, optional
            frames per second in movie (the default is 30)
        codec : str, optional
            defines the used codec, e.g. mpeg, libx264, libx265 (the default is 'libx264')
        quality : int, optional
            defines the quality of the movie (0...63, the -crf option in ffmpeg)
            (the default is 26, which is a good balance)
        shape : tuple, optional
            defines dimensions of the movie, will be set automatically
            from the first frame/image/plot that is defined.
            (the default is None, which means it is fetched automatically)
        FFMPEG_DIR : str, optional
            path to ffmpeg bin folder (the default is r"C:\ffmpeg\bin")
        
        """

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
        """clears internal cache
        
        Returns
        -------
        bool
            If operation was successful
        """

        self.cache = []
        return True
    
    def _get_image_from_figure(self, fig, autoclose=True):
        """Creates image from Figure object
        
        Parameters
        ----------
        fig : matplotlib.Figure
            The figure handler, e.g.
            fig = plt.figure()
        autoclose : bool, optional
            closes automatically the figure to free RAM
            (the default is True)
        
        Raises
        ------
        ValueError
            If shape from figure is not the same as previous shapes
        
        Returns
        -------
        numpy.ndarray
            The image as numpy array in the shape (x,y,3)
        """
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
        """adds a Figure (plt.figure()) to internal cache
        
        Parameters
        ----------
        fig : matplotlib.Figure
            The figure handler from a matplotlib plot
        autoclose : bool, optional
            closes automatically the figure to free RAM (the default is True)
        
        """
        self.cache.append(self._get_image_from_figure(fig, autoclose))

    def addImageToCache(self, im):
        """adds an Image to internal cache
        
        Parameters
        ----------
        im : numpy.ndarray
            Either grayscale or RGB array
        
        """

        if im.ndim == 2:
            im = np.repeat(im[...,None], 3, 2)

        self.cache.append(im)   

    def initHandler(self):
        """Initiates handler depending on handler type
        
        """

        if self.handler == 'io' or self.handler == 'imageio':
            self.h = hImageIo(self.filename)
            
        else:
            self.h = hPipe(self.filename, 
                            shape=self.shape, 
                            fps=self.fps, 
                            codec=self.codec, 
                            quality=self.crf, 
                            FFMPEG_DIR=self.FFMPEG_DIR)

        # Open handler
        self.h.open()
         
    
    def writeFigure(self, fig, autoclose=True):
        """Write Figure directly to file stream
        
        Parameters
        ----------
        fig : matplotlib.Figure
            The figure handler from a matplotlib plot
        autoclose : bool, optional
            closes automatically the figure to free RAM (the default is True)
        
        """

        im = self._get_image_from_figure(fig, autoclose)

        if self.h is None:
            self.initHandler()

        self.h.write(im)
        
    def writeImage(self, im):
        """Writes Image directly to stream
        
        Parameters
        ----------
        im : numpy.ndarray
            Either grayscale or RGB array
        
        """

        if im.ndim == 2:
            im = np.repeat(im[...,None], 3, 2)
        
        if self.h is None:
            self.initHandler()

        self.h.write(im)
        
    def addTitle(self, slideshow, to_beginning=True):
        """Adds title page to cache. If you would like to
        add title directly to stream, add the title page
        as single image frames via the ```write image``` method.
        
        Parameters
        ----------
        slideshow : numpy.ndarray
            A numpy ndarray that is creates using the Title.create() function

        to_beginning : bool, optional
            Defines if the title page should be added to the beginning of the movie,
            or at the end (the default is True)
        
        """

        # If there's no shape available yet
        if self.shape is None:
            self.shape = slideshow.shape[1:3]
        
        cache = []
        
        for s in slideshow:
            cache.append(s)

        # Put title page either to beginning or to the end  
        if to_beginning:
            self.cache = cache + self.cache
            
        else:
            self.cache.extend(cache)
        
    def writeCache(self, clear_cache=True, close_handler=True):
        """Writes cache to file
        
        Parameters
        ----------
        clear_cache : bool, optional
            clears the cache list (the default is True)
        close_handler : bool, optional
            closes handler object, finishes file (the default is True)
        
        """
        if self.h is None:
            self.initHandler()

        for c in self.cache:
            self.h.write(c)

        if clear_cache:
            self.clearCache()

        if close_handler:
            self.h.close()
        
    def cacheToArray(self):
        """Returns the cache list as numpy ndarray
        
        Returns
        -------
        numpy.ndarray
            Returns array with shape (frames, X, Y, 3)
        """

        return np.array(self.cache, dtype=np.uint8)

    def toGIF(self, gif_filename=None, method='pipe'):
        """Creates GIF from file
        
        Parameters
        ----------
        gif_filename : str, optional
            path to new GIF file (the default is None, which replaces mp4 ending with GIF)
        method : str, optional
            method to create GIF (the default is 'pipe')
        
        """
        if gif_filename is None:
            gif_filename = self.filename.replace('mp4','gif')

        movie2GIF(self.filename, gif_filename, method=method, fps=self.fps)

def framesToGIF(filename, frames, subrectangles=True, **kwargs):
    """Uses ImageIO to write a GIF file from frames
    
    Parameters
    ----------
    filename : str
        path to your new GIF file
    frames : numpy.ndarray
        Numpy array with the shape (frames, X, Y, 3)
    subrectangles : bool, optional
        Sets GIF compression method (the default is True)
    
    Returns
    -------
    bool
        If writing GIF file succeeded
    """
    if not filename.endswith('.gif'):
        filename += '.gif'

    return io.mimwrite(filename, frames, subrectangles=subrectangles, **kwargs)


def movie2GIF(src, 
              out, 
              method='pipe', 
              skip=None, 
              duration=None, 
              fps=10, 
              subrectangles=True, 
              FFMPEG_EXE=r"C:\ffmpeg\bin\ffmpeg.exe"):
    """Creates a GIF from a movie file using either ffmpeg 
    or imageio (that uses ffmpeg as well)
    
    Parameters
    ----------
    src : str
        Path to source file
    out : str
        Path to output file
    method : str, optional
        defines the method for GIF, either 'pipe' or 'imageio'
        (the default is 'pipe')
    skip : int, optional
        skips the first 'skip' seconds (the default is None, which skips nothing)
    duration : int, optional
        duration of video (the default is None, which takes the whole video)
    fps : int, optional
        sets the frames per second in GIF (the default is 10)
    subrectangles : bool, optional
        compression method for imageio (the default is True)
    FFMPEG_EXE : regexp, optional
        path to ffmpeg.exe (the default is r"C:\ffmpeg\bin\ffmpeg.exe")
    
    """
    if method == 'io':
        frames = io.mimread(src)
        io.mimwrite(out, frames, fps=fps, subrectangles=subrectangles)

    else:
        #### Create palette for high quality GIFs ####
        command = [FFMPEG_EXE,
                '-y'] # (optional) overwrite output file if it exists]
            
        # Skip part [s]
        if skip:
            command += ['-ss', skip]

        # Max   duration    seconds
        if duration:
            command += ['-t', duration]

        command += ['-i', src, # Source video
                '-vf', 'fps={},scale=-1:-1:flags=lanczos,palettegen'.format(fps),
                'palette.png']

        pipe = sp.Popen(command, 
                        stdout=sp.PIPE, 
                        stdin=sp.PIPE) 

        pipe.stdin.close()
        pipe.communicate() # Wait that pipe finished.

        #### Create GIF using PALETTE ####
        command = [FFMPEG_EXE,
                '-y'] # (optional) overwrite output file if it exists
            
        # Skip part [s]
        if skip:
            command += ['-ss', skip]

        # Max   duration    seconds
        if duration:
            command += ['-t', duration]

        command += ['-i', src, # Source video
                '-i', 'palette.png', # High res palette
                '-filter_complex', 'fps={},scale=-1:-1:flags=lanczos [x]; [x][1:v] paletteuse'.format(fps),
                out] # Filename of output

        # Use PIPE
        pipe = sp.Popen(command, 
                        stdout=sp.PIPE,
                        bufsize=10**8) 

        pipe.communicate() # Wait 


if __name__ == '__main__':
    mm = MovieMaker("test.mp4", handler='io')
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

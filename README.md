# MovieMaker

The MovieMaker can create movies using [imageio](https://github.com/imageio/imageio) or directly [ffmpeg](https://www.ffmpeg.org/).
You can provide a ```Figure``` object from [Matplotlib](https://matplotlib.org/) or just single [numpy](http://www.numpy.org/) arrays in RGB or grayscale.

The MovieMaker class provides two options

* directly stream image/figure to file
* add image/figure to cache and write cache to file

Further, you can add some title pages (doesn't have to be a title though)
with nice fade in and fade out effect. For ultimate flexibility you can return
the cache as a numpy array (internally, the cache is a ```list``` of ```np.ndarray```s).

# Installation

If you would like to use ImageIO, you need to install ```ffmpeg``` via the conda:

```
conda install ffmpeg -c conda-forge
```

or using imageio directly

```python
imageio.plugins.ffmpeg.download()
```

Clone the github repository and install it using

```
python -m pip install -e MovieMaker
```

# Usage

The main classes are ```MovieMaker``` and ```Title``` in ```MovieMaker.Tools```.

Here is some example code that creates an mp4 movie with a title page and a sine plot that is evolving over time.

```python

from MovieMaker.Tools import MovieMaker, Title
import numpy as np
import matplotlib.pyplot as plt

# Init MovieMaker with filename
mm = MovieMaker(r"C:\path\to\file\test.mp4")

# Create a figure that plots a sine wave over time
x0 = np.linspace(0, 4*np.pi, 1000)

for i in range(0, x0.size, 10):
    fig = plt.figure(figsize=(12,7), 
                     facecolor='white', 
                     dpi=96) # ensures that it is divisible by 16(!)
    
    plt.plot(x0[:i], np.sin(2 * x0[:i]))
    plt.xlim([0, x0.max()])
    plt.ylim([-1.2, 1.2])
    
    # Add Figure to cache, automatically close Figure
    mm.addFigureToCache(fig)

# Add a nice title to the movie with fade in and fade out
t = Title("My fancy sine plot", shape=mm)
mm.addTitle(t.create())

# Write cache to file
mm.writeCache()

# Create a GIF from the written file
mm.toGIF()

```

Here's the gif:

![The nice sine plot movie as gif][gif]

The above code needs on my machine 25.28 s with ```pipe``` handler,
and 21.78 seconds with ```imageio``` handler. 

# Todo

I need to explain better two helper functions that are useful:

* writeTextOnImage
* writeCenteredTextOnImage

(however, they seem to be quite self explanatory...)

Further, I would like to write some documentation about the complete usage.


[gif]: https://github.com/anki-xyz/MovieMaker/blob/master/test.gif "Nice sine plot GIF"

# MovieMaker

The MovieMaker can create movies using ffmpeg.
You can provide a ```Figure``` object from Matplotlib or just single numpy frames.

The MovieMaker class provides two options

* directly stream image/figure to file
* add image/figure to cache and write cache to file

Further, you can add some title pages (doesn't have to be a title though)
with nice fade in and fade out effect. For ultimate flexibility you can return
the cache (everything is stored as ```str```) as a numpy array.

# Installation

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
    fig = plt.figure(figsize=(12,7), facecolor='white')
    
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
mm.convertMovie2GIF()

```

Here's the gif:

![The nice sine plot movie as gif][gif]

# Todo

The docstring needs to be added to each function 
and I need to explain better two helper functions that are useful:

* writeTextOnImage
* writeCenteredTextOnImage

(however, they seem to be quite self explanatory...)

Further, I would like to write some documentation about the complete usage.


[gif]: https://github.com/anki-xyz/MovieMaker/blob/master/test.gif "Nice sine plot GIF"

# GetRotated

Using a neural network to rotate images, because I can.

### modeling choices
In the expected solution every pixel would be shifted among its radius,
this is possible using a single fully connected layer, with deactivated bias.

### interesting insights
It turned out to be quite effective using different data at different
stages of model training. Gradually increasing the amount of pixels,
until the given image becomes pure uniform noise.

## Results
For inference, I passed the initial image through the network and 
repeated with its output, until I achieved a whole rotation.  
In the final model, the results are almost identical to scipy's image rotation, 
showing that the model correctly learned the mapping for each pixel
(refresh page to synchronise rotations)

| **Scipy** | **Neural Network** |
|:---------------------------------------------:|:----------------------------------------------:|
| ![Rotating Line](resources/rotating_line_sy.gif) | ![Another Animation](resources/rotating_line.gif) |

<br>

| **Scipy** | **Neural Network** |
|:---------------------------------------------:|:----------------------------------------------:|
| ![Rotating Line](resources/rotating_cat_sy.gif) | ![Another Animation](resources/rotating_cat.gif) |

<br><br><br>
<br><br><br>
<br><br><br>
<br><br><br>

<figure>
    <img src="resources/get-rotated.gif" width="200" height="200">
</figure>

# GetRotated

- Using a neural network to rotate images, just for fun

### modeling choices 

- SGD, because every data instance contains the same operation

- a single fully connected layer, as every pixel should be shifted once

- start with zero bias, since it's not necessary for an optimal solution

## Results

![get-rotated.gif](get-rotated.gif)
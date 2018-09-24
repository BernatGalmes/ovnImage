# Ovnimage
Useful helper functions of image processing and machine learning

## Libraries
Libraries contained in the repository

### dataframe.py
Useful functions to manipulate pandas dataframe

### functions.py
Random useful functions

### images.py
Image processing functions

### masks.py
Functions to modify binary masks

### LikelihoodGenerator.py
Class to help to generate likelihood masks of regions of
  interest in images.

  Build your likelihood by marking with the mouse the contours of
  the roi region.

  Keyboard commands:
  > key a: Submit region selected.
  >
  > key b: discard last mouse click

  Example of use:
  ```python
  image = "RGB image matrix"
  path_output_file = "Path to save the likelihood"
  LikelihoodGenerator(image).build_your_mask(path_output_file)
  ```

#### Methods
> build_your_mask(path)
>
> Call this function to init the process of finger_regions selection.
>
> In the process:

>     Press A (shift + a) to end selection

>     Press b to cancel the last selection

>    When process ends the method return the mask build. This mask is the region
>    Inside the selected zone..
### plots.py
Functions to help plotting images.

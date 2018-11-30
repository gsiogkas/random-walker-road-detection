# random-walker-road-detection
Implementation of paper:
G. K. Siogkas and E. S. Dermatas, "Random-Walker Monocular Road Detection in Adverse Conditions Using Automated Spatiotemporal Seed Selection," in IEEE Transactions on Intelligent Transportation Systems, vol. 14, no. 2, pp. 527-538, June 2013.

doi: 10.1109/TITS.2012.2223686

URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6338335&isnumber=6521414


## Dependencies
- The first thing we need, is a module that allows for image I/O and as many basic manipulations as possible. We could go for OpenCV, but it would be an overkill and we could be even more Pythonic in this exercise. Also, we would like to be closer to Matlab style, to be able to compare the methods easier. For this reason, I chose scikit-image.
- For general array manipulations, indexing and code vectorization to be possible, of course numpy.
- For visualization, matplotlib and for easy streaming visualization at runtime (optional), drawnow.
- For file reading, directory content lists etc, os from native Python.
- For optical flow, there is a handy module named pyoptflow.
- For Otsu's thresholding method, scikit-image provides an implementation in filters.
- For basic processing time monitoring and logging (optional), time and logging from native Python will be used (I love it when things are so obvious).
- For nice process bars (optional), tqdm.
- Bonus track: the most difficult and tricky part is random walker. This is included in scikit-image. For performance, we should also install pyamg, numexpr and scikit-umfpack.
- A nice-to-have couple of modules that may come in handy, are also scikit-learn and scipy.
- Finally, jupyter for using ipython and jupyter notebooks. Also, (optional) jupyterlab.

## Setting up your environment using conda
Concluding, you can easily recreate my Linux python environment by installing miniconda and then running:

```
conda create -n rdpaper python=3.6 numpy scikit-image scikit-learn matplotlib scipy jupyter tqdm pyamg numexpr
source activate rdpaper
pip install pyoptflow
pip install scikit-umfpack
pip install drawnow

```

## Basic usage
You can run a smoke test on the 2 images included here by running test() from inside the python folder:
```python
import road_detection
road_detection.test()
```

## Running test on DIPLODOC sequence
You can run the algorithm on all the frames of the DIPLODOC sequence by running:

```python
import road_detection
results = road_detection.test_on_diplodoc_sequence()
```

This will give you a list of dictionaries with the performance metrics per frame.

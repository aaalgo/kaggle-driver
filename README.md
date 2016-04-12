# kaggle-driver

1. Dependency

Image DB and Streamer
https://github.com/aaalgo/picpac

Wrappers for various frameworks
https://github.com/aaalgo/xnn

Caffe with PicPac support
https://github.com/aaalgo/caffe-picpac

2. Data Import

Rename the 10 folders under train as 0, 1, ..., 9.

Import the data with 

```
# scan a directory for images and create a database
# named db
# the directory train should contain sub-directories
# named 0, 1, ... (n-1), with n = # category
picpac-import -f 2 train db
```

3. Training with Caffe

Update train_val.prototxt and set the parameter path
under picpac_param to the above produced db file.

The score 0.74081 was produced with 18634 iterations.

4. Prediction

Copy the caffe snapshot model to model/caffe.params.
Run "submit" to produce submition file.


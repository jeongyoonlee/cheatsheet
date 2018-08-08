# Keras + Tensorflow

## Troubleshooting

### Windows

#### Tensorflow crashes with `CUBLAS_STATUS_ALLOC_FAILED`

* Reference: [Stackoverflow](https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed)
* Cause: Tensorflow doesn't allocate all GPU memory available on Windows.
* Solution: Use dynamic memory growth as follows:
```python
import tensorflow as tf
tf.Session(config=tf.ConfigProto(allow_growth=True))
```

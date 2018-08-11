# Keras + Tensorflow

## Tensorflow Object Detection API

### Common Workflow

#### Object detection with a pre-trained model from model zoo

* Inputs:
  * `model_file`: a binary pre-trained model file (e.g. `frozen_inference_graph.pb` from `faster_rcnn_resnet101_coco_2018_01_28.tar.gz`)
  * `label_map_file`: a text file that contains the dictonary of classes and their index (e.g. `mscoco_label_map.pbtxt`)
  * `image_dir`: a directory that contains images
* Outputs:
  * `detection_file`: a CSV file to save detection results
* Load a frozen model into memory
```python
import tensorflow as tf

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
```
* Load a label map and get the categories and category index
```python
from utils import label_map_util

label_map = label_map_util.load_labelmap(label_map_file)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=n_class, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
```
* Detect objects in images and write a detection file
```python
from glob import glob
from PIL import Image
from tqdm import tqdm
import os
import pathlib

pathlib.Path(os.path.dirname(detection_file)).mkdir(parents=True, exist_ok=True)

# Create output detections file
with open(detection_file, 'w') as out_file:
    out_file.write('filename,width,height,class,score,xmin,ymin,xmax,ymax\n')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Get list of source images
            print('Getting list of source images')
            im_paths = glob(image_dir + "/*.jpg")

            for im_path in tqdm(im_paths):
                image = Image.open(im_path)
                (im_width, im_height) = image.size
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Actual detection
                (boxes, scores, classes, num) = sess.run(
                      [detection_boxes, detection_scores, detection_classes, num_detections],
                      feed_dict={image_tensor: image_np_expanded})
                boxes   = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.int32)
                scores  = np.squeeze(scores)
                # Write detections to output file
                # TF boxes = (ymin, xmin, ymax, xmax) normalized by width and height
                # Detection output is in (xmin, ymin, xmax, ymax)
                for i in range(int(num)):
                    if scores[i] > threshold:
                        if classes[i] in category_index.keys():
                            class_name = category_index[classes[i]]['name']
                        else:
                            class_name = 'N/A'
                        box = boxes[i].tolist()
                        out_file.write('%s,%d,%d,%s,%f,%d,%d,%d,%d\n' % (os.path.basename(im_path), im_width, im_height,
                                                                     class_name, scores[i],
                                                                     box[1] * im_width, box[0] * im_height,
                                                                     box[3] * im_width, box[2] * im_height))
```

#### Generating training data with labels in the TFrecords format from detection labels and images
* Reference: [Using your own dataset](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md) from the object detection Github
* Inputs
  * `detection_file`: a CSV file that contains detection results
  * `image_dir`: a directory that contains images related to `detection_file`
* Outputs
  * `train_record`: a TFRecords file for training that contains images and detection labels
  * `valid_record`: a TFRecords file for evaluation that contains images and detection labels
* Code
```python
import os
import io
import pandas as pd
import sys
import tensorflow as tf
from tqdm import tqdm

from PIL import Image
from collections import namedtuple, OrderedDict

sys.path.append('~/tensorflow/models/research/')
sys.path.append('~/tensorflow/models/research/slim/')
from object_detection.utils import dataset_util

# Which model was used for detection
MODEL_NAME = 'faster_rcnn_resnet101_coco_2018_01_28'

# Define input locations (images directory and detections file)
image_dir  = 'images'
detection_file  'detection.csv'
train_record = 'train.record'
valid_record = 'valid.record'

# '__background__' used as filler to denote unwanted class
OUT_CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']

flags = tf.app.flags
flags.DEFINE_string('csv_input', detection_file, 'Path to the CSV input')
flags.DEFINE_string('train_record', train_record, 'Path to output TFRecord for training')
flags.DEFINE_string('valid_record', VALID_RECORD, 'Path to output TFRecord for validation')
FLAGS = flags.FLAGS

def class_text_to_int(row_label):
    if row_label in OUT_CLASSES:
        return OUT_CLASSES.index(row_label)
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        cls = class_text_to_int(row['class'])
        # Include desired classes only
        if cls > 0:
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(cls)

    # Include image in TF records only if it has desired classes
    tf_example = None
    if len(classes) > 0:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    n_training = 2000       # inlcude every keep_record only

    writer = tf.python_io.TFRecordWriter(FLAGS.train_record)
    path = image_dir
    examples = pd.read_csv(FLAGS.csv_input)

    grouped = split(examples, 'filename')
    for i, group in enumerate(tqdm(grouped)):
        if i < n_training:
            tf_example = create_tf_example(group, path)
            # Write TF record if not empty
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

    writer.close()

    writer = tf.python_io.TFRecordWriter(FLAGS.valid_record)
    path = image_dir
    examples = pd.read_csv(FLAGS.csv_input)

    grouped = split(examples, 'filename')
    for i, group in enumerate(tqdm(grouped)):
        if i >= n_training:
            tf_example = create_tf_example(group, path)
            # Write TF record if not empty
            if tf_example is not None:
                writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
```

#### Training and evaluating a new object detection model with TFRecords data
* Reference: [export_inference_graph.py](https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py) at the object detection API Github
* Inputs
  * `config_file`: a JSON file for detection model training (e.g. faster_rcnn_resnet101_coco_2018_01_28.json`) that contains:
    * `model:fine_tune_checkpoint`: e.g. `faster_rcnn_resnet101_coco_2018_01_28/model.ckpt`
      * `train_input_reader:label_map_path`: a label map file (e.g. `mscoco_label_map.pbtxt`)
    * `train_input_reader:tf_record_input_reader:input_path`: a TFRecords file for training (e.g. `train_record`)
    * `eval_input_reader:label_map_path`: a label map file (e.g. `mscoco_label_map.pbtxt`)
    * `eval_input_reader:tf_record_input_reader`: a TFRecords file for evaluation (e.g. `valid_record`)
  * `train_dir`: a directory to save outputs from training
  * `train_dir`: a directory to save outputs from training
* Training
```python
from object_detection import trainer
from object_detection.builders import input_reader_builder, model_builder
from object_detection.utils import config_util, label_map_util
import functools
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

config_pipeline = config_util.get_configs_from_pipeline_file(config_file)
model_config = config_pipeline['model']
train_config = config_pipeline['train_config']
input_config = config_pipeline['train_input_config']

model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=True)
create_input_dict_fn = functools.partial(input_reader_builder.build, input_config)

trainer.train(create_tensor_dict_fn=create_input_dict_fn,
              create_model_fn=model_fn,
              train_config=train_config,
              train_dir=train_dir,
              num_clones=1,
              clone_on_cpu=False,
              ps_tasks=0,
              worker_replicas=1,
              worker_job_name='worker',
              task=0,
              is_chief=True,
              master='')
```
* Evaluation
```python
tf.logging.set_verbosity(tf.logging.INFO)

config_pipeline = config_util.get_configs_from_pipeline_file(config_file)
model_config = config_pipeline['model']
eval_config = config_pipeline['eval_config']
input_config = config_pipeline['eval_input_config']

model_fn = functools.partial(model_builder.build, model_config=model_config, is_training=False)
create_input_dict_fn = functools.partial(input_reader_builder.build, input_config)

label_map = label_map_util.load_labelmap(input_config.label_map_path)
max_num_classes = max([item.id for item in label_map.item])
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes)

evaluator.evaluate(create_input_dict_fn, model_fn, eval_config, categories, 
                   train_dir,
                   eval_dir)

```
* Saving the trained model
```python
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.gfile.GFile(config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)

step = 2000 # the number of epochs at which you want to save the model

exporter.export_inference_graph('image_tensor',
                                pipeline_config,
                                os.path.join(train_dir, 'model.ckpt-{}'.format(step)),
                                train_dir)

```
#### Object detection with the newly trained model

* Inputs:
  * `model_file`: the saved model file from the training step
  * `label_map_file`: a text file that contains the dictonary of classes and their index (e.g. `mscoco_label_map.pbtxt`)
  * `image_dir`: a directory that contains images
* Outputs:
  * `detection_file`: a CSV file to save detection results
* Code: See code from the object detection with a pre-trained model from model zoo


## Troubleshooting

### Ubuntu

#### Tensorflow installation from source failed with `/usr/bin/ld: cannot find Scrt1.o`

* Reference: [Stackoverflow](https://stackoverflow.com/questions/6329887/compiling-problems-cannot-find-crt1-o/9004026#9004026)
* Cause:
* Solution:
```bash
sudo ln -s /usr/lib/x86_64-linux-gnu /usr/lib64
```

### Windows

#### Tensorflow crashes with `CUBLAS_STATUS_ALLOC_FAILED`

* Reference: [Stackoverflow](https://stackoverflow.com/questions/41117740/tensorflow-crashes-with-cublas-status-alloc-failed)
* Cause: Tensorflow doesn't allocate all GPU memory available on Windows.
* Solution: Use dynamic memory growth as follows:
```python
import tensorflow as tf
tf.Session(config=tf.ConfigProto(allow_growth=True))
```

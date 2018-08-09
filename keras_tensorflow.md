# Keras + Tensorflow

## Tensorflow Object Detection API

### Common Workflow

#### Object detection with a pre-trained model from model zoo

* Inputs:
  * Pre-trained model (e.g. `frozen_inference_graph.pb` from `faster_rcnn_resnet101_coco_2018_01_28.tar.gz`)
  * Label map file (e.g. `mscoco_label_map.pbtxt`): Dictonary of classes and their index
  * Images (e.g. `*.jpg`)
* Outputs:
  * Detection results (e.g. `detection.csv` or `detection.json`)
* Code
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
from PIL import Image
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
            im_names = getFilesInDirectory(image_dir, ".jpg")

            for im_name in tqdm(im_names):
                im_path = os.path.join(image_dir,  im_name)
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
                        out_file.write('%s,%d,%d,%s,%f,%d,%d,%d,%d\n' % (im_name, im_width, im_height,
                                                                     class_name, scores[i],
                                                                     box[1] * im_width, box[0] * im_height,
                                                                     box[3] * im_width, box[2] * im_height))
```




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

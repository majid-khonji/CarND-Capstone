from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import rospy
import os

IMG_MAX_WIDTH = 300
IMG_MAX_HEIGHT = 300
MIN_SCORE_THRESHOLD = .5

# FROZEN_MODEL_PATH = "frozen_inference_graphs/frozen_inference_graph.pb"
# FROZEN_MODEL_PATH = "frozen_inference_graphs/frozen_inference_graph3.pb"
FROZEN_MODEL_FILE = "frozen_inference_graphs/frozen_inference_graph0.pb"
class TLClassifier(object):
    def __init__(self):
        self.model_graph = None
        self.session = None
        self.classes = {1: TrafficLight.RED,
                        2: TrafficLight.YELLOW,
                        3: TrafficLight.GREEN,
                        4: TrafficLight.UNKNOWN}

        cwd = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cwd, FROZEN_MODEL_FILE)
        self.load_model(model_path)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        class_index, probability = self.predict(image)

        if class_index is not None:
            rospy.logdebug("class: %d, probability: %f", class_index, probability)

        return class_index

    def load_model(self, model_path):
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.model_graph = tf.Graph()
        with tf.Session(graph=self.model_graph, config=config) as sess:
            self.session = sess
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.model_graph.get_tensor_by_name('num_detections:0')

    def predict(self, image_np, min_score_thresh=MIN_SCORE_THRESHOLD):
        image_np = cv2.resize(image_np, (IMG_MAX_WIDTH, IMG_MAX_HEIGHT))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        (boxes, scores, classes, num) = self.session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: np.expand_dims(image_np, axis=0)})
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)
        boxes = np.squeeze(boxes)

        # rospy.logwarn("tl_classifier: {} Traffic Light Class detected: classes: {} \n scores: {}".format(num, classes, scores))

        for i, box in enumerate(boxes):
            if scores[i] > min_score_thresh:
                light_class = self.classes[classes[i]]
                # rospy.logwarn("tl_classifier: Traffic Light Class detected: %d", light_class)
                return light_class, scores[i]

        return None, None



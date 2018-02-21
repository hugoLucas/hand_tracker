import tensorflow as tf
import numpy as np
import cv2 as cv

from tracking_utils.constants import PATH_TO_CKPT


def load_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def start_camera(camera=0):
    cap = cv.VideoCapture(camera)
    ret, image_np = cap.read()
    rows, columns, channels = image_np.shape

    return cap, rows, columns


def extract_tensors(detection_graph):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    return image_tensor, detection_boxes, detection_scores, detection_classes, num_detections


def detect_hands(frame, session, d_boxes, d_scores, d_classes, n_detections, i_tensor):
    image_detect = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_detect, axis=0)
    (boxes, scores, classes, num) = session.run([d_boxes, d_scores, d_classes, n_detections],
                                                feed_dict={i_tensor: image_np_expanded})

    return boxes, scores


def draw_boxes(frame, boxes, scores, frame_width, frame_height, threshold=0.50):
    max_score, max_box = 0, None
    for bx, sc in zip(boxes[0], scores[0]):
        if sc >= threshold:
            left, right, top, bottom = int(frame_width * bx[1]), int(frame_width * bx[3]), \
                                       int(frame_height * bx[0]), int(frame_height * bx[2])
            cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            if sc > max_score:
                max_score = sc
                max_box = (left, top, int(np.absolute(right - left)), int(np.absolute(bottom - top)))
    return max_box


def put_text(frame, message, frame_number, color=(0, 0, 255)):
    cv.putText(frame, '{} ({})'.format(message, frame_number), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1,
               color, 1, cv.LINE_AA)


def track_boxes(best_box, frame, output_boxes, output_scores, frame_width, frame_height):
    b_box = draw_boxes(frame, output_boxes, output_scores, frame_width, frame_height)
    return b_box if b_box is not None else best_box

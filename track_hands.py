import tensorflow as tf
import cv2 as cv

from utils.general_utils import load_graph, start_camera, extract_tensors, detect_hands, draw_boxes, put_text
from utils.constants import DETECTION_FRAMES, TOTAL_FRAMES

detection_graph = load_graph()
cap, height, width = start_camera()


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        i_tensor, d_boxes, d_scores, d_classes, n_detections = extract_tensors(detection_graph)

        frame_tracker, best_box = 0, None
        while True:
            ret, frame = cap.read()

            if frame_tracker < DETECTION_FRAMES:
                put_text(frame, 'DETECTING', frame_tracker)
                boxes, scores = detect_hands(frame, sess, d_boxes, d_scores, d_classes, n_detections, i_tensor)
                b_box = draw_boxes(frame, boxes, scores, width, height)
                best_box = b_box if b_box is not None else best_box
            else:
                put_text(frame, 'TRACKING', frame_tracker, color=(255, 255, 0))

                if best_box is not None:
                    if frame_tracker == DETECTION_FRAMES:
                        tracker = cv.Tracker_create('MEDIANFLOW')
                    ok = tracker.init(frame, best_box)

                    ok, best_box = tracker.update(frame)
                    if ok:
                        p1, p2 = (int(best_box[0]), int(best_box[1])), (int(best_box[0] + best_box[2]),
                                                                        int(best_box[1] + best_box[3]))
                        cv.rectangle(frame, p1, p2, (200, 0, 0), 2, 1)
                    else:
                        cv.putText(frame, 'Hand Lost', (10, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 200), 3,
                                   cv.LINE_AA)
                else:
                    frame_tracker = TOTAL_FRAMES

            frame_tracker = (frame_tracker + 1) % TOTAL_FRAMES
            if frame_tracker == 0:
                best_box = None

            cv.imshow('CAMERA', frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break

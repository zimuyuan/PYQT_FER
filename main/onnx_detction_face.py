import cv2
import numpy as np
import onnx
import vision.utils.box_utils_numpy as box_utils
import time
import onnxruntime as ort


class face_detecter:
    def __init__(self):
        # self.label_path = "../models/voc-model-labels.txt"
        # self.class_names = [name.strip() for name in open(self.label_path).readlines()]
        self.onnx_path = "../models/onnx/Mb_Tiny_RFB_FD_train_input_320.onnx"
        self.class_names = ['BACKGROUND', 'face']

        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.threshold = 0.7

    def pre_processing_image(self, image):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        # image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        return image

    def predit_image(self, image):

        input_image = self.pre_processing_image(image)
        # time_time = time.time()
        confidences, boxes = self.ort_session.run(None, {self.input_name: input_image})

        # print("cost time:{}".format(time.time() - time_time))

        return self.predict(image.shape[1], image.shape[0], confidences, boxes, self.threshold)
        # boxes, labels, probs = self.predict(image.shape[1], image.shape[0], confidences, boxes, self.threshold)
        # for i in range(boxes.shape[0]):
        #     box = boxes[i, :]
        #     label = f"{self.class_names[labels[i]]}: {probs[i]:.2f}"
        #     if probs[i] > 0.97:
        #         w = box[2] - box[0]
        #         h = box[3] - box[1]
        #         if h > w:
        #             d = (h - w) / 2
        #             box[0] = box[0] - d
        #             box[2] = box[2] + d
        #         else:
        #             d = (w - h) / 2
        #             box[1] = box[1] - d
        #             box[3] = box[3] + d
        #
        #         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
        #         cv2.putText(image, label,
        #                     (box[0] + 20, box[1] + 40),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     1,  # font scale
        #                     (255, 0, 255),
        #                     2)  # line type
        # return image

    def predict(self, width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = box_utils.hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k,)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])
        if not picked_box_probs:
            return np.array([]), np.array([]), np.array([])
        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

        # return picked_box_probs[:, :4], np.array(picked_labels), picked_box_probs[:, 4]


def detection_main2():
    cap = cv2.VideoCapture(0)  # capture from camera

    ddd_face = face_detector()

    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            print("no img")
            break

        orig_image = ddd_face.predit_image(orig_image)
        orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":

    # detcttion_face_main()
    detection_main2()


def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def detcttion_face_main():
    label_path = "../models/voc-model-labels.txt"
    onnx_path = "../models/onnx/Mb_Tiny_RFB_FD_train_input_320.onnx"
    class_names = [name.strip() for name in open(label_path).readlines()]
    ort_session = ort.InferenceSession(onnx_path)
    input_name = ort_session.get_inputs()[0].name

    cap = cv2.VideoCapture(0)  # capture from camera
    threshold = 0.7
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            print("no img")
            break
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        # image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        # confidences, boxes = predictor.run(image)

        confidences, boxes = ort_session.run(None, {input_name: image})

        boxes, labels, probs = predict(orig_image.shape[1], orig_image.shape[0], confidences, boxes, threshold)

        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

            if probs[i] > 0.9:
                cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

                cv2.putText(orig_image, label,
                            (box[0] + 20, box[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,  # font scale
                            (255, 0, 255),
                            2)  # line type

        orig_image = cv2.resize(orig_image, (0, 0), fx=0.7, fy=0.7)
        cv2.imshow('annotated', orig_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


"""
YOLO ONNX Detector

Responsibilities:
1. Load the YOLO ONNX model.
2. Preprocess OpenCV frames.
3. Run ONNX inference.
4. Convert model output into detection results.
"""

import logging

import cv2
import numpy as np
import onnxruntime as ort


logger = logging.getLogger(__name__)

# COCO dataset class names used by YOLOv8

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


class YOLOONNXDetector:
    def __init__(
        self,
        model_path: str,
        input_size: int = 640,
        confidence_threshold: float = 0.4,
        nms_threshold: float = 0.45,
    ):
        """
        Initialize the YOLO detector.

        The model is loaded only once when the worker starts.
        """

        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        logger.info("loading_yolo_onnx_model model_path=%s", model_path)

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Name of the model input tensor
        self.input_name = self.session.get_inputs()[0].name

        logger.info("yolo_onnx_model_loaded model_path=%s", model_path)

    # ---------------------------------------------------------
    # Preprocess Frame
    # ---------------------------------------------------------

    def preprocess(self, frame: np.ndarray):
        """
        Convert an OpenCV frame into YOLO input format.

        OpenCV:
            (H, W, C)
            BGR

        YOLO:
            (1, 3, 640, 640)
            RGB
            float32
        """

        original_height, original_width = frame.shape[:2]

        # Resize image
        image = cv2.resize(
            frame,
            (self.input_size, self.input_size),
        )

        # Convert BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # Normalize pixels to 0-1
        image = image.astype(np.float32) / 255.0

        # Convert HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1),
        )

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0,
        )

        return image, original_width, original_height

    # ---------------------------------------------------------
    # Run Model
    # ---------------------------------------------------------

    def inference(self, input_tensor):
        """
        Run the ONNX model.
        """

        outputs = self.session.run(
            None,
            {
                self.input_name: input_tensor,
            },
        )

        return outputs

    # ---------------------------------------------------------
    # Postprocess
    # ---------------------------------------------------------

    def postprocess(
        self,
        outputs,
        original_width,
        original_height,
    ):
        """
        Convert raw YOLO predictions into final detections.
        """

        predictions = outputs[0]

        # Remove batch dimension
        predictions = np.squeeze(predictions)

        # Convert (84,8400) -> (8400,84)
        predictions = predictions.T

        boxes = []
        confidences = []
        class_ids = []

        x_scale = original_width / self.input_size
        y_scale = original_height / self.input_size

        # Process every prediction
        for prediction in predictions:

            x_center, y_center, width, height = prediction[:4]

            class_scores = prediction[4:]

            class_id = int(np.argmax(class_scores))
            confidence = float(class_scores[class_id])

            # Skip weak detections
            if confidence < self.confidence_threshold:
                continue

            # Convert center format to rectangle
            x1 = int((x_center - width / 2) * x_scale)
            y1 = int((y_center - height / 2) * y_scale)

            w = int(width * x_scale)
            h = int(height * y_scale)

            boxes.append([x1, y1, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

        detections = []

        # No detections found
        if len(boxes) == 0:
            return detections

        # Remove duplicate overlapping boxes
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.confidence_threshold,
            self.nms_threshold,
        )

        if len(indices) == 0:
            return detections

        for index in indices.flatten():

            x, y, w, h = boxes[index]

            detections.append(
                {
                    "class_id": class_ids[index],
                    "class_name": CLASS_NAMES[class_ids[index]],
                    "confidence": round(confidences[index], 2),
                    "bbox": [
                        x,
                        y,
                        x + w,
                        y + h,
                    ],
                }
            )

        return detections

    # ---------------------------------------------------------
    # Complete Detection Pipeline
    # ---------------------------------------------------------

    def detect(self, frame):
        """
        Complete detection pipeline.

        Frame
            ↓
        Preprocess
            ↓
        ONNX Runtime
            ↓
        Postprocess
            ↓
        Detection List
        """

        input_tensor, width, height = self.preprocess(frame)

        outputs = self.inference(input_tensor)

        detections = self.postprocess(
            outputs,
            width,
            height,
        )

        return detections

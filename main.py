import cv2
import time

# Function to detect objects using a webcam
def Camera():
    cam = cv2.VideoCapture(0)  # Use the default PC camera
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Load class names from file
    classNames = []
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    # Load model configuration and weights
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightpath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightpath, configPath)
    net.setInputSize(640, 640)  # Increase input size
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        start_time = time.time()
        success, img = cam.read()
        if not success:
            print("Failed to read from camera")
            break

        # Detect objects in the webcam image
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)  # Lowered threshold

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(bbox, confs, score_threshold=0.6, nms_threshold=0.4)

        if len(indices) != 0:
            for idx in indices.flatten():
                box = bbox[idx]
                classId = classIds[idx]
                confidence = confs[idx]

                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, f'{classNames[classId - 1]}: {confidence:.2f}', 
                            (box[0] + 10, box[1] + 20), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), thickness=2)

        # Calculate FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=2)

        # Display the processed video feed
        cv2.imshow('Processed Video', img)

        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

Camera()
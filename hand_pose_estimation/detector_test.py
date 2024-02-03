import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
# model = YOLO(r'C:\Users\mikhail.klyazhev\Desktop\study\PycharmProjects\MouseCursorHandControl\src\hand_pose_estimation\yolov8n-pose.pt')  # load a pretrained model
# model = YOLO(r"C:\Users\mikhail.klyazhev\Desktop\study\PycharmProjects\MouseCursorHandControl\src\hand_pose_estimation\weights\yolov8n-pose-freihand-best-3ep.pt")
model = YOLO(r"C:\Users\mikhail.klyazhev\Desktop\study\PycharmProjects\MouseCursorHandControl\src\hand_pose_estimation\weights\yolov8n-pose-freihand-last-30ep.pt")

# Using camera
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    # frame = cv2.flip(frame, 1)

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

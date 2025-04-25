from ultralytics import YOLO
import cv2
import time
import torch

# Check if CUDA is available and set the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load a pretrained YOLOv8 model and move it to the GPU if available
model = YOLO("yolov8n.pt").to(device)

def analyze_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    # Get the width and height of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_GPUv8.mp4', fourcc, 30.0, (frame_width, frame_height))

    # Open the text file to write detections
    with open('detections_video.txt', 'w') as f:
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run inference on the frame
            results = model(frame, device=device)

            # Draw the results on the frame and write to the text file
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    label = f"{model.names[class_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Write detection to the text file
                    f.write(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})\n")

            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the frame to the output video
            out.write(frame)

            # Display the frame
            cv2.imshow('YOLOv8 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Display total processing time
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

def analyze_camera():
    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get the width and height of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_GPUv8.mp4', fourcc, 30.0, (frame_width, frame_height))

    # Open the text file to write detections
    with open('detections_camera.txt', 'w') as f:
        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run inference on the frame
            results = model(frame, device=device)

            # Draw the results on the frame and write to the text file
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])
                    label = f"{model.names[class_id]} {confidence:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    # Write detection to the text file
                    f.write(f"Class: {model.names[class_id]}, Confidence: {confidence:.2f}, BBox: ({x1}, {y1}, {x2}, {y2})\n")

            # Calculate and display FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the frame to the output video
            out.write(frame)

            # Display the frame
            cv2.imshow('YOLOv8 Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Display total processing time
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.2f} seconds")

def main():
    while True:
        print("\nMenu:")
        print("1. Analyze sample video")
        print("2. Use webcam")
        print("3. Exit")
        choice = input("Enter your choice: ")

        if choice == '1':
            analyze_video('sampleVideo.mp4')
        elif choice == '2':
            analyze_camera()
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == '__main__':
    main()
import logging
import cv2
from ultralytics import YOLO

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_model(config_path):
    try:
        model = YOLO(config_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def process_video(video_path, output_path, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file {video_path}")
        return
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply the model to the frame
        results = model(frame)
        
        # Draw the results on the frame
        for result in results:
            annotated_frame = result.plot()  # Use the plot method of the result object
        
        # Write the frame into the output file
        out.write(annotated_frame)
    
    # Release everything
    cap.release()
    out.release()
    logging.info(f"Processed video saved to {output_path}")

def main():
    setup_logging()

    model_config_path = "runs/detect/train9/weights/best.pt"  # Path to your model config or trained model
    video_path = "data/vid/bus.mp4"  # Correct path to your input video
    output_path = "data/vid/bus_processed.mp4"  # Path to save the output video

    model = load_model(model_config_path)

    try:
        process_video(video_path, output_path, model)
        logging.info("Video processing completed successfully.")
    except Exception as e:
        logging.error(f"Error during video processing: {e}")

if __name__ == '__main__':
    main()
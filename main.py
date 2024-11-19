import logging
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

def get_augmentation_settings():
    return {
        'flipud': 0.5,  # 50% chance of vertical flip
        'fliplr': 0.5,  # 50% chance of horizontal flip
        'hsv_h': 0.015,  # Hue adjustment
        'hsv_s': 0.7,  # Saturation adjustment
        'hsv_v': 0.4,  # Exposure adjustment
        'translate': 0.1,  # Random translation
        'scale': 0.5,  # Random scaling
        'shear': 0.0,  # No shear
        'perspective': 0.0,  # No perspective
        'erasing': 0.4,  # Random erasing
        'mosaic': 1.0,  # Always use mosaic augmentation
    }

def get_training_params():
    return {
        'epochs': 1000,  # Number of training epochs
        'augment': True,  # Use data augmentation
        'dropout': 0.5,  # Dropout rate for regularization
        'weight_decay': 0.0005,  # Weight decay for regularization
        'patience': 50,  # Early stopping patience
    }

def main():
    setup_logging()

    model_config_path = "yolov8n.yaml"
    data_config_path = "data.yaml"

    model = load_model(model_config_path)

    augmentation = get_augmentation_settings()
    training_params = get_training_params()

    training_config = {**training_params, **augmentation}

    try:
        model.train(data=data_config_path, **training_config)
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}")

if __name__ == '__main__':
    main()

from src.detection_utils import train_yolo_model
from src import parameters

if __name__ == '__main__':

    annotations_data_directory = parameters.DIRECTORY_WITH_LABELED_TRAINING_EXAMPLES
    epochs = parameters.NUMBER_OF_EPOCHS
    image_size = parameters.IMAGE_SIZE

    train_yolo_model(annotations_data_directory, epochs, image_size)

    
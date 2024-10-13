from src.detection_utils import train_yolo_model
from src import parameters

if __name__ == '__main__':

    directory_with_dataset_to_be_annotated =  parameters.DIRECTORY_WITH_IMAGES_TO_BE_ANNOTATED

    make_predictions(directory_with_dataset_to_be_annotated)
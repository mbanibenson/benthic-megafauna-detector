import pandas as pd
from pathlib import Path
import yaml
import subprocess



def generate_data_config_file(annotations_data_directory):
    '''
    Given annotations stored in a directory, generate the YAML file
    
    '''
    dataset_path = Path(annotations_data_directory)
    
    #Set parameters
    classes_file = dataset_path / 'classes.txt'
    
    #Read class names
    with classes_file.open('r') as f:
        class_names = f.read().strip().split('\n')
    
    #Create data dictionary
    data = {
        'names': class_names,
        'nc': len(class_names),
        'test': f'images/test',
        'val': f'images/val',
        'train': f'images/train',
        'path': str(dataset_path.absolute()),
    }
    
    #Write dictionary to YAML
    yaml_file = dataset_path / 'data.yaml'
    
    with yaml_file.open('w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f'data.yaml file created successfully at {yaml_file}')

    return yaml_file


def train_yolo_model(annotations_data_directory, epochs=10, imgsz=640):
    '''
    Given directory to labeled examples, generate YAML file and train model
    
    '''
    #Generate YAML file
    yaml_file = generate_data_config_file(annotations_data_directory)

    train_command = ['yolo', 'train', f'epochs={epochs}', f'imgsz={imgsz}', 'model=yolo11n.pt', f'data={yaml_file}']

    completed_process = subprocess.run(train_command, )
    
    return completed_process


def make_predictions(directory_with_dataset_to_be_annotated):
    '''
    Scan the directory with model runs and execute inferencing
    
    '''
    directory_with_training_model_runs = Path.cwd() / 'runs/detect'

    assert directory_with_training_model_runs.exists(), 'Please train the model first'

    directory_with_last_model_run = list(sorted(directory_with_training_model_runs.iterdir(), key=lambda x: x.stem, reverse=True))[0]

    path_to_the_best_model_in_the_last_run = directory_with_last_model_run / 'weights/best.pt'

    #trained_model = YOLO(path_to_the_best_model_in_the_last_run)

    directory_with_dataset_to_be_annotated = Path(directory_with_dataset_to_be_annotated).absolute()

    print(f'Prediction model path: {str(path_to_the_best_model_in_the_last_run)}')

    #inference_results = trained_model.predict(source=directory_with_dataset_to_be_annotated)

    predict_command = ['yolo', 'detect', 'predict', f'model={str(path_to_the_best_model_in_the_last_run)}', f'source={directory_with_dataset_to_be_annotated}']

    completed_process = subprocess.run(predict_command, )
    
    return completed_process
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path, encoding='cp949')
    return data

def preprocess_data(data):
    emotion_to_character = {
        'happiness': 'Joy',
        'neutral': 'Joy',     
        'sadness': 'Sadness',
        'angry': 'Anger',
        'surprise': 'Joy',    
        'disgust': 'Disgust',
        'fear': 'Fear'
    }

    data['캐릭터'] = data['상황'].map(emotion_to_character)
    character_to_number = {
        'Joy': 0,
        'Sadness': 1,
        'Anger': 2,
        'Disgust': 3,
        'Fear': 4
    }

    data['캐릭터_번호'] = data['캐릭터'].map(character_to_number)
    
    data_list = []
    for ques, label in zip(data['발화문'], data['캐릭터_번호']):
        data_list.append([ques, str(label)])
    
    dataset_train, dataset_test = train_test_split(data_list, test_size=0.2, shuffle=True, random_state=32)
    
    return dataset_train, dataset_test

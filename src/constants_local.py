BATCH_SIZE = 64
FOLDS = 5
GOEMOTIONS_TO_CEDR_MAPPING = {"0":"1", "2":"3", "3":"2", "4":"5", "6":"4", "8":"0"}
LABELS = ["no_emotion", "joy", "sadness", "surprise", "fear", "anger"]
LABEL_TRANSLATION = {"no_emotion": "нейтрально", "joy": "радость", "sadness": "грусть", "surprise": "удивление", "fear": "страх", "anger": "злость"}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
LABELS_DICT = dict(zip(range(6), LABELS))
NUM_LABELS = 6
PROBLEM_TYPE="multi_label_classification"
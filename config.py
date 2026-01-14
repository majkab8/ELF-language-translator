import os

DATA_DIR = "data"
XML_FILE = "eldamo.xml"
CSV_FILE = os.path.join(DATA_DIR, "elvish_dictionary.csv")
OUTPUT_DIR = "./eldamo_model_final"

MODEL_NAME = "google-t5/t5-small"
SOURCE_LANG = "english"
TARGET_LANG = "elvish"
MODEL_PREFIX = "translate English to Quenya: "

MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-4
EPOCHS = 10
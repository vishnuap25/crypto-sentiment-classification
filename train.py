import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sentence_transformers import SentenceTransformer

#initializations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = f"{BASE_DIR}/data/raw/crypto_currency_sentiment_dataset.csv"
OUTPUT_DIR = f"{BASE_DIR}/models/model"
encoder = SentenceTransformer('all-mpnet-base-v2')
label_map = {"Positive" : 1, "Negative" : 0}
early_stop = EarlyStopping(monitor='val_loss', patience=3,\
                            restore_best_weights=True)

#Loading Data
data = pd.read_csv(INPUT_DIR)
X = data["Comment"]
y = data["Sentiment"].map(label_map)

#Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(768,)),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer=Adam(learning_rate=0.0001),\
               loss='binary_crossentropy', metrics=['accuracy'])

#Encode & Train
encoded = encoder.encode(list(data['Comment'].apply(str.lower)), batch_size=64)
history = model.fit(encoded, y, \
                    epochs=50, batch_size=8, validation_split=0.2,\
                 callbacks=[early_stop], shuffle=True)

model.export(OUTPUT_DIR)
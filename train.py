import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sentence_transformers import SentenceTransformer


def parse_arguments():
    parser = argparse.ArgumentParser(description=\
        "Train a sentiment classification model for cryptocurrency comments.")
    parser.add_argument("--input_file", \
        type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", \
        type=str, required=True, help="Directory to save the trained model.")
    parser.add_argument("--epochs", \
        type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", \
        type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", \
        type=float, default=0.0001, help="Learning rate for Adam optimizer.")
    parser.add_argument("--dropout", \
        type=float, default=0.4, help="Dropout rate.")
    parser.add_argument("--validation_split", \
        type=float, default=0.2, help="Fraction of data to use for validation.")
    return parser.parse_args()


def load_data(input_file):
    data = pd.read_csv(input_file)
    label_map = {"Positive": 1, "Negative": 0}
    X = data["Comment"].astype(str).str.lower()
    y = data["Sentiment"].map(label_map)
    return X, y


def build_model(dropout_rate, learning_rate):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(768,)),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),\
                   loss='binary_crossentropy', metrics=['accuracy'])
    return model


def main():
    args = parse_arguments()
    
    # Load Data
    X, y = load_data(args.input_file)
    encoder = SentenceTransformer('all-mpnet-base-v2')
    X_encoded = encoder.encode(list(X), batch_size=64)
    
    # Build Model
    model = build_model(args.dropout, args.learning_rate)
    early_stop = EarlyStopping(monitor='val_loss', patience=3, \
                               restore_best_weights=True)
    
    # Train Model
    model.fit(X_encoded, y, epochs=args.epochs, batch_size=args.batch_size, \
              validation_split=args.validation_split, \
                callbacks=[early_stop], shuffle=True)
    
    # Save Model
    os.makedirs(args.output_dir, exist_ok=True)
    model.export(args.output_dir)
    print(f"Model exported to {args.output_dir}")


if __name__ == "__main__":
    main()

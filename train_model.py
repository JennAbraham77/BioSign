import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

df = pd.read_csv("gesture_data/gesture_dataset.csv")
df = df.dropna()

X = df.drop(columns=["user_id", "gesture_label"])
y = df["user_id"]

sequence_length = 30

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_sequences = []
y_sequences = []

for i in range(0, len(X_scaled) - sequence_length, sequence_length):
    X_seq = X_scaled[i:i + sequence_length]
    y_seq = y_encoded[i:i + sequence_length]

    if len(set(y_seq)) == 1:
        X_sequences.append(X_seq)
        y_sequences.append(y_seq[0])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

y_categorical = to_categorical(y_sequences)

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_sequences
)

model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(sequence_length, X_sequences.shape[2])))
model.add(Dropout(0.3))

model.add(LSTM(32))
model.add(Dropout(0.3))

model.add(Dense(32, activation="relu"))
model.add(Dense(y_categorical.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop]
)

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=encoder.classes_.astype(str)))

model.save("gesture_lstm_model.h5")

with open("gesture_lstm_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("gesture_label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("\nLSTM model, scaler, and label encoder saved successfully!")

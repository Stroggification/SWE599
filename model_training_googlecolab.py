import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Input, Add,
                                     LayerNormalization, GlobalAveragePooling1D,
                                     MultiHeadAttention)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from google.colab import drive
from sklearn.metrics import classification_report, roc_curve

import math

# -------------------------------
# CONFIG
# -------------------------------

if tf.config.list_physical_devices('GPU'):
    print("Using GPU for training!")
else:
    print("GPU not detected. Check runtime settings or driver installation.")


# Mount Google Drive
drive.mount('/content/drive')

# Update folder path
data_path = '/content/drive/MyDrive/balanced_data_sweet'

print(f"Data folder path: {data_path}")


# Split ratios for each user
train_ratio = 0.8  # 80% goes to (train+val), 20% goes to test
val_ratio   = 0.2  # of the (train+val) portion, 20% goes to val
# => net effect:  (train+val) = 80% of user data
#                 val = 0.2 * 0.8 = 16% of total
#                 train = 0.8 * 0.8 = 64% of total
#                 test = 20% of total

random_seed = 42
timesteps = 60
batch_size = 32
epochs = 1

# Choose which model architecture to use:
use_transformer = False  # Set to False to revert to LSTM

# Fix random seeds
random.seed(random_seed)
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# -------------------------------
# 1. User-Wise Splitting per CSV
# -------------------------------
train_parts = []
val_parts   = []
test_parts  = []

files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]

for file in files:
    df = pd.read_csv(file)
    # Assume last column is user ID
    user_col_name = df.columns[-1]

    # Group by each user in this CSV
    for user_id, user_data in df.groupby(user_col_name):
        # Shuffle each user's rows for random splitting
        user_data = user_data.sample(frac=1.0, random_state=random_seed)

        # 1) Split each user's data into train_val vs. test
        cutoff_train_val = int(train_ratio * len(user_data))  # e.g. 80%
        user_train_val = user_data.iloc[:cutoff_train_val]
        user_test      = user_data.iloc[cutoff_train_val:]

        # 2) Now split the train_val portion into train vs. val
        cutoff_val = int(val_ratio * len(user_train_val))  # e.g. 20% of train_val
        user_val   = user_train_val.iloc[:cutoff_val]
        user_train = user_train_val.iloc[cutoff_val:]

        # Accumulate
        train_parts.append(user_train)
        val_parts.append(user_val)
        test_parts.append(user_test)

# Concatenate all parts
train_df = pd.concat(train_parts, ignore_index=True)
val_df   = pd.concat(val_parts,   ignore_index=True)
test_df  = pd.concat(test_parts,  ignore_index=True)

# -------------------------------
# 2. Extract Features & Labels
# -------------------------------
X_train = train_df.iloc[:, :-1].values
y_train = train_df.iloc[:,  -1].values

X_val   = val_df.iloc[:, :-1].values
y_val   = val_df.iloc[:,  -1].values

X_test  = test_df.iloc[:, :-1].values
y_test  = test_df.iloc[:,  -1].values

print("Train shape (raw):", X_train.shape,
      "Val shape (raw):", X_val.shape,
      "Test shape (raw):", X_test.shape)

print("Unique users in train:", np.unique(y_train))
print("Unique users in val:",   np.unique(y_val))
print("Unique users in test:",  np.unique(y_test))

# -------------------------------
# 3. Scale & One-Hot Encode
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1))
y_val_encoded   = encoder.transform(y_val.reshape(-1, 1))
y_test_encoded  = encoder.transform(y_test.reshape(-1, 1))

num_features = X_train.shape[1]
num_classes  = y_train_encoded.shape[1]
print("Number of classes (users):", num_classes)

# -------------------------------
# 4. Generators for Sliding Windows
# -------------------------------
def windowed_batch_generator(X_data, y_data, timesteps= timesteps, batch_size=32, shuffle=True):
    """
    Infinite generator that yields batches of (X_window, y_window).
    Each X_window => (batch_size, timesteps, num_features)
    Each y_window => (batch_size, num_classes)
    """
    n_samples = len(X_data)
    indices = np.arange(n_samples - timesteps)

    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            X_batch = np.array([X_data[i:i+timesteps] for i in batch_idx], dtype=np.float32)
            y_batch = np.array([y_data[i+timesteps-1] for i in batch_idx], dtype=np.float32)
            yield X_batch, y_batch

def one_pass_window_generator(X_data, y_data, timesteps= timesteps, batch_size=32):
    """
    Single-pass generator (for test/eval).
    """
    n_samples = len(X_data)
    indices = np.arange(n_samples - timesteps)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        X_batch = np.array([X_data[i:i+timesteps] for i in batch_idx], dtype=np.float32)
        y_batch = np.array([y_data[i+timesteps-1] for i in batch_idx], dtype=np.float32)
        yield X_batch, y_batch

train_size = len(X_train)
val_size   = len(X_val)
test_size  = len(X_test)

train_steps = (train_size - timesteps) // batch_size
val_steps   = (val_size   - timesteps) // batch_size

print(f"train_steps={train_steps}, val_steps={val_steps}")

train_gen = windowed_batch_generator(X_train, y_train_encoded, timesteps, batch_size, shuffle=True)
val_gen   = windowed_batch_generator(X_val,   y_val_encoded,   timesteps, batch_size, shuffle=False)

# -------------------------------
# 5A. Define LSTM Model
# -------------------------------
def build_lstm_model(timesteps, num_features, num_classes):
    model = Sequential([
        LSTM(64, input_shape=(timesteps, num_features), return_sequences=True),
        LSTM(32),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------
# 5B. Define Transformer Model
# -------------------------------
def build_transformer_model(timesteps, num_features, num_classes, num_heads=4):
    """
    A simple Transformer encoder-based model for sequence classification.
    """
    inputs = tf.keras.Input(shape=(timesteps, num_features))

    # Multi-Head Self-Attention
    x = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=num_features,
        dropout=0.1
    )(inputs, inputs)  # self-attention on "inputs"

    # Residual & LayerNorm
    x = Add()([x, inputs])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward network
    ff = tf.keras.Sequential([
        Dense(64, activation='relu'),
        Dense(num_features),
    ])
    x_ff = ff(x)

    # Residual & LayerNorm
    x = Add()([x, x_ff])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Pooling (global average)
    x = GlobalAveragePooling1D()(x)

    # Classification head
    x = Dense(16, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# -------------------------------
# 5C. Choose Model & Train
# -------------------------------
if use_transformer:
    print("\nUsing TRANSFORMER model...\n")
    model = build_transformer_model(timesteps, num_features, num_classes, num_heads=4)
else:
    print("\nUsing LSTM model...\n")
    model = build_lstm_model(timesteps, num_features, num_classes)

with tf.device('/GPU:0'):
    history = model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs
    )

# -------------------------------
# 6. Evaluate on Test Set
# -------------------------------
test_steps = math.ceil((test_size - timesteps) / batch_size)
test_gen = one_pass_window_generator(X_test, y_test_encoded, timesteps, batch_size)

with tf.device('/GPU:0'):
    loss, accuracy = model.evaluate(test_gen, steps=test_steps)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# -------------------------------
# 7. Classification Report
# -------------------------------
y_pred_list = []
y_true_list = []

test_gen2 = one_pass_window_generator(X_test, y_test_encoded, timesteps, batch_size)
for X_batch, y_batch in test_gen2:
    preds = model.predict(X_batch, verbose = 0)
    y_pred_batch = np.argmax(preds, axis=1)
    y_true_batch = np.argmax(y_batch, axis=1)
    y_pred_list.append(y_pred_batch)
    y_true_list.append(y_true_batch)

y_pred_final = np.concatenate(y_pred_list)
y_true_final = np.concatenate(y_true_list)

# Convert numeric user IDs to strings to avoid TypeError in classification_report
class_names_str = [str(cls) for cls in encoder.categories_[0]]

report = classification_report(y_true_final, y_pred_final, target_names=class_names_str)
print(report)

# -------------------------------
# 8. EER Calculation
# -------------------------------

from joblib import Parallel, delayed

def compute_eer_for_one_class(c, y_true, y_scores):
    """
    Binary classification for class c vs. all others.
    """
    # Create binary labels: 1 if class = c, else 0
    y_true_bin = (y_true == c).astype(int)
    # Extract probabilities for class c
    y_scores_bin = y_scores[:, c]

    # Compute FPR/TPR for various thresholds
    fpr, tpr, _ = roc_curve(y_true_bin, y_scores_bin)
    fnr = 1 - tpr

    # Find the threshold index where FPR and FNR are closest
    abs_diff = np.abs(fpr - fnr)
    idx = np.argmin(abs_diff)

    # Equal Error Rate = average of FPR and FNR at that point
    eer_val = (fpr[idx] + fnr[idx]) / 2.0
    return c, eer_val

def compute_eer_per_class_parallel(y_true, y_scores, n_classes, n_jobs=-1):
    """
    Parallelizes EER computation for each class using joblib.

    :param y_true: 1D array of shape (num_samples,) with integer class labels.
    :param y_scores: 2D array of shape (num_samples, num_classes) with
                     predicted probabilities for each class.
    :param n_classes: number of classes (users)
    :param n_jobs: number of parallel jobs (default=-1 uses all cores)
    :return: dict {class_index: EER_value}
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_eer_for_one_class)(c, y_true, y_scores)
        for c in range(n_classes)
    )
    # Convert list of (class_idx, eer_val) to a dict
    return dict(results)

# 1) Gather predicted probabilities for entire test set
test_gen3 = one_pass_window_generator(X_test, y_test_encoded, timesteps, batch_size)
y_scores_list = []

for X_batch, _ in test_gen3:
    # Predict probabilities for this batch
    preds_prob = model.predict(X_batch, verbose=0)  # shape: (batch_size, num_classes)
    y_scores_list.append(preds_prob)

# 2) Concatenate into a single array, shape = (num_test_samples, num_classes)
all_y_scores = np.concatenate(y_scores_list, axis=0)

# 3) Compute EER in parallel
eers_dict = compute_eer_per_class_parallel(y_true_final, all_y_scores, num_classes, n_jobs=-1)

# 4) Print EER results
print("\nEqual Error Rate (EER) per class (one-vs-rest):")
for c in sorted(eers_dict.keys()):
    print(f"  Class {class_names_str[c]} => EER: {eers_dict[c]*100:.2f}%")

avg_eer = sum(eers_dict.values()) / len(eers_dict)
print(f"\nAverage EER across classes: {avg_eer*100:.2f}%")

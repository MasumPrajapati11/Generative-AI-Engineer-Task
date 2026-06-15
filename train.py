import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import pickle
import string

# =========================
# Load Dataset
# =========================

with open("data/shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.lower()

# Remove punctuation
text = text.translate(str.maketrans('', '', string.punctuation))

# =========================
# Tokenization
# =========================

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# =========================
# Create Sequences
# =========================

input_sequences = []

for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(token_list)):
        n_gram = token_list[:i+1]
        input_sequences.append(n_gram)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(
    pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding='pre'
    )
)

X = input_sequences[:, :-1]
y = input_sequences[:, -1]

y = tf.keras.utils.to_categorical(
    y,
    num_classes=total_words
)

# =========================
# Train Validation Split
# =========================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# Build Model
# =========================

model = Sequential([
    Embedding(total_words, 100,
              input_length=max_sequence_len - 1),

    LSTM(150, return_sequences=True),

    Dropout(0.2),

    LSTM(100),

    Dense(total_words,
          activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# =========================
# Callbacks
# =========================

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/text_generator.h5",
    save_best_only=True
)

# =========================
# Train
# =========================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=128,
    callbacks=[early_stop, checkpoint]
)

# =========================
# Save Tokenizer
# =========================

with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training Complete")
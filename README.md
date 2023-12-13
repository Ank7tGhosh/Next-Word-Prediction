# Next-Word-Prediction
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os

from google.colab import files
uploaded = files.upload()

file = open("gutenberg.txt", "r", encoding = "utf8")

# store file in list
lines = []
for i in file:
    lines.append(i)

# Convert list to string
data = ""
for i in lines:
  data = ' '. join(lines)

#replace unnecessary stuff with space
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')  #new line, carriage return, unicode character --> replace by space

#remove unnecessary spaces
data = data.split()
data = ' '.join(data)
data[:500]

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function
pickle.dump(tokenizer, open('token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:15]
len(sequence_data)
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
sequences = []

for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)

print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]
X = []
y = []

for i in sequences:
    X.append(i[0:3])
    y.append(i[3])

X = np.array(X)
y = np.array(y)
print("Data: ", X[:10])
print("Response: ", y[:10])
y = to_categorical(y, num_classes=vocab_size)
y[:5]

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))
model.summary()

!apt-get install graphviz -y
!pip install pydot pydotplus

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from IPython.display import Image

# Example model
model = Sequential()
model.add(Dense(10, input_shape=(5,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
Image('model_plot.png')    

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

# Sample data (replace with your actual data)
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Example input sequences (word indices)
y = np.array([[0, 1], [1, 0], [0, 1]])  # Example one-hot encoded target categories

# Define your model
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))  # Input layer with 64 neurons
model.add(Dense(32, activation='relu'))  # Hidden layer with 32 neurons
model.add(Dense(y.shape[1], activation='softmax'))  # Output layer with appropriate units for output categories

# Compile your model
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))

# Checkpoint setup
checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)

# Verify the shapes and formats of your X and y data before fitting the model
print("Shape of X:", X.shape if hasattr(X, 'shape') else "X is not properly defined")
print("Shape of y:", y.shape if hasattr(y, 'shape') else "y is not properly defined")

# Fit the model with your data
model.fit(X, y, epochs=70, batch_size=64, callbacks=[checkpoint])


from tensorflow.keras.models import load_model
import numpy as np
import pickle

def Predict_Next_Words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break

    print(predicted_word)
    return predicted_word

try:
    # Load the model and tokenizer
    model = load_model('next_words.h5')
    tokenizer = pickle.load(open('token.pkl', 'rb'))

    while True:
        text = input("Enter your line: ")

        if text == "0":
            print("Execution completed.....")
            break

        else:
            try:
                text = text.split(" ")
                text = text[-3:]
                print(text)

                Predict_Next_Words(model, tokenizer, text)

            except Exception as e:
                print("Error occurred during prediction: ", e)
                continue

except Exception as e:
    print("Loading model/tokenizer failed:", e)

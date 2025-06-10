import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

print("Loading prepared data...")
with np.load("prepared_data.npz", allow_pickle=True) as data:
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    modulation_types = data['modulation_types']

X_train = X_train.reshape(X_train.shape[0], 2, 128, 1)
X_test = X_test.reshape(X_test.shape[0], 2, 128, 1)
num_classes = len(modulation_types)

print("Data loaded and reshaped for training.")

print("Building the CNN model...")
model = Sequential()

model.add(Conv2D(64, (1, 3), activation='relu', input_shape=(2, 128, 1)))
model.add(MaxPooling2D((1, 2)))

model.add(Conv2D(32, (2, 3), activation='relu'))
model.add(MaxPooling2D((1, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.summary()

print("\nCompiling the model...")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Training the model... (This may take some time)")
history = model.fit(X_train, Y_train,
                    epochs=20,
                    batch_size=256,
                    validation_data=(X_test, Y_test))

print("\nTraining complete. Saving model to 'signal_classifier.h5'...")
model.save('signal_classifier.h5')
print("Model saved successfully.")

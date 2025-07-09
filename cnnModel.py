import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard 
import datetime
from sklearn.metrics import classification_report, confusion_matrix



def load_brain_tumor_data(data_dir, img_size=(224, 224), test_size=0.2):
    """
    Load brain tumor data and return train/test splits
    
    Args:
        data_dir: Path to main directory containing 'yes' and 'no' folders
        img_size: Target image size (height, width)
        test_size: Fraction of data to use for testing
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    
    images = []
    labels = []
    
    
    yes_folder = os.path.join(data_dir, 'yes')
    for filename in os.listdir(yes_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(yes_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(1)  
    
    
    no_folder = os.path.join(data_dir, 'no')
    for filename in os.listdir(no_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(no_folder, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(0)  
    
    
    X = np.array(images)
    y = np.array(labels)
    
    
    X = X.astype('float32') / 255.0
    
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_brain_tumor_data("path to your saved model, if required rerun it to cretae your own model")
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(y_train[1:])

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) 


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

log_dir = "logs/fit" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorflow_callback = TensorBoard(
    log_dir,
    histogram_freq=1,
    write_graph = True,
    write_images = True
    )

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
    
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    callbacks=[tensorflow_callback, early_stopping_callback],
    batch_size=64
)


print("Training completed, Saving the model")
model.save('brain_tumor_model.h5')  

print("Model saved successfully!")


test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int).flatten()

## additional for evalaution purpose , optional use if required


print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred_binary, target_names=['No Tumor', 'Tumor']))

print("\n=== CONFUSION MATRIX ===")
cm = confusion_matrix(y_test, y_pred_binary)
print(cm)
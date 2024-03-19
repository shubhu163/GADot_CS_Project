import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Conv2D, Dropout, GlobalMaxPooling2D, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
import numpy as np
import os

# Seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants
CSV_FILE_PATH = '/nfs/hpc/share/joshishu/CS_Project/GADot_CS_Project/data/DATA/CICD2018_train.csv'
MODEL_SAVE_PATH = '/nfs/hpc/share/joshishu/CS_Project/GADot_CS_Project/Output_Models/Lucid_50.h5'
OUTPUT_FOLDER = "/nfs/hpc/share/joshishu/CS_Project/GADot_CS_Project/Output_Models"

# Ensure output directory exists
if not os.path.isdir(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Function to load and preprocess the CSV data
def load_and_preprocess_csv(filepath):
    df = pd.read_csv(filepath)
    X = df.drop(' Label', axis=1)
    y = df[' Label'].apply(lambda x: 1 if x != 'Benign' else 0)
    # y = df[' Label']
    scaler = MinMaxScaler()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.median(), inplace=True)
    X_scaled = scaler.fit_transform(X)

    # Split data into training and testing sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=SEED, stratify=y_train_val)  # 0.25 x 0.8 = 0.2

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1, 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1, 1)
    X_val = X_val.reshape(X_test.shape[0], X_test.shape[1], 1, 1)


    return X_train, X_val, X_test, y_train, y_val, y_test


# Model definition
def Conv2DModel(input_shape, kernels=64, kernel_size=(3, 1), learning_rate=0.01, dropout=None):
    model = Sequential(name="CNN_Model")
    model.add(Conv2D(kernels, kernel_size, strides=(1, 1), input_shape=input_shape, name='conv0'))
    if dropout is not None and isinstance(dropout, float):
        model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid', name='fc1'))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def report_results(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (tp + fn)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")

# Main function
def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_csv(CSV_FILE_PATH)
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

    # Grid search for hyperparameter tuning
    model = KerasClassifier(build_fn=Conv2DModel, input_shape=(X_train.shape[1], 1, 1), verbose=0)
    param_grid = {
        'learning_rate': [0.1, 0.01,0.001],
        'kernels': [32, 64],
        'dropout': [0.2, 0.3],
        'epochs': [20],  # You can add more epochs here
        'batch_size': [64,128]  # You can add more batch sizes here
    }

    # Grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3, refit=True)
    grid_result = grid.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[es])
    # Best: 0.927691 using {'batch_size': 128, 'dropout': 0.2, 'epochs': 50, 'kernels': 64, 'learning_rate': 0.01}


    # # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # # best_params = grid_result.best_params_
    # best_model = Conv2DModel(input_shape=(X_train.shape[1], 1, 1),
    #                          kernels=64,
    #                          learning_rate=0.001,
    #                          dropout=0.2)
    # history = best_model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_val, y_val),
    #                          callbacks=[es],
    #                          verbose=1)
    # print("Training Accuracy: ", history.history['accuracy'][-1])
    # print("Training Loss: ", history.history['loss'][-1])

    # best_model.save(MODEL_SAVE_PATH)

    # # Evaluate the best model
    # y_pred = best_model.predict(X_test)
    # y_pred = (y_pred > 0.5).astype(int)
    # print("Test Set Classification Report:")
    # print(classification_report(y_test, y_pred))
    # print("Test Set Confusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))


    # # Print test set metrics
    # report_results(y_test, y_pred)

if __name__ == '__main__':
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    main()

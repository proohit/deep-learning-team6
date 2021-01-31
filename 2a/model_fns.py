import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np

def build_model(input_shape):    
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    return model

def train_model(model, x, y, x_val, y_val):
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(x, y, epochs=10,validation_data=(x_val, y_val))
    return history
    
def evaluate_model(history, model, test_data, test_labels):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(history.history['loss'], label="Training Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

    test_loss, test_acc = model.evaluate(test_data,  test_labels, verbose=2)
    print(f"Test Accuracy: {test_acc}, Test Loss: {test_loss}")
    return test_acc
    
def run_kfold(x,y):
    num_folds = 5
    acc_per_fold = []
    
    kfold = KFold(n_splits=num_folds, shuffle=True)

    fold_no = 1
    for train, test in kfold.split(x, y):

        model = build_model()
        model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(x[train], y[train],epochs=3)

        scores = model.evaluate(x[test], y[test], verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)

        fold_no = fold_no + 1
    print(f'Total Score: {np.array(acc_per_fold).mean()}')
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def create_data_generators(data_dir_train, data_dir_val, data_dir_test, img_size=256, batch_size=32):
    # Data Preprocessing
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        data_dir_train,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['Adenocarcinoma', 'Squamous cell carcinoma', 'Small cell carcinoma', 'Normal']
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir_val,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['Adenocarcinoma', 'Squamous cell carcinoma', 'Small cell carcinoma', 'Normal']
    )

    test_generator = test_datagen.flow_from_directory(
        data_dir_test,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=['Adenocarcinoma', 'Squamous cell carcinoma', 'Small cell carcinoma', 'Normal']
    )

    return train_generator, val_generator, test_generator

def build_cnn_model(img_size=256, num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_and_evaluate(data_dir_train, data_dir_val, data_dir_test, img_size=256, batch_size=32, epochs=30):
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators(
        data_dir_train, data_dir_val, data_dir_test, img_size, batch_size
    )

    # Build and compile model
    model = build_cnn_model(img_size, num_classes=len(train_generator.class_indices))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Define callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # checkpoint = callbacks.ModelCheckpoint(
    #     'best_model_cnn.h5',
    #     monitor='val_accuracy',
    #     save_best_only=True,
    #     mode='max'
    # )

    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        # callbacks=[early_stopping, checkpoint]
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"\nTest loss: {test_loss:.4f}")

    # Make predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Print classification report
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    return history, model

def plot_training_history(history, filename='training_history_cnn.png'):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'bo-', label='Training accuracy')
    plt.plot(history.history['val_accuracy'], 'r^-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'bo-', label='Training loss')
    plt.plot(history.history['val_loss'], 'r^-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.clf()

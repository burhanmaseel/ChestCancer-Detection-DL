import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def create_data_generators(data_dir_train, data_dir_val, data_dir_test, img_size=224, batch_size=32):
    # Data Preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input
    )

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        data_dir_train,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir_val,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        data_dir_test,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, val_generator, test_generator

def build_transfer_model(img_size=224, num_classes=4):
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(img_size, img_size, 3)
    )

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Create new model on top
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model

def train_and_evaluate(data_dir_train, data_dir_val, data_dir_test, img_size=224, batch_size=32, 
                      initial_epochs=15, fine_tune_epochs=15):
    # Create data generators
    train_generator, val_generator, test_generator = create_data_generators(
        data_dir_train, data_dir_val, data_dir_test, img_size, batch_size
    )

    # Build and compile model
    model = build_transfer_model(img_size, num_classes=len(train_generator.class_indices))
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


    # Initial training with frozen layers
    print("Training with frozen layers")
    history_frozen = model.fit(
        train_generator,
        epochs=initial_epochs,
        validation_data=val_generator,
        callbacks=[early_stopping]
    )

    # Fine-tuning
    print("\nFine-tuning the model")
    for layer in model.layers[0].layers[-4:]:  # Unfreeze the last 4 convolutional layers
        layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_finetune = model.fit(
        train_generator,
        epochs=fine_tune_epochs,
        validation_data=val_generator,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Make predictions
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)

    # Print classification report
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_labels))

    return history_frozen, history_finetune, model

def plot_training_history(history_frozen, history_finetune, filename='training_history_vgg16.png'):
    plt.figure(figsize=(15, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)

    # Plot frozen training
    epochs_frozen = range(1, len(history_frozen.history['accuracy']) + 1)
    plt.plot(epochs_frozen, history_frozen.history['accuracy'], 'bo-', label='Training acc (frozen)')
    plt.plot(epochs_frozen, history_frozen.history['val_accuracy'], 'b^-', label='Validation acc (frozen)')

    # Plot fine-tuning training
    start_epoch = len(history_frozen.history['accuracy'])
    epochs_finetune = range(start_epoch + 1, start_epoch + len(history_finetune.history['accuracy']) + 1)
    plt.plot(epochs_finetune, history_finetune.history['accuracy'], 'ro-', label='Training acc (fine-tuning)')
    plt.plot(epochs_finetune, history_finetune.history['val_accuracy'], 'r^-', label='Validation acc (fine-tuning)')

    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)

    # Plot frozen training
    plt.plot(epochs_frozen, history_frozen.history['loss'], 'bo-', label='Training loss (frozen)')
    plt.plot(epochs_frozen, history_frozen.history['val_loss'], 'b^-', label='Validation loss (frozen)')

    # Plot fine-tuning training
    plt.plot(epochs_finetune, history_finetune.history['loss'], 'ro-', label='Training loss (fine-tuning)')
    plt.plot(epochs_finetune, history_finetune.history['val_loss'], 'r^-', label='Validation loss (fine-tuning)')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    plt.clf()

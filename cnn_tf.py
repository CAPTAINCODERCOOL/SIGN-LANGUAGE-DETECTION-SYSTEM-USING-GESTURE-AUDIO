import tensorflow as tf
import numpy as np
import pickle, os, cv2

def get_image_size():
    img = cv2.imread('gestures/0/100.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(os.listdir('gestures/'))

image_x, image_y = get_image_size()

def cnn_model():
    num_of_classes = get_num_of_classes()  # Get the number of classes here
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(image_x, image_y, 1)),
        tf.keras.layers.Conv2D(16, (2, 2), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_of_classes, activation='softmax')
    ])
    return model

def main():
    # Load your data and preprocess it

    # Get the number of classes by calling the function
    num_of_classes = get_num_of_classes()

    # Create an instance of your model
    model = cnn_model()

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Load your data and preprocess it
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    with open("test_images", "rb") as f:
        test_images = np.array(pickle.load(f))
    with open("test_labels", "rb") as f:
        test_labels = np.array(pickle.load(f), dtype=np.int32)

    train_images = np.reshape(train_images, (train_images.shape[0], image_x, image_y, 1))

    # Call get_num_of_classes to get the number of classes
    num_of_classes = get_num_of_classes()

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_of_classes)

    # Train the model
    model.fit(train_images, train_labels, epochs=80, batch_size=500, validation_split=0.1)

    # Evaluate the model on your test data
    test_images = np.reshape(test_images, (test_images.shape[0], image_x, image_y, 1))
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_of_classes)
    test_results = model.evaluate(test_images, test_labels)
    print("Test loss:", test_results[0])
    print("Test accuracy:", test_results[1])

if __name__ == "__main__":
    main()

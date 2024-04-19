# Python-TensorFlow

This repository provides examples and tutorials for TensorFlow, an open-source machine learning library developed by Google. TensorFlow offers multiple levels of abstraction so you can choose the right one for your needs. Build and train models easily using high-level Keras API, or control every little detail in the lower-level APIs.

## Prerequisites

- Python 3.6 or higher
- pip package manager

## Installation

To set up TensorFlow on your system, follow these steps:

1. It's recommended to use a virtual environment to isolate package installations:

   ```bash
   python -m venv tf-env
   source tf-env/bin/activate  # On Windows use `tf-env\Scripts\activate`
   ```

2. Install TensorFlow:

   ```bash
   pip install tensorflow
   ```

   For GPU support (optional):

   ```bash
   pip install tensorflow-gpu
   ```

## Example - Basic Neural Network

This example demonstrates a basic neural network that classifies the fashion MNIST dataset using the Keras API.

### `keras_example.py`

```python
import tensorflow as tf
from tensorflow import keras

# Load dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Normalize the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
```

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

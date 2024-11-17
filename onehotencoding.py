
from sklearn.preprocessing import OneHotEncoder
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshape y_train to be a 2D array (required by OneHotEncoder)
y_train_reshaped = y_train.reshape(-1, 1)

# Initialize the OneHotEncoder
encoder = OneHotEncoder()

# Fit and transform the labels
one_hot_encoded = encoder.fit_transform(y_train_reshaped)
y_train = one_hot_encoded.toarray()

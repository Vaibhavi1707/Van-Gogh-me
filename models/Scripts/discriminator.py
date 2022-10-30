from tensorflow import keras
from tensorflow.keras import layers

IMG_DIMS = (128, 128)

def make_discriminator():
    model = keras.Sequential()
    # Input Layer
    model.add(layers.Input([*IMG_DIMS, 3]))
    
    # Convolution Net
    model.add(layers.Conv2D(64, (4, 4), strides = 2))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(128, (4, 4), strides = 2))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(256, (4, 4), strides = 2))
    model.add(layers.Conv2D(512, (4, 4), strides = 2))
    model.add(layers.LeakyReLU())
    
    model.add(layers.Flatten())
    
    # Connected Net
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(64))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(2))
    model.add(layers.Softmax())
    
    return model
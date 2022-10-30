from tensorflow import keras
from tensorflow.keras import layers

IMG_DIMS = (128, 128)

# Generator Code
## Step 1: Down Sample the image 
## Step 2: Upsample the image using Transpose Convolution

def make_generator():
    ## Input Layer
    input_layer = layers.Input([*IMG_DIMS, 3])
    x = input_layer
    
    ## Downsampler
    skips = []
    x = layers.Conv2D(filters = 64, kernel_size = (4, 4), 
                      strides = 2, padding = 'same', activation = 'relu')(x) # 64 * 64 * 64
    skips.append(x)
    x = layers.Conv2D(filters = 128, kernel_size = (4, 4), 
                      strides = 2, padding = 'same', activation = 'relu')(x) # 32 * 32 * 128
    skips.append(x)
    x = layers.Conv2D(filters = 256, kernel_size = (4, 4), 
                      strides = 2, padding = 'same', activation = 'relu')(x) # 16 * 16 * 256
    skips.append(x)
    x = layers.Conv2D(filters = 512, kernel_size = (4, 4), 
                      strides = 2, padding = 'same', activation = 'relu')(x) # 8 * 8 * 512
    skips.append(x)
    x = layers.Conv2D(filters = 512, kernel_size = (4, 4), 
                      strides = 2, padding = 'same', activation = 'relu')(x) # 4 * 4 * 512
    skips.append(x)
    x = layers.Conv2D(filters = 512, kernel_size = (4, 4), 
                      strides = 2, padding = 'same', activation = 'relu')(x) # 2 * 2 * 512
    skips.append(x)
    x = layers.Conv2D(filters = 512, kernel_size = (4, 4), 
                      strides = 2, padding = 'same', activation = 'relu')(x) # 1 * 1 * 512
    
    ## Upsampling
    x = layers.Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = 2, 
                               padding = 'same', activation = 'relu')(x) # 2 * 2
    x = layers.Concatenate()([x, skips[-1]])
    x = layers.Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = 2, 
                               padding = 'same', activation = 'relu')(x) # 4 * 4
    x = layers.Concatenate()([x, skips[-2]])
    x = layers.Conv2DTranspose(filters = 512, kernel_size = (4, 4), strides = 2, 
                               padding = 'same', activation = 'relu')(x) # 8 * 8
    x = layers.Concatenate()([x, skips[-3]])
    x = layers.Conv2DTranspose(filters = 256, kernel_size = (4, 4), strides = 2, 
                               padding = 'same', activation = 'relu')(x) # 16 * 16
    x = layers.Concatenate()([x, skips[-4]])
    x = layers.Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = 2, 
                               padding = 'same', activation = 'relu')(x) # 32 * 32
    x = layers.Concatenate()([x, skips[-5]])
    x = layers.Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = 2, 
                               padding = 'same', activation = 'relu')(x) # 64 * 64
    x = layers.Concatenate()([x, skips[-6]])
    x = layers.Conv2DTranspose(filters = 3, kernel_size = (4, 4), strides = 2, 
                               padding = 'same', activation = 'relu')(x) # 128 * 128
    
    return keras.Model(inputs = input_layer, outputs = x)
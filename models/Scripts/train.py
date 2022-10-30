import tensorflow as tf

from preprocessing.create_dataset import get_zipped_data
from CycleGAN import CycleGAN

model = CycleGAN(lambda_cycle = 10)
model.compile()
model.fit(get_zipped_data(), epochs = 1)
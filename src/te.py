import tensorflow  as tf

# print(tf.__version__)

# print("the ")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# print num of cpu's available
print("Num CPU's Available: ", len(tf.config.experimental.list_physical_devices('CPU')))
import tensorflow as tf


def set_gpu_memory(value, output: bool = False):
    """
    Sets gpu memory limit depending on available ram.
    Args:
        value: int
            Total ram available.
        output: bool, Optional
            Whether to print the output or not. (Default is False.)
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=value * 0.98)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            if output:
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

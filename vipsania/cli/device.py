import sys

NO_GPU_WARNING = """\
!! No GPU found. Vipsania will run on the CPU, which is far slower and not
!! practical for annotating or training on whole genomes.
!!
!! If this machine does have a GPU, TensorFlow was unable to load its CUDA
!! libraries. Run the command again with TF_CPP_MIN_LOG_LEVEL=0 to see which
!! library failed, and see docs/troubleshooting.md in the Vipsania repository.
"""


def report_devices() -> bool:
    """Print the devices TensorFlow is going to use and warn if it did
    not find a GPU. Returns whether a GPU is available.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print(NO_GPU_WARNING, file=sys.stderr, flush=True)
        return False

    names = []
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        names.append(details.get("device_name") or gpu.name.rsplit("/", 1)[-1])
    print(
        f"Using {len(gpus)} GPU{'s' if len(gpus) > 1 else ''}: "
        f"{', '.join(names)}",
        flush=True,
    )
    return True

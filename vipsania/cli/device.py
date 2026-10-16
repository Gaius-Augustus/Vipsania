import ctypes
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


def free_gpu_memory(gpu_index: int = 0) -> float | None:
    try:
        cuda = ctypes.CDLL("libcuda.so.1")
    except OSError: return None

    device = ctypes.c_int()
    context = ctypes.c_void_p()
    if (
        cuda.cuInit(0) != 0
        or cuda.cuDeviceGet(ctypes.byref(device), gpu_index) != 0
        or cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) != 0
    ): return None

    free_bytes = ctypes.c_size_t()
    total_bytes = ctypes.c_size_t()
    pushed = cuda.cuCtxPushCurrent_v2(context) == 0
    queried = pushed and cuda.cuMemGetInfo_v2(
        ctypes.byref(free_bytes), ctypes.byref(total_bytes),
    ) == 0
    if pushed: cuda.cuCtxPopCurrent_v2(ctypes.byref(context))
    cuda.cuDevicePrimaryCtxRelease_v2(device)
    return free_bytes.value / 1024**3 if queried else None


def estimate_max_batch_size(
    context_length: int,
    model_size_params: int,
    available_memory_gb: float | None,
    safety_factor: float = 0.9,
    finetune: bool = False,
) -> int:
    if available_memory_gb is None:
        raise RuntimeError(
            "Unable to determine the free GPU memory from the CUDA driver. "
            "Specify the batch size manually using "
            f"'{'--finetune_B' if finetune else '-B'}'"
        )

    A = 3.535714285714286e-6 / 14_384_704
    if not finetune:
        # Calibration for 80GB GPU, 25M model, 200k context:
        #   -> batch size: 32
        # Calibration for 24GB GPU, 10M model, 200k context:
        #   -> batch size: 14
        B = 1.125e-5 - 25_047_166 * A
    else:
        # Calibration for 80GB GPU, 25M model, 200k context:
        #   -> batch size: 4
        # Calibration for 90GB GPU, 25M model, 200k context:
        #   -> batch size: 8
        # Calibration for 24GB GPU, 10M model, 200k context:
        #   -> batch size: 2
        B = 5.0625e-5 - 25_047_166 * A

    usable = safety_factor * available_memory_gb
    cost_per_sample = context_length * (A * model_size_params + B)

    raw_max_batch = usable / cost_per_sample + 1e-9
    if finetune:
        for divisor in (64, 32, 16, 8, 4, 2, 1):
            if divisor <= raw_max_batch: return divisor
    return max(1, int(raw_max_batch))

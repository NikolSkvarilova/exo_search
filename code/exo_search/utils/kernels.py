import gpflow

# Predefined kernels for the command line commands
kernels = {
    "se+periodic": gpflow.kernels.SquaredExponential()
    + gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
    "se": gpflow.kernels.SquaredExponential(),
    "matern12": gpflow.kernels.Matern12(),
    "matern32": gpflow.kernels.Matern32(),
    "matern52": gpflow.kernels.Matern52(),
    "periodic": gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential()),
}


def get_kernel(kernel_name: str) -> gpflow.kernels.Kernel:
    """Get the kernel from kernel name.

    Args:
        kernel_name (str): kernel name.

    Raises:
        SystemExit: when the kernel was not found.

    Returns:
        gpflow.kernels.Kernel: kernel.
    """
    if kernel_name in list(kernels.keys()):
        return gpflow.utilities.deepcopy(kernels[kernel_name])

    raise SystemExit(f"Unknown kernel '{kernel_name}'. Exiting.")

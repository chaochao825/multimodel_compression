from src.methods.bcm import BCMLinear


class BCALinear(BCMLinear):
    """
    Block-Circulant Adapter baseline.

    Under the current implementation, BCA and the structure-only BCM adapter
    share the same block-circulant parameterization, but BCA gets its own class
    and method name so it can be tracked as a distinct baseline in PEFT runs.
    """


from src.modules.block_circulant_linear import BlockCirculantLinear


class FFTBlockCirculantLinear(BlockCirculantLinear):
    def __init__(self, *args, **kwargs):
        kwargs["use_fft"] = True
        super().__init__(*args, **kwargs)

import numpy as np

class DropoutLayer:
    def __init__(self, p: float):
        """Initialize the dropout layer.
        
        Attributes to set:
            self.p: the dropout rate
            self.mask: stores the dropout mask (initially None)
        """
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout rate p must be in [0, 1)")
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """Forward pass of the dropout layer.
        
        Generate a new mask on each training forward pass and store it in self.mask.
        """
        if not training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        # 二值掩码：随机值小于 keep_prob 的位置保留（概率为 keep_prob）
        self.mask = (np.random.rand(*x.shape) < keep_prob).astype(x.dtype)
        # Inverted dropout：训练时按 1/keep_prob 放大，推理时无需任何操作
        return x * self.mask / keep_prob

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass of the dropout layer.
        
        Use the stored self.mask from the most recent forward pass.
        """
        if self.mask is None:
            raise ValueError("backward called before a training forward pass")
        return grad * self.mask / (1.0 - self.p)
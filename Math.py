import numpy as np

class Vec2:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y

    # v * scalar (左乘)
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vec2(self.x * other, self.y * other)
        elif isinstance(other, np.array(2, 2)):
            return Vec2(other[0, 0] * self.x + other[0, 1] * self.y, other[1, 0] * self.x + other[1, 1] * self.y)
        else:
            return NotImplemented

    # scalar * v (右乘)
    def __rmul__(self, other):
        # 对于乘法，右乘通常直接复用左乘的逻辑
        return self.__mul__(other)
        
    #  打印向量
    def __repr__(self):
            return f"Vec2({self.x}, {self.y})"
    
    def Abs(self):
        return Vec2(abs(self.x), abs(self.y))

    # --- 矩阵乘法: M @ v (矩阵左乘列向量) ---
    def __rmatmul__(self, matrix: np.ndarray) -> 'Vec2':
        """
        支持: matrix @ vector (标准列向量变换)
        例如: Rot @ v  -> 将向量v旋转
        """
        if not isinstance(matrix, np.ndarray):
            return NotImplemented
        if matrix.shape != (2, 2):
            raise ValueError(f"Matrix must be (2,2), got {matrix.shape}")
        # 列向量变换: [x'] = [m00 m01] [x]
        #              [y']   [m10 m11] [y]
        x_new = matrix[0, 0] * self.x + matrix[0, 1] * self.y
        y_new = matrix[1, 0] * self.x + matrix[1, 1] * self.y
        return Vec2(x_new, y_new)
    
    # --- 矩阵乘法: v @ M (行向量右乘矩阵) ---
    def __matmul__(self, matrix: np.ndarray) -> 'Vec2':
        """
        支持: vector @ matrix (行向量变换，较少使用)
        例如: v @ Rot  -> 相当于 (Rot^T @ v) 的转置
        """
        if not isinstance(matrix, np.ndarray):
            return NotImplemented
        if matrix.shape != (2, 2):
            raise ValueError(f"Matrix must be (2,2), got {matrix.shape}")
        # 行向量变换: [x' y'] = [x y] @ [m00 m01]
        #                                [m10 m11]
        x_new = self.x * matrix[0, 0] + self.y * matrix[1, 0]
        y_new = self.x * matrix[0, 1] + self.y * matrix[1, 1]
        return Vec2(x_new, y_new)

def FromAngleToMatrix(angle: float):
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
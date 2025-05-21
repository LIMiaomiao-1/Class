import numpy as np
# 1. 构造设计矩阵X (5x2: 第一列为浓度c，第二列为全1)
c = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
X = np.column_stack([c, np.ones_like(c)])  # 设计矩阵 (5x2)
a = np.array([0.14, 0.31, 0.46, 0.60, 0.73])  # 观测值 (5x1)

# 2. 正规方程推导 (β = (XᵀX)⁻¹Xᵀa)
X_T = X.T                    # X的转置 (2x5)
X_T_X = np.dot(X_T, X)       # XᵀX (2x2矩阵)
X_T_X_inv = np.linalg.inv(X_T_X)  # (XᵀX)⁻¹ (2x2矩阵)
X_T_y = np.dot(X_T, a)       # Xᵀy (2x1向量)
beta = np.dot(X_T_X_inv, X_T_y)  # β = [β1, β0]

# 3. 计算浓度cu (当au=0.42时，解方程 a = β1*cu + β0)
au = 0.42
cu = (au - beta[1]) / beta[0]

print("设计矩阵 X (含截距项):\n", X)
print("\nXᵀX:\n", X_T_X)
print("\n(XᵀX)⁻¹:\n", X_T_X_inv)
print("\nXᵀy:\n", X_T_y)
print("\n回归系数 β:\n", beta)
print(f"\ncu (au=0.42时): {cu:.3f}")
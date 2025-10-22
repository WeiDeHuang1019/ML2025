# **MLP 前向傳遞 (Forward Propagation)** 的公式流程  
目前的結構是：

* 輸入層：特徵向量 $\mathbf{x} = [x_1, x_2, ..., x_N]^T$
* 隱藏層：(K) 個神經元，每個有激勵函數 $f(\cdot)$
* 輸出層：4 個神經元（對應 4 個類別），最後會接 loss function

---

### 1. 輸入層 → 隱藏層

對第 (j) 個隱藏層神經元：

$$z_j = \sum_{i=1}^{N} w_{ji}^{(1)} x_i + b_j^{(1)}$$

$h_j = f(z_j)$

* $w_{ji}^{(1)}$：輸入層到隱藏層的權重
* $b_j^{(1)}$：隱藏層神經元的 bias
* $f(\cdot)$：激勵函數 (sigmoid, ReLU, tanh...)
* $h_j$：隱藏層的輸出

矩陣化後：

$\mathbf{h} = f\big(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\big)$

其中：
$\mathbf{h} \in \mathbb{R}^K ; W^{(1)} \in \mathbb{R}^{K \times N}$

---

### 2. 隱藏層 → 輸出層

對第 (k) 個輸出神經元：

$$o_k = \sum_{j=1}^{K} w_{kj}^{(2)} h_j + b_k^{(2)}$$

如果輸出層要做分類（4 個類別），通常會接 **softmax**：

$$y_k = \frac{e^{o_k}}{\sum_{m=1}^{4} e^{o_m}}, \quad k = 1,2,3,4$$

矩陣化後：

$\mathbf{o} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}\$

$$\mathbf{y} = \text{softmax}(\mathbf{o})$$

其中：
$\mathbf{y} \in \mathbb{R}^4$ 為預測的類別機率分佈

---

總結 Forward Pass 的兩步：

1. **輸入層到隱藏層：**
   $\mathbf{h} = f\big(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\big)$

2. **隱藏層到輸出層：**
   $\mathbf{o} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}, \quad \mathbf{y} = \text{softmax}(\mathbf{o})$

---


# MLP 反向傳播 (Back Propagation)** 的公式流程 

考慮一個兩層的多層感知器 (MLP)：

輸入層 → 隱藏層 → 輸出層  
權重與偏置如下：

隱藏層：  <br>
$z_j^{(1)} = \sum_i W_{ji}^{(1)} x_i + b_j^{(1)}$  
$h_j = f(z_j^{(1)})$

輸出層： <br>
$z_k^{(2)} = \sum_j W_{kj}^{(2)} h_j + b_k^{(2)}$  
$y_k = g(z_k^{(2)})$

---

### **1. 損失函數 (Mean Squared Error)**

$$ L = \frac{1}{2} \sum_{k=1}^{K} (y_k - t_k)^2 $$

---

### **2. 輸出層誤差項 δ**

對每個輸出層神經元 $k$：

$$ \delta_k^{(2)} = \frac{\partial L}{\partial z_k^{(2)}} = (y_k - t_k) \, g'(z_k^{(2)}) $$

> 若輸出層為線性輸出（$g'(z) = 1$），則：  
> $\delta_k^{(2)} = y_k - t_k$

---

### **3. 輸出層權重與偏置的梯度**

對每一條連線 $W_{kj}^{(2)}$：

$$ \frac{\partial L}{\partial W_{kj}^{(2)}} = \delta_k^{(2)} \, h_j $$

對每一個輸出層偏置：

$$ \frac{\partial L}{\partial b_k^{(2)}} = \delta_k^{(2)} $$

---

### **4. 隱藏層的誤差項 δ**

對每個隱藏層神經元 $j$：

$$ \delta_j^{(1)} = f'(z_j^{(1)}) \sum_{k} W_{kj}^{(2)} \, \delta_k^{(2)} $$

---

### **5. 隱藏層權重與偏置的梯度**

對每一條連線 $W_{ji}^{(1)}$：

$$ \frac{\partial L}{\partial W_{ji}^{(1)}} = \delta_j^{(1)} \, x_i $$

對每一個隱藏層偏置：

$$ \frac{\partial L}{\partial b_j^{(1)}} = \delta_j^{(1)} $$

---

### **6. 參數更新（以梯度下降為例）**

設學習率為 $\eta$：

$$ W_{kj}^{(2)} \leftarrow W_{kj}^{(2)} - \eta \, \frac{\partial L}{\partial W_{kj}^{(2)}} $$  
$$ b_k^{(2)} \leftarrow b_k^{(2)} - \eta \, \frac{\partial L}{\partial b_k^{(2)}} $$  
$$ W_{ji}^{(1)} \leftarrow W_{ji}^{(1)} - \eta \, \frac{\partial L}{\partial W_{ji}^{(1)}} $$  
$$ b_j^{(1)} \leftarrow b_j^{(1)} - \eta \, \frac{\partial L}{\partial b_j^{(1)}} $$

---

### **7. 總結 Backpropagation（逐節點形式）**

1. **Forward pass**  
   $$z_j^{(1)} = \sum_i W_{ji}^{(1)} x_i + b_j^{(1)}, \quad h_j = f(z_j^{(1)})$$  
   $$z_k^{(2)} = \sum_j W_{kj}^{(2)} h_j + b_k^{(2)}, \quad y_k = g(z_k^{(2)})$$

2. **Compute loss**  
   $$L = \frac{1}{2} \sum_k (y_k - t_k)^2$$

3. **Backward pass**  
   $$\delta_k^{(2)} = (y_k - t_k) g'(z_k^{(2)})$$  
   $$\frac{\partial L}{\partial W_{kj}^{(2)}} = \delta_k^{(2)} h_j, \quad \frac{\partial L}{\partial b_k^{(2)}} = \delta_k^{(2)}$$  
   $$\delta_j^{(1)} = f'(z_j^{(1)}) \sum_k W_{kj}^{(2)} \delta_k^{(2)}$$  
   $$\frac{\partial L}{\partial W_{ji}^{(1)}} = \delta_j^{(1)} x_i, \quad \frac{\partial L}{\partial b_j^{(1)}} = \delta_j^{(1)}$$

---



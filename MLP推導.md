# **MLP 前向傳遞 (Forward Propagation)** 的公式流程  
目前的結構是：

* 輸入層：特徵向量 $\mathbf{x} = [x_1, x_2, ..., x_N]^T$
* 隱藏層：(K) 個神經元，每個有激勵函數 $f(\cdot)$
* 輸出層：4 個神經元（對應 4 個類別），最後會接 loss function

---

### 1. 輸入層 → 隱藏層

對第 (j) 個隱藏層神經元：

$$u_j = \sum_{i=1}^{N} w_{ji}^{(1)} x_i + b_j^{(1)}$$

$$h_j = f(u_j)$$

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

$$z_k = \sum_{j=1}^{K} w_{kj}^{(2)} h_j + b_k^{(2)}$$

如果輸出層要做分類（4 個類別），通常會接 **softmax**：

$$y_k = \frac{e^{z_k}}{\sum_{m=1}^{4} e^{z_m}}, \quad k = 1,2,3,4$$

矩陣化後：

$\mathbf{z} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}\$

$$\mathbf{y} = \text{softmax}(\mathbf{z})$$

其中：
$\mathbf{y} \in \mathbb{R}^4$ 為預測的類別機率分佈

---

總結 Forward Pass 的兩步：

1. **輸入層到隱藏層：**
   $\mathbf{h} = f\big(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\big)$

2. **隱藏層到輸出層：**
   $\mathbf{z} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}, \quad \mathbf{y} = \text{softmax}(\mathbf{z})$

---


# MLP 反向傳播 (Back Propagation) 的公式流程 

考慮一個兩層的多層感知器 (MLP)：

輸入層 → 隱藏層 → 輸出層  
權重與偏置如下：

隱藏層：  <br>
$u_j^{(1)} = \sum_i W_{ji}^{(1)} x_i + b_j^{(1)}$  
$h_j = f(u_j^{(1)})$

輸出層： <br>
$z_k^{(2)} = \sum_j W_{kj}^{(2)} h_j + b_k^{(2)}$  
$y_k = g(z_k^{(2)})$

---

### **1. 損失函數 (Mean Squared Error)**

$$ L = \frac{1}{2} \sum_{k=1}^{K} (y_k - t_k)^2 $$
我們要求：
$$
\frac{\partial L}{\partial w_{kj}^{(2)}},    \frac{\partial L}{\partial w_{kj}^{(1)}}
$$

---
### **2. 輸出層 → 隱藏層**
#### 套用鏈式法則 (Chain Rule)

先觀察：

$$
L \to y_k \to z_k^{(2)} \to w_{kj}^{(2)}
$$

因此：

$$
\frac{\partial L}{\partial w_{kj}^{(2)}} 
= \frac{\partial L}{\partial y_k} 
  \cdot \frac{\partial y_k}{\partial z_k^{(2)}}
  \cdot \frac{\partial z_k^{(2)}}{\partial w_{kj}^{(2)}}
$$

#### 各部分分開計算

- (a) 損失對輸出：

$$
\frac{\partial L}{\partial y_k} = (y_k - t_k)
$$

- (b) 輸出對輸入：

$$
\frac{\partial y_k}{\partial z_k^{(2)}} = g'(z_k^{(2)})
$$

- (c) 輸入對權重：

$$
z_k^{(2)} = \sum_{j=1}^{K} w_{kj}^{(2)} h_j + b_k^{(2)} 
\Rightarrow \frac{\partial z_k^{(2)}}{\partial w_{kj}^{(2)}} = h_j
$$


#### 三者相乘得到結果

$$
\boxed{
\frac{\partial L}{\partial w_{kj}^{(2)}} 
= (y_k - t_k)\cdot g'(z_k^{(2)}) \cdot h_j
}
$$

> 若輸出層為線性輸出（ $g'(z) = 1$ ），則：  
> $\boxed{\frac{\partial L}{\partial w_{kj}^{(2)}} = (y_k - t_k)\cdot h_j}$

---
### **3. 隱藏層 → 輸入層**

沿著計算圖的依賴關係：

$$
L \to y_k \to z_k^{(2)} \to h_j \to z_j^{(1)} \to w_{ji}^{(1)}
$$

對任意輸出節點 \(k\)：

$$
\frac{\partial L}{\partial w_{ji}^{(1)}}
= \sum_{k=1}^{4}
\underbrace{\frac{\partial L}{\partial y_k}}_{(a)}
\cdot
\underbrace{\frac{\partial y_k}{\partial z_k^{(2)}}}_{(b)}
\cdot
\underbrace{\frac{\partial z_k^{(2)}}{\partial h_j}}_{(c)}
\cdot
\underbrace{\frac{\partial h_j}{\partial z_j^{(1)}}}_{(d)}
\cdot
\underbrace{\frac{\partial z_j^{(1)}}{\partial w_{ji}^{(1)}}}_{(e)}
$$

分別計算五個局部梯度：

- (a) 損失對輸出：
  
$$
\frac{\partial L}{\partial y_k} = (y_k - t_k)
$$
- (b) 輸出對其輸入：
  
$$
\frac{\partial y_k}{\partial z_k^{(2)}} = g'\big(z_k^{(2)}\big)
$$
- (c) 輸出層線性組合對 \(h_j\)：

$$
\frac{\partial z_k^{(2)}}{\partial h_j} = w_{kj}^{(2)}
$$
- (d) 隱藏激勵的導數：
  
$$
\frac{\partial h_j}{\partial z_j^{(1)}} = f'\big(z_j^{(1)}\big)
$$
- (e) 隱藏層線性組合對第一層權重：
  
$$
\frac{\partial z_j^{(1)}}{\partial w_{ji}^{(1)}} = x_i
$$

將它們相乘並對 \(k\) 加總：

$$
\frac{\partial L}{\partial w_{ji}^{(1)}}
= \left[\sum_{k=1}^{4} (y_k - t_k)\cdot g'\big(z_k^{(2)}\big)\cdot w_{kj}^{(2)} \right]
\cdot f'\big(z_j^{(1)}\big)\cdot x_i
$$


### **6. 參數更新（以梯度下降為例）**

設學習率為 $\eta$：

$$ W_{kj}^{(2)} \leftarrow W_{kj}^{(2)} - \eta  \frac{\partial L}{\partial W_{kj}^{(2)}} $$  
$$ b_k^{(2)} \leftarrow b_k^{(2)} - \eta  \frac{\partial L}{\partial b_k^{(2)}} $$  
$$ W_{ji}^{(1)} \leftarrow W_{ji}^{(1)} - \eta  \frac{\partial L}{\partial W_{ji}^{(1)}} $$  
$$ b_j^{(1)} \leftarrow b_j^{(1)} - \eta  \frac{\partial L}{\partial b_j^{(1)}} $$

---






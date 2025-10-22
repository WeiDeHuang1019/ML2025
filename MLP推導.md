## **MLP 前向傳遞 (Forward Propagation)** 的公式流程  
目前的結構是：

* 輸入層：特徵向量 $\mathbf{x} = [x_1, x_2, ..., x_N]^T$
* 隱藏層：(K) 個神經元，每個有激勵函數 $f(\cdot)$
* 輸出層：4 個神經元（對應 4 個類別），最後會接 loss function

---

### 1. 輸入層 → 隱藏層

對第 (j) 個隱藏層神經元：

$
z_j = \sum_{i=1}^{N} w_{ji}^{(1)} x_i + b_j^{(1)}
$

$
h_j = f(z_j)
$

* $w_{ji}^{(1)}$：輸入層到隱藏層的權重
* $b_j^{(1)}$：隱藏層神經元的 bias
* $f(\cdot)$：激勵函數 (sigmoid, ReLU, tanh...)
* $h_j$：隱藏層的輸出

矩陣化後：

$
\mathbf{h} = f\big(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\big)
$

其中：
$\mathbf{h} \in \mathbb{R}^K ; W^{(1)} \in \mathbb{R}^{K \times N}$

---

### 2. 隱藏層 → 輸出層

對第 (k) 個輸出神經元：

$
o_k = \sum_{j=1}^{K} w_{kj}^{(2)} h_j + b_k^{(2)}
$

如果輸出層要做分類（4 個類別），通常會接 **softmax**：

$
y_k = \frac{e^{o_k}}{\sum_{m=1}^{4} e^{o_m}}, \quad k = 1,2,3,4
$

矩陣化後：

$
\mathbf{o} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}\
$

$
\mathbf{y} = \text{softmax}(\mathbf{o})
$

其中：
$\mathbf{y} \in \mathbb{R}^4$ 為預測的類別機率分佈

---

總結 Forward Pass 的兩步：

1. **輸入層到隱藏層：**
   $
   \mathbf{h} = f\big(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\big)
   $

2. **隱藏層到輸出層：**
   $
   \mathbf{o} = W^{(2)} \mathbf{h} + \mathbf{b}^{(2)}, \quad \mathbf{y} = \text{softmax}(\mathbf{o})
   $



---

## **MLP 反向傳播 (Backpropagation)** 的公式流程

以 MSE 為目標函數，目的是計算各層權重與偏置的梯度，以更新參數。

---

### 1. 損失函數 (Mean Squared Error)

對於輸出向量 $\mathbf{y} = [y_1, y_2, y_3, y_4]^T$ 與真實標籤 $\mathbf{t} = [t_1, t_2, t_3, t_4]^T$：

$
L = \frac{1}{2} \sum_{k=1}^{4} (y_k - t_k)^2
$

矩陣化表示：

$
L = \frac{1}{2} |\mathbf{y} - \mathbf{t}|^2
$

---

### 2. 輸出層的誤差項

令輸出層輸出為：
$
\mathbf{y} = g(\mathbf{z}^{(2)}) = g(W^{(2)}\mathbf{h} + \mathbf{b}^{(2)})
$

則損失對輸出層輸入的偏導數為：

$
\boldsymbol{\delta}^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} = (\mathbf{y} - \mathbf{t}) \odot g'(\mathbf{z}^{(2)})
$  
<br>*未矩陣化:$\delta_k^{(2)} = (y_k - t_k)  g'(z_k^{(2)})$

> 若輸出層為**線性輸出**（即 $g'(z) = 1$），則
> $
> \boldsymbol{\delta}^{(2)} = \mathbf{y} - \mathbf{t}
> $
---

### 3. 輸出層的權重與偏置梯度

$
\frac{\partial L}{\partial W^{(2)}} =  \frac{\partial L}{\partial \mathbf{z}^{(2)}} \cdot \frac{\partial \mathbf{z}^{(2)}}{\partial W^{(2)}}\boldsymbol = {\delta}^{(2)} \mathbf{h}^T
$

$
\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} \cdot \frac{\partial \mathbf{z}^{(2)}}{\partial b^{(2)}}\boldsymbol =\boldsymbol{\delta}^{(2)}
$

---

### 4. 隱藏層的誤差項

將誤差往前傳遞：

$
\boldsymbol{\delta}^{(1)} = (W^{(2)T} \boldsymbol{\delta}^{(2)}) \odot f'(\mathbf{z}^{(1)})
$

其中：

* $f'(\cdot)$ 為隱藏層激勵函數的導數 (例如 ReLU、sigmoid、tanh)
* $\odot$ 表示逐元素相乘（element-wise multiplication）

---

### 5. 隱藏層的權重與偏置梯度

$
\frac{\partial L}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T
$

$
\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}
$

---

### 6. 參數更新（以梯度下降為例）

設學習率為 $\eta$：

$
W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}, \quad
\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}
$

其中 $l = 1, 2$ 分別代表隱藏層與輸出層。

---

### 7. 總結 Backpropagation 流程

1. **Forward pass**
   $
   \mathbf{h} = f(W^{(1)} \mathbf{x} + \mathbf{b}^{(1)}), \quad
   \mathbf{y} = g(W^{(2)} \mathbf{h} + \mathbf{b}^{(2)})
   $

2. **Compute loss**
   $
   L = \frac{1}{2}|\mathbf{y} - \mathbf{t}|^2
   $

3. **Backward pass**

   * $\boldsymbol{\delta}^{(2)} = (\mathbf{y} - \mathbf{t}) \odot g'(\mathbf{z}^{(2)})$
   * $\frac{\partial L}{\partial W^{(2)}} = \boldsymbol{\delta}^{(2)} \mathbf{h}^T$
   * $\frac{\partial L}{\partial \mathbf{b}^{(2)}} = \boldsymbol{\delta}^{(2)}$
   * $\boldsymbol{\delta}^{(1)} = (W^{(2)T}\boldsymbol{\delta}^{(2)}) \odot f'(\mathbf{z}^{(1)})$
   * $\frac{\partial L}{\partial W^{(1)}} = \boldsymbol{\delta}^{(1)} \mathbf{x}^T$
   * $\frac{\partial L}{\partial \mathbf{b}^{(1)}} = \boldsymbol{\delta}^{(1)}$

---










# 🧠 MLP 反向傳播（非矩陣形式推導）

考慮一個兩層的多層感知器 (MLP)：

輸入層 → 隱藏層 → 輸出層  
權重與偏置如下：

- 隱藏層：  
  $begin:math:text$ z_j^{(1)} = \\sum_i W_{ji}^{(1)} x_i + b_j^{(1)} $end:math:text$  
  $begin:math:text$ h_j = f(z_j^{(1)}) $end:math:text$

- 輸出層：  
  $begin:math:text$ z_k^{(2)} = \\sum_j W_{kj}^{(2)} h_j + b_k^{(2)} $end:math:text$  
  $begin:math:text$ y_k = g(z_k^{(2)}) $end:math:text$

---

## **1️⃣ 損失函數 (Mean Squared Error)**

$begin:math:display$
L = \\frac{1}{2} \\sum_{k=1}^{K} (y_k - t_k)^2
$end:math:display$

---

## **2️⃣ 輸出層誤差項 δ**

對每個輸出層神經元 $begin:math:text$k$end:math:text$：

$begin:math:display$
\\delta_k^{(2)} = \\frac{\\partial L}{\\partial z_k^{(2)}} = (y_k - t_k) \\, g'(z_k^{(2)})
$end:math:display$

> 若輸出層為線性輸出（$begin:math:text$ g'(z) = 1 $end:math:text$），則：
> $begin:math:display$
> \\delta_k^{(2)} = y_k - t_k
> $end:math:display$

---

## **3️⃣ 輸出層權重與偏置的梯度**

對每一條連線 $begin:math:text$ W_{kj}^{(2)} $end:math:text$：

$begin:math:display$
\\frac{\\partial L}{\\partial W_{kj}^{(2)}} = \\delta_k^{(2)} \\, h_j
$end:math:display$

對每一個輸出層偏置：

$begin:math:display$
\\frac{\\partial L}{\\partial b_k^{(2)}} = \\delta_k^{(2)}
$end:math:display$

---

## **4️⃣ 隱藏層的誤差項 δ**

對每個隱藏層神經元 $begin:math:text$ j $end:math:text$：

$begin:math:display$
\\delta_j^{(1)} = f'(z_j^{(1)}) \\sum_{k} W_{kj}^{(2)} \\, \\delta_k^{(2)}
$end:math:display$

這表示第 $begin:math:text$ j $end:math:text$ 個隱藏層神經元的誤差，是由所有連到輸出層的誤差信號反傳後加權求和，再乘上激勵函數導數。

---

## **5️⃣ 隱藏層權重與偏置的梯度**

對每一條連線 $begin:math:text$ W_{ji}^{(1)} $end:math:text$：

$begin:math:display$
\\frac{\\partial L}{\\partial W_{ji}^{(1)}} = \\delta_j^{(1)} \\, x_i
$end:math:display$

對每一個隱藏層偏置：

$begin:math:display$
\\frac{\\partial L}{\\partial b_j^{(1)}} = \\delta_j^{(1)}
$end:math:display$

---

## **6️⃣ 參數更新（以梯度下降為例）**

設學習率為 $begin:math:text$ \\eta $end:math:text$：

$begin:math:display$
W_{kj}^{(2)} \\leftarrow W_{kj}^{(2)} - \\eta \\, \\frac{\\partial L}{\\partial W_{kj}^{(2)}}
$end:math:display$
$begin:math:display$
b_k^{(2)} \\leftarrow b_k^{(2)} - \\eta \\, \\frac{\\partial L}{\\partial b_k^{(2)}}
$end:math:display$
$begin:math:display$
W_{ji}^{(1)} \\leftarrow W_{ji}^{(1)} - \\eta \\, \\frac{\\partial L}{\\partial W_{ji}^{(1)}}
$end:math:display$
$begin:math:display$
b_j^{(1)} \\leftarrow b_j^{(1)} - \\eta \\, \\frac{\\partial L}{\\partial b_j^{(1)}}
$end:math:display$

---

## **7️⃣ 總結 Backpropagation（逐節點形式）**

1. **Forward pass**
   $begin:math:display$
   z_j^{(1)} = \\sum_i W_{ji}^{(1)} x_i + b_j^{(1)}, \\quad h_j = f(z_j^{(1)})
   $end:math:display$
   $begin:math:display$
   z_k^{(2)} = \\sum_j W_{kj}^{(2)} h_j + b_k^{(2)}, \\quad y_k = g(z_k^{(2)})
   $end:math:display$

2. **Compute loss**
   $begin:math:display$
   L = \\frac{1}{2} \\sum_k (y_k - t_k)^2
   $end:math:display$

3. **Backward pass**
   $begin:math:display$
   \\delta_k^{(2)} = (y_k - t_k) g'(z_k^{(2)})
   $end:math:display$
   $begin:math:display$
   \\frac{\\partial L}{\\partial W_{kj}^{(2)}} = \\delta_k^{(2)} h_j, \\quad
   \\frac{\\partial L}{\\partial b_k^{(2)}} = \\delta_k^{(2)}
   $end:math:display$
   $begin:math:display$
   \\delta_j^{(1)} = f'(z_j^{(1)}) \\sum_k W_{kj}^{(2)} \\delta_k^{(2)}
   $end:math:display$
   $begin:math:display$
   \\frac{\\partial L}{\\partial W_{ji}^{(1)}} = \\delta_j^{(1)} x_i, \\quad
   \\frac{\\partial L}{\\partial b_j^{(1)}} = \\delta_j^{(1)}
   $end:math:display$

---

✅ 如此一來，整個反向傳播過程都以**逐節點、逐權重的標量形式**推導完成，不依賴任何矩陣或向量表示。  
這樣的形式也方便直接實作成巢狀 `for` 迴圈版本的程式。
✅ 如此一來，整個反向傳播過程都以逐節點、逐權重的標量形式推導完成，不依賴任何矩陣或向量表示。
這樣的形式也方便直接實作成巢狀 for 迴圈版本的程式。




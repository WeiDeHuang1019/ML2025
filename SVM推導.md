# SVM 推導

## 設定
- 二維資料點： $x = (x_1, x_2)^\top$
- 權重向量： $w = (w_1, w_2)^\top$
- 決策邊界： $w \cdot x + b = 0$
- 兩條邊界線：
  - 正邊界： $w \cdot x + b = +1$
  - 負邊界： $w \cdot x + b = -1$

---

## 任取兩點在兩邊界上($x_{m}$ 與 $x_{n}$)
$$
\begin{cases}
w_1 x_{1m} + w_2 x_{2m} + b = +1 \\
w_1 x_{1n} + w_2 x_{2n} + b = -1
\end{cases}
$$

---

## 相減以消去 $b$
$$
w_1(x_{1m} - x_{1n}) + w_2(x_{2m} - x_{2n}) = 2
$$

以向量形式：

$$
w \cdot (x_m - x_n) = 2
$$

---

## 投影到法向量方向
$\vec{w}$即為決策線的法向量，利用內積算出 $(\vec{x_m} - \vec{x_n})$ 在 $\vec{w}$ 上的投影量，則兩條平行線之間的垂直距離（即 margin）為：

$$
w \cdot (x_m - x_n) = 2
$$

$$
L = \frac{w}{\|w\|} \cdot (x_m - x_n)
   = \frac{w \cdot (x_m - x_n)}{\|w\|}
   = \frac{2}{\|w\|}
$$

$$
\| \vec{x}_m - \vec{x}_n \| \cdot \cos\theta \cdot \| \vec{w} \| = 2
$$

$$
\| \vec{x}_m - \vec{x}_n \| \cdot \cos\theta = L
$$

$$
L \cdot \| \vec{w} \| = 2
$$

$$
L = \frac{2}{\| \vec{w} \|}
$$


---

## 得出間隔(Margin)
$$
\boxed{L = \frac{2}{\|w\|}}
$$

> 若邊界設為 $w \cdot x + b = \pm k$，則距離為  
> $$L = \dfrac{2k}{\|w\|}$$

以圖表示:
<img width="910" height="658" alt="image" src="https://github.com/user-attachments/assets/46870fba-61f1-4101-b32c-80d977f76dfd" /> <br>

---

## 最大化Margin 
我們希望讓兩類之間的距離 **盡可能大**：

$$
\max L = \max \frac{2}{\|w\|}
$$

但為了方便運算(mathematically convenient)，我們可以忽略常數 2（不影響最優解），變成：

$$
\max \frac{1}{\|w\|}
$$

---

## 最大化轉為最小化 
因為 $\frac{1}{\|w\|}$ 與 $\|w\|$ 呈 **單調遞減** 關係：

> 當 $\|w\|$ 越小，間隔越大。

所以：

$$
\max \frac{1}{\|w\|}\Longleftrightarrow\min \|w\|
$$

除此之外，我們將其平方化以便求導，因為平方函數是單調遞增的（在 $\|w\|>0$ 區間內），因此最小化 $\|w\|$ 與最小化 $\|w\|^2$ 結果相同。 
而平方後的形式在數學上更平滑，方便計算導數。

$$
\min \|w\|\Longleftrightarrow\min \frac{1}{2}\|w\|^2
$$


---

## 目標與約束

$$
\begin{aligned}
\min_{w,b} & \frac{1}{2}\|w\|^2 \\
\text{s.t. } & y_i(w \cdot x_i + b) \ge 1, \quad \forall i
\end{aligned}
$$

這是一個 帶不等式限制條件 (inequality constraints) 的最小化問題。
我們要最小化 $\frac{1}{2}\|w\|^2$ ，但又要同時滿足約束條件: $y_i(w \cdot x_i + b) \ge 1, \quad \forall i$。

---
## Lagrange 運算子
如果沒有約束，我們可以直接對 \(w,b\) 求導取極值。  
但現在有約束條件：

$$
g_i(w,b) = 1 - y_i(w\cdot x_i + b) \le 0
$$

要處理這種「有條件」的最小化，就需要用 **Lagrange 乘子法 (Lagrange Multiplier Method)**。  


### Lagrangian 的構造方式

一般形式：

$$
\mathcal{L}(x,\alpha) = f(x) + \sum_i \alpha_i g_i(x),
\quad \alpha_i \ge 0
$$

當 $\(g_i(x)\le 0\)$（條件滿足）時，懲罰項對結果沒有影響；  
若 $\(g_i(x) > 0\)$（條件違反），則該項會讓 $\(\mathcal{L}\)$ 增大，  
從而「懲罰」違反條件的情況。


### 套用到 SVM 的具體形式

- 目標函數：

$$
f(w,b) = \frac{1}{2}\|w\|^2
$$

- 約束條件：

$$
1 - y_i(w\cdot x_i + b) \le 0
$$

因此構造 Lagrangian：

$$
\boxed{
\mathcal{L}(w,b,\alpha)
= \frac{1}{2}\|w\|^2 - \sum_{i=1}^{m} \alpha_i\big[\,y_i(w\cdot x_i + b) - 1\,\big]}
$$

---

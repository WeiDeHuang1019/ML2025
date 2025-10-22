# 🎯 A Survey on Vision Transformer
**Han et al., IEEE TPAMI 2023**

---

## 🕒 報告時間配置（20 分鐘）

| 區段 | 主題 | 時間 | 重點說明 |
|------|------|------|-----------|
| 1️⃣ | 導言：Transformer 為什麼重要 | 3 分鐘 | 引出 Transformer 在 AI 的革命性角色，特別是它如何從 NLP 延伸到視覺。 |
| 2️⃣ | Transformer 基礎原理 | 4 分鐘 | 用直觀圖解介紹 Self-Attention、Multi-Head Attention、Positional Encoding。 |
| 3️⃣ | Vision Transformer（ViT）核心概念 | 4 分鐘 | 解釋 ViT 架構、Patch Embedding、分類流程與 CNN 比較。 |
| 4️⃣ | 各類 Vision Transformer 模型與應用 | 6 分鐘 | 用表格或圖展示代表性模型（DETR、Swin Transformer、IPT、CLIP、DALL·E等）。 |
| 5️⃣ | 現有挑戰與未來方向 | 3 分鐘 | 總結挑戰（資料需求、運算量、效率化）與研究趨勢。 |

---

## 🧩 簡報結構與內容建議

### 🔹 Slide 1：封面
- **標題**：A Survey on Vision Transformer  
- **作者／報告人／日期**  
- **開場白建議：**
  > “Transformer 不只是懂語言的 AI，它也正在改變我們看世界的方式。”

---

### 🔹 Slide 2–3：導言 — Transformer 的崛起
**重點：建立背景與動機**

- 傳統神經網路的分工：
  - MLP：全連接層
  - CNN：影像特化，捕捉局部特徵
  - RNN：序列資料
- Transformer 的特色：
  - 基於 **Self-Attention**，可同時捕捉全域依存關係
- NLP 成功案例：**BERT、GPT-3**
- 問題引出：
  > “如果 Transformer 能理解語言，那它能不能『理解影像』？”

---

### 🔹 Slide 4–6：Transformer 核心原理
**重點：用圖講清楚 Attention 怎麼運作**

- **Self-Attention：**
  - 每個輸入向量生成 Query（Q）、Key（K）、Value（V）
  - 計算注意力分數：
    ```
    Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) * V
    ```
- **Multi-Head Attention：**  
  不同注意力頭從多角度學習特徵關聯。
- **Positional Encoding：**  
  彌補 Transformer 缺乏序列／空間位置信息的限制。
- **小結：**
  > “Self-Attention 幫助模型知道誰該關注誰。”

---

### 🔹 Slide 7–9：Vision Transformer（ViT）架構
**重點：將影像轉成序列處理**

- 流程：
  1. 將影像切成固定大小的 Patch（如 16×16）
  2. 每個 Patch 經線性投影 → 形成 **Patch Embedding**
  3. 加上 **Positional Encoding**
  4. 丟入 Transformer Encoder
  5. [CLS] token 作為整張圖的最終表示  
- **成效與限制：**
  - 小資料集下不如 CNN（缺乏歸納偏好）
  - 大資料集（如 JFT-300M）則能超越 CNN  
- **延伸模型：** DeiT（Data-efficient Image Transformer）

---

### 🔹 Slide 10–14：Vision Transformer 的家族與應用
**重點：用圖表整理分類與代表模型**

| 類別 | 範例模型 | 特點 | 應用 |
|------|-----------|------|------|
| 🧱 Backbone | ViT、DeiT、Swin、TNT | 不同注意力設計 | 影像分類 |
| 🎯 High-level vision | DETR、Mask2Former | Transformer for Object Detection / Segmentation | 目標偵測、語意分割 |
| 🌈 Low-level vision | IPT、TTSR | Image Processing Transformer | 超解析度、影像增強 |
| 🎥 Video processing | TimeSformer、STTN | 時序注意力 | 影片理解、補幀 |
| 🔤 Multimodal | CLIP、DALL·E、UniT | 圖文對齊與生成 | 文字生成影像、跨模態任務 |

> 可搭配時間軸圖展示演進：ViT → Swin → CLIP → DALL·E

---

### 🔹 Slide 15–17：現有挑戰與未來方向
**重點：引導思考**

#### ⚠️ 現有挑戰
- 高運算成本、記憶體需求大  
- 資料需求高（data hungry）  
- 訓練不穩定、缺乏歸納偏好  

#### 🚀 研究趨勢
- **Efficient Transformer**：Swin、MobileViT  
- **CNN + Transformer 混合架構**：ConvFormer、CMT  
- **自監督學習／遮罩預訓練**：MAE、SimMIM  
- **多模態融合**：CLIP、DALL·E、GPT-4V 類模型  

> “Vision Transformer 不只是模型，而是一種新的視覺理解範式。”

---

### 🔹 Slide 18：結語
**重點：簡潔收尾**

- Transformer 已成為電腦視覺的重要支柱  
- 它讓我們重新思考「如何讓模型看懂世界」  
- Takeaway：
  > “從 CNN 到 Transformer，是從『卷積』到『注意力』的時代轉換。”

---

### 🔹 Slide 19–20：Q&A
- 預留 2–3 分鐘互動  
- 可放一張總覽圖或時間軸作為回顧畫面

---

## 🧭 報告技巧建議
- ✅ **用圖勝於文字**：Attention、ViT、Swin 等示意圖。  
- ✅ **挑重點講模型**：每類只挑 1–2 個代表模型。  
- ✅ **從 CNN 對比切入**：初學者容易理解轉變。  
- ✅ **結尾強化印象**：Transformer 改變了電腦視覺的思考方式。

---

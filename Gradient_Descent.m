% MATLAB 程式：展示 MLP 的梯度下降與可視化
clear all; close all; clc;

% 隨機種子（確保結果可重現）
rng(42);

% === 模擬數據 ===
% 輸入特徵向量 x (2 維，模擬簡單數據)
X = [1, 2]'; % 輸入向量 (2x1)
t = 3;       % 目標輸出（標量，簡單回歸問題）

% === MLP 參數初始化 ===
% 隱藏層：2 個神經元
W1 = rand(2, 2); % 第一層權重 (2x2)
b1 = rand(2, 1); % 第一層偏置 (2x1)
% 輸出層：1 個神經元
W2 = rand(1, 2); % 第二層權重 (1x2)
b2 = rand(1, 1); % 第二層偏置 (1x1)

% === 超參數 ===
eta = 0.01;       % 學習率
num_iterations = 1000; % 迭代次數
losses = zeros(num_iterations, 1); % 儲存每次迭代的損失

% === 激活函數（隱藏層使用 sigmoid） ===
sigmoid = @(z) 1 ./ (1 + exp(-z));
sigmoid_deriv = @(z) sigmoid(z) .* (1 - sigmoid(z));

% === 梯度下降主迴圈 ===
for iter = 1:num_iterations
    % --- 前向傳播 ---
    % 隱藏層
    z1 = W1 * X + b1; % z_j^{(1)} = W_{ji}^{(1)} x_i + b_j^{(1)}
    h = sigmoid(z1);  % h_j = f(z_j^{(1)})
    
    % 輸出層
    z2 = W2 * h + b2; % z_k^{(2)} = W_{kj}^{(2)} h_j + b_k^{(2)}
    y = z2;           % y_k = z_k^{(2)} (線性輸出，無激活函數)
    
    % --- 計算損失（MSE） ---
    loss = 0.5 * (y - t)^2; % L = 1/2 (y_k - t_k)^2
    losses(iter) = loss;
    
    % --- 反向傳播 ---
    % 輸出層誤差項
    delta2 = (y - t); % δ_k^{(2)} = (y_k - t_k) (線性輸出，g'(z) = 1)
    
    % 輸出層梯度
    dW2 = delta2 * h'; % ∂L/∂W_{kj}^{(2)} = δ_k^{(2)} h_j
    db2 = delta2;      % ∂L/∂b_k^{(2)} = δ_k^{(2)}
    
    % 隱藏層誤差項
    delta1 = sigmoid_deriv(z1) .* (W2' * delta2); % δ_j^{(1)} = f'(z_j^{(1)}) * Σ_k W_{kj}^{(2)} δ_k^{(2)}
    
    % 隱藏層梯度
    dW1 = delta1 * X'; % ∂L/∂W_{ji}^{(1)} = δ_j^{(1)} x_i
    db1 = delta1;      % ∂L/∂b_j^{(1)} = δ_j^{(1)}
    
    % --- 更新參數 ---(這邊optimizer使用標準梯度下降法(SGD)，後續改進可使用Adam（Adaptive Moment Estimation）)
    W2 = W2 - eta * dW2; % W_{kj}^{(2)} ← W_{kj}^{(2)} - η * ∂L/∂W_{kj}^{(2)}
    b2 = b2 - eta * db2; % b_k^{(2)} ← b_k^{(2)} - η * ∂L/∂b_k^{(2)}
    W1 = W1 - eta * dW1; % W_{ji}^{(1)} ← W_{ji}^{(1)} - η * ∂L/∂W_{ji}^{(1)}
    b1 = b1 - eta * db1; % b_j^{(1)} ← b_j^{(1)} - η * ∂L/∂b_j^{(1)}
end

% === 可視化結果 ===
% 繪製損失隨迭代次數的變化
figure;
plot(1:num_iterations, losses, 'b-', 'LineWidth', 2);
xlabel('Iteration');
ylabel('Mean Squared Error Loss');
title('Gradient Descent: Loss vs. Iteration');
grid on;

% 顯示最終預測
z1 = W1 * X + b1;
h = sigmoid(z1);
y_final = W2 * h + b2;
fprintf('最終預測值: %.4f\n', y_final);
fprintf('目標值: %.4f\n', t);

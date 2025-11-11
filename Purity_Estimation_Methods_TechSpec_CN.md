# 混样测序籽粒纯度定量：方法学与技术方案
**Purity Estimation for Pooled Resequencing: Methods & Technical Specification**


---

## 摘要（Executive Summary）
我们提出一套用于**混样重测序**场景下估计**标准样（目标品种）比例**（即“纯度”）的方法学体系：

1. **方法一：协方差-矩估计（MoM, Method of Moments）** —— 在标准样基因型已知的前提下，利用大量有效差异位点上**观测等位比例**与**标准样基因型**之间的**加权协方差**构造**闭式解**，快速、稳健、低方差。  
2. **方法二：K-EM 背景簇（K Backgrounds via EM）** —— 将非标准样的混入来源抽象为 $K$ 个潜在背景簇（$K\in[3,9]$ 自适应），在**误差校正**与**稳健损失**下，联合估计纯度 $w$ 及背景参数，并以 **BIC/ICL** 选择最优 $K$。

同时包含：误判率 $e$ 的对称误差校正、加权与稳健筛点、**Bootstrap** 置信区间、**仿真标定（Calibration）**的线性校正等。

---

## 1. 记号与数据结构（Notation）
- $N$：有效差异位点数（推荐 $2\!\sim\!4\times 10^5$）。  
- 位点 $i$ 的量：
  - $n_i$：总覆盖深度（reads 数）。  
  - $k_i$：ALT 等位支持的 reads 数。  
  - $p_i = k_i / n_i$：观测 ALT 比例。  
  - $g_i \in \{0,1,0.5\}$：**标准样**在位点 $i$ 的基因型（REF/ALT/Het）。默认仅用纯合位点 $g_i\in\{0,1\}$。  
- 其他：
  - $e \in [0,0.5)$：对称等位**误判率**（REF↔ALT），通常 $0.005\!\sim\!0.02$。  
  - $w\in[0,1]$：**待估**的标准样比例（纯度）。

---

## 2. 观测层与误差校正
对称误判率 $e$ 下，观测层比例 $p_i$ 与生物层真实比例 $q_i$ 的关系近似为：

$$
\mathbb{E}[\,p_i \mid q_i, e\,] \;\approx\; e + (1-2e)\,q_i .
$$

因此定义**误差校正后的等位比例**：

$$
c_i \;=\; \frac{p_i - e}{1 - 2e}\,.
$$

> 实操筛点与加权：  
> 选取集合 $\mathcal{I}=\{\,i \mid g_i\in\{0,1\},\, n_i\ge n_{\min},\, c_i \text{ 有效}\,\}$；  
> 设加权 $w_i^\ast \propto n_i$，并对极端深度做上分位截断（如 99%）；  
> 以 MAD 构造 $z$ 分数对异常位点做稳健切尾（如 $|z|\le 5$）。

---

## 3. 协方差-矩估计（MoM）：闭式主估计
在线性混合近似下（$t_i$ 表示非标准样背景的 ALT 比例）：

$$
c_i \;\approx\; w\,g_i + (1-w)\,t_i\,.
$$

若 $t_i$ 与 $g_i$ 近似不相关，则可用加权协方差给出 $w$ 的闭式解：

$$
\widehat{w}_{\text{MoM}}
\;=\;
\frac{\operatorname{Cov}_w(g,c)}{\operatorname{Var}_w(g)}
\;=\;
\frac{\sum_{i\in\mathcal{I}} w_i^\ast\,(g_i-\bar g)\,(c_i-\bar c)}{\sum_{i\in\mathcal{I}} w_i^\ast\,(g_i-\bar g)^2}\,,
\qquad \widehat{w}_{\text{MoM}}\in[0,1]\,.
$$

其中 $\bar g,\bar c$ 为加权均值。直觉上，$c$ 与 $g$ 线性关系越强，协方差越大；归一化到 $g$ 的方差即得比例 $w$。

---

## 4. 二项式似然视角（补充推导）
测序层面遵循：
$$
k_i \mid n_i \sim \operatorname{Binomial}(n_i,\, p_i)\,.
$$
以生物层比例 $q_i$ 表示：
$$
q_i \;\approx\; w\,g_i + (1-w)\,t_i,\qquad
\mathbb{E}[p_i] \approx e + (1-2e)q_i\,.
$$
MoM 可看作对线性混合模型的一阶广义矩估计，兼具**快速与稳健**。当背景 $t_i$ 异质且与 $g_i$ 相关性弱时表现尤佳。

---

## 5. K-EM 背景簇模型（K Backgrounds via EM）
为吸收复杂混入来源，将背景表示为 $K$ 个潜在簇：
$$
c_i \;\approx\; w\,g_i \;+\; (1-w)\sum_{k=1}^{K} r_{ik}\, f_k,
\qquad \sum_{k=1}^{K} r_{ik}=1,\; r_{ik}\ge 0\,,
$$
其中 $f_k\in[0,1]$ 为第 $k$ 簇中心，$r_{ik}$ 为位点 $i$ 的软分配（责任度）。

在**加权稳健**（Huber 风格）损失下最小化目标：
$$
\mathcal{L}(w,\mathbf f,\mathbf r)
\;=\;
\sum_{i\in\mathcal{I}} w_i^\ast\,
\rho\!\Big(
c_i - \big[w\,g_i + (1-w)\sum_{k=1}^{K} r_{ik} f_k\big]
\Big)\,.
$$

**E 步（软分配）**：令
$$
t_i \;=\; \frac{c_i - w\,g_i}{\,1-w+\varepsilon\,}\,,
$$
给定温度 $\tau$ 与先验 $\pi_k$：
$$
r_{ik}\;\propto\;\pi_k\cdot
\exp\!\Big(\,-\,\frac{w_i^\ast\, (t_i - f_k)^2}{2\tau}\Big)\,,
\qquad \sum_k r_{ik}=1\,.
$$

**M 步**：
1. 更新簇中心  
$$
f_k \;\leftarrow\;
\frac{\sum_i r_{ik}\,w_i^\ast\, t_i}{\sum_i r_{ik}\,w_i^\ast}\,,
\quad\text{随后排序并施加最小间距约束 } f_k - f_{k-1}\ge \delta\,.
$$
2. 闭式更新 $w$（把 $\bar f_i=\sum_k r_{ik}f_k$ 看作背景估计）：
$$
\widehat{w}\;\leftarrow\;
\frac{\operatorname{Cov}_w\!\bigl(g,\,c-\bar f\bigr)}{\operatorname{Var}_w\!\bigl(g-\bar f\bigr)}\;\;\xrightarrow{\text{裁剪}}\; [0,1]\,.
$$
3. 更新簇权  
$$
\pi_k \;\propto\; \sum_i r_{ik}\,w_i^\ast\,.
$$

**退火与先验**：采用 $\tau$ 退火（由大到小）提升收敛稳健性；对 $w$ 可加弱先验 $w\sim\mathcal{N}(w_0,\tau_w^2)$（中心取 MoM）。

---

## 6. 模型选择（BIC / ICL）
记
$$
\mathrm{SSE} \;\approx\;
\sum_{i\in\mathcal{I}} w_i^\ast\,
\Big( c_i - \big[w\,g_i + (1-w)\sum_k r_{ik} f_k\big] \Big)^2\,,
$$
则
$$
\mathrm{BIC} \;=\; N \cdot \log\!\Big(\frac{\mathrm{SSE}}{N}\Big) + p \cdot \log N\,,
\qquad
\mathrm{ICL} \;=\; \mathrm{BIC} + N \cdot \mathbb{E}\big[\mathrm{Entropy}(\mathbf r)\big]\,,
$$
其中参数自由度 $p=(K-1)+K+1$（簇权 $\pi_k$ 的自由度 + 簇中心 $f_k$ + 一个 $w$）。ICL 更保守，常有效抑制过拟合的 $K$。

---

## 7. 不确定性评估（Bootstrap）
对 $\mathcal{I}$ 做 $B$ 次（如 200–500）**有放回重采样**：每次重新估计 $\widehat{w}_{\text{MoM}}$ 与 $\widehat{w}_{\text{K-EM}}$，
取分位数（2.5% 与 97.5%）给出**95% 置信区间**（Percentile/BCa）。

---

## 8. 仿真标定（Calibration）
为消除小幅系统性偏差（如浅深度截断/异常点影响），用较少位点（如 $2\times 10^4$）做 $S$ 次快速仿真：
得到成对数据 $\{(w^{\text{true}}, \widehat{w})\}$，拟合一阶线性关系
$$
w^{\text{true}} \;\approx\; a + b\,\widehat{w}\,,
$$
并对实际样本的估计应用：
$$
\widehat{w}_{\text{cal}} \;=\; a + b\,\widehat{w}\,.
$$
报告中同时给出**原始值**与**校正值**。

---

## 9. 质量控制（QC）与诊断图
- 位点利用率与剔除比例；深度分布与权重截断阈值。  
- $p_i$ 与 $c_i$ 的分布（直方/核密度）及极端点。  
- 拟合残差 $c_i - \big[w\,g_i + (1-w)\sum_k r_{ik} f_k\big]$ 的直方图 / QQ 图。  
- $K$ 的稳定性（不同初始化/子抽样的一致性）。  
- MoM 与 K-EM 的估计一致性对比图（Parity plot 等）。

---

## 10. 推荐参数与实践建议
- **位点规模**：$N\approx 2\sim4\times 10^5$；  
- **深度**：平均 $\sim 20\times$，位点最小深度 $n_{\min}=5$；  
- **误判率 $e$**：建议用纯对照/纯合位点**实测**（常见 $0.005\!\sim\!0.02$）。  
- **$K$ 范围**：$K\in[3,9]$；选择 ICL 更保守；最小中心间距 $\delta\approx 0.02$；  
- **稳健处理**：权重对深度做 99% 分位截断；MAD 切尾 $|z|\le 5$；  
- **标定**：仿真位点 $\sim 2\times 10^4$ / 次，重复 $S=50\!\sim\!200$（可仅用 MoM 加速）。

---

## 11. 实施流程（CLI/管线示例）
输入：面板位点 VCF、样本 BAM/CRAM、标准样基因型、误判率 $e$ 与参数配置。

```
purity_estimate \
  --vcf panel.vcf.gz --bam sample.bam --sample-id S1 \
  --ref-gt standard.genotypes.gz \
  --error-rate 0.01 --min-depth 5 \
  --k-min 3 --k-max 9 --use-icl \
  --bootstrap 300 \
  --calibrate-sims 100 --calibrate-snps 20000 \
  --out report_dir/
```

输出：JSON（数值）、PDF（图形）、CSV（诊断）、HTML（汇总）。

---

## 12. 边界与扩展
- **未知 $e$**：可并入增广模型联合估计，或按批次标定。  
- **复杂群体结构**：对 $f_k$ 引入分层/平滑先验或按 LD/染色体分块。  
- **标准样杂合**：可扩展为显式 $g_i\in\{0,0.5,1\}$ 的似然框架。  
- **贝叶斯化**：对 $(w,\mathbf f,\boldsymbol\pi)$ 赋先验，MCMC/VI 获得全概率与自然区间。  
- **多批次对齐**：引入批次效应项稳健合并多批次。

---

## 13. 结论
**MoM** 提供快速稳健的闭式主估计；**K-EM** 在复杂背景下提供可解释的联合拟合与模型证据（$K$ 的选择、簇中心/权重）。配合**误差校正**、**稳健处理**、**仿真标定**与**Bootstrap** 区间，形成严谨、可审计、可扩展的工业级纯度定量方案。


# 研报复现——开源证券《独家量价因子的高频测试》

1. **Polars 负责后勤和 IO**：极速读取 Parquet，极速过滤出每天合法的 `symbol`，对分钟线进行切片和排序。
2. **Numba 负责突击和运算**：`calc_smart_money_core` 被 `@njit` 编译成 C 语言级别的机器码，在拿到 numpy 数组后，瞬间完成 `for` 循环、状态判断和比例插值切割。

## 代码结构

```
Kysec_Quant_Project/
├── data/                   # 原始数据目录（日线、分钟线、资金流向等）
├── cache/                  # 缓存目录（存放计算好的单因子、复合因子的 Parquet 文件）
├── core/                   # 核心基础组件
│   ├── data_loader.py      # 数据读取与清洗（使用 Polars LazyFrame 处理大规模 IO）
│   ├── preprocessor.py     # 因子预处理（去极值、标准化、中性化，Polars 向量化实现）
│   └── optimizer.py        # 权重优化器（复合因子的 ICIR 最大化求解）
├── factors/                # 因子逻辑池（只关心因子怎么算）
│   ├── base.py             # 🌟 因子基类（定义统一的标准接口）
│   ├── smart_money.py      # 聪明钱因子 (Numba 计算核心)
│   ├── amplitude.py        # 理想振幅 & 长端动量
│   ├── apm.py              # APM因子
│   └── money_flow.py       # 大小单资金流 & 主动买卖因子
├── backtest/               # 回测与分析引擎
│   ├── evaluator.py        # 因子评价（IC, RankIC, 多空收益, Turnover, MaxDD 计算）
│   └── portfolio.py        # 分组逻辑与频率切换（周频/双周/月频映射）
├── notebooks/              # 用于调试和生成研报图表的交互环境
│   └── report_visualization.ipynb 
└── run_pipeline.py         # 顶层执行脚本，串联所有流程
```

## 量化因子定义汇总

---

### 1. 聪明钱 2.0 (Smart Money 2.0)

**计算步骤：**
1. **数据获取**：回溯选定股票过去 10 个交易日的分钟行情数据。
2. **指标构造**：计算每分钟的指标 $S_t$：
   $$S_t = \frac{|R_t|}{V_t^{0.25}}$$
   其中，$R_t$ 为第 $t$ 分钟涨跌幅，$V_t$ 为第 $t$ 分钟成交量。
3. **识别聪明钱**：将分钟数据按照 $S_t$ 从大到小排序，取成交量累积占比前 20% 的分钟，视为“聪明钱交易”。
4. **计算均价**：
   * 计算聪明钱交易的成交量加权平均价：$VWAP_{smart}$
   * 计算所有交易的成交量加权平均价：$VWAP_{all}$
5. **因子构造**：聪明钱因子 $Q$ 定义为：
   $$Q = \frac{VWAP_{smart}}{VWAP_{all}}$$

---

### 2. APM 因子 (Anticipated Profits Momentum)

**计算步骤：**
1. **收益率拆分**：回溯过去 20 日数据。记：
   每日上午股票收益率为 $r_t^{am}$，指数收益率为 $R_t^{am}$；
   每日下午股票收益率为 $r_t^{pm}$，指数收益率为 $R_t^{pm}$。
2. **残差回归**：将得到的 40 组 $(r, R)$ 数据进行回归：
   $$r_i = \alpha + \beta R_i + \varepsilon_i$$
   得到残差项 $\varepsilon_i$。
3. **计算残差差值**：将 40 个残差分为上午残差 $\varepsilon_t^{am}$ 和下午残差 $\varepsilon_t^{pm}$，计算每日差值：
   $$\delta_t = \varepsilon_t^{am} - \varepsilon_t^{pm}$$
4. **构造统计量**：
   $$stat = \frac{\mu(\delta_t)}{\sigma(\delta_t) / \sqrt{N}}$$
   其中 $\mu$ 为均值，$\sigma$ 为标准差。
5. **消除动量影响**：将 $stat$ 对动量因子（过去 20 日收益率 $Ret20$）进行横截面回归：
   $$stat_j = b \cdot Ret20_j + \varepsilon_j$$
6. **因子定义**：回归得到的残差值 $\varepsilon$ 即为 **APM 因子**。

---

### 3. 理想振幅 (Ideal Amplitude)

**计算步骤：**
1. **基本数据**：回溯最近 $N=20$ 个交易日的数据。
2. **计算振幅**：每日振幅 = (最高价 / 最低价 - 1)。
3. **分类计算**：
   * 选择收盘价较高的前 $\lambda$ (如 25%) 的交易日，计算其振幅均值，得到高价振幅因子 $V_{high}(\lambda)$。
   * 选择收盘价较低的前 $\lambda$ (如 25%) 的交易日，计算其振幅均值，得到低价振幅因子 $V_{low}(\lambda)$。
4. **因子定义**：
   $$V(\lambda) = V_{high}(\lambda) - V_{low}(\lambda)$$

---

### 4. 主动买卖因子 (Active Buying/Selling)

**计算步骤：**
1. **逐日计算买卖因子**：
   * **中大单主动买卖因子**：
       $$ACT_{正向,t} = \frac{\text{主动买入金额(大单+中单)} - \text{主动卖出金额(大单+中单)}}{\text{主动买入金额(大单+中单)} + \text{主动卖出金额(大单+中单)}}$$
   * **小单主动买卖因子**：
       $$ACT_{负向,t} = \frac{\text{主动买入金额(小单)} - \text{主动卖出金额(小单)}}{\text{主动买入金额(小单)} + \text{主动卖出金额(小单)}}$$
2. **筛选交易日**：回溯过去 20 个交易日，取收益率最高 $\lambda$ 比例的日期为“高收益日”，最低 $\lambda$ 比例的为“低收益日”。
3. **因子定义**：对高收益日的 $ACT_{正向,t}$ 取平均记为 $ACT_{正向}$；对低收益日的 $ACT_{负向,t}$ 取平均记为 $ACT_{负向}$。

---

### 5. 长端动量 (Long-term Momentum)

**计算步骤：**
1. **数据获取**：回溯最近 160 个交易日的数据。
2. **计算振幅**：每日振幅 = (最高价 / 最低价 - 1)。
3. **因子定义**：选择振幅较低的 70% 交易日，将这些日子的涨跌幅累加，得到 **长端动量因子**。

---

### 6. 大小单资金流 (Big/Small Order Flow)

**计算步骤：**
1. **计算资金流强度**：
   针对小单和大单分别计算强度 $S_t$，分子为（买额 - 卖额）之和，分母为其绝对值之和：
   $$S_t = \frac{\sum_{t-T}^{t} (buy_t - sell_t)}{\sum_{t-T}^{t} |buy_t - sell_t|}$$
2. **残差回归**：为了去相关，将其对过去 20 日涨跌幅做回归，得到残差 $\varepsilon_t$ 作为最终因子：
   $$S_t = a + b \cdot Ret20_t + \varepsilon_t$$



## 数据储存结构和示例

获取区间：2014-2021（为了精确计算因子值，部分数据延长到2013）

---

### (1) ST 情况 / `is_st.csv`
* **字段说明**：记录股票被实施特别处理（ST）的状态。
* **示例数据**：
    
    ```csv
    trade_date,symbol
    2013-01-04,000035
    2013-01-04,000056
    ```

### (2) 上市日期 / `list_date.csv`
* **字段说明**：股票首次上市交易的日期。
* **示例数据**：
    ```csv
    symbol,list_date
    000001,1991-04-03
    000002,1991-01-29
    ```

### (3) 交易日历 / `days.csv`
* **字段说明**：市场的有效交易日期列表。
* **示例数据**：
    ```csv
    trade_date
    2013-01-04
    2013-01-07
    ```

### (4) 停复牌情况 / `is_suspend.csv`
* **字段说明**：`suspend_timing` 代表日内的停牌时间段（如有）。
* **示例数据**：
    ```csv
    symbol,trade_date,suspend_timing
    000002,2013-01-04,
    000010,2013-01-04,
    002176,2013-01-07,09:30-13:00
    ```

### (5) 分钟级数据 / `year=YYYY/month=MM/day=DD/YYYY-MM-DD.parquet`
* **字段规范**：
    * `symbol`: 字符串格式，六位股票代码。
    * `time`: `pd.timedelta` 格式，表示日内时间 HH:MM。
    * `close`, `open`, `high`, `low`: `float64` 浮点数。
    * `volume`: 成交量（股），`int64`。
    * `amount`: 成交额（元），`float64`。
* **数据示例**：
    ```csv
    symbol,time,open,high,low,close,volume,amount
    000001, 34200000000, 13.64, 13.64, 13.64, 13.64, 386500, 5271860
    ```

### (6) 复权因子 / `adj_factor_YYYY.csv`
* **示例数据**：
    ```csv
    symbol,trade_date,adj_factor
    000001,2013-01-04,36.173
    000002,2013-01-04,115.013
    ```

### (7) 市值数据 / `mv_data_YYYY.csv`
* **字段说明**：`total_mv` (总市值), `circ_mv` (流通市值)。
* **示例数据**：
    ```csv
    symbol,trade_date,total_mv,circ_mv
    000001,2014-01-02,10025372.0933,6819327.9931
    000002,2014-01-02,8799966.2502,7730278.0508
    ```

### (8) 无风险利率 / `yield.csv`
* **字段说明**：`yield_10y` 通常指 10 年期国债收益率。
* **示例数据**：
    ```csv
    trade_date,yield_10y
    2014-01-02,4.6018
    2014-01-03,4.6415
    ```

### (9) 日级数据 / `daily_YYYY.csv`
* **字段定义**：
    * `pre_close`: 昨收价【除权价】。
    * `change`: 涨跌额。
    * `pct_chg`: 涨跌幅【基于除权后的昨收计算：$(今收 - 除权昨收) / 除权昨收$】。
    * `vol`: 成交量（手）。
    * `amount`: 成交额（千元）。
* **示例数据**：
    ```csv
    symbol,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount
    000001, 2013-01-04, 16.32, 16.45, 15.92, 15.99, 16.02, -0.03, -0.19, 443851.37, 717567.5466
    ```

### (10) 沪深 300-60min / `000300_SH_60min_YYYY.parquet`
* **字段说明**：包含 `close`, `open`, `high`, `low`, `trade_date`, `time` (`pd.timedelta` 格式)。
* **示例数据**：
    ```csv
    close,open,high,low,trade_date,time
    2423.97, 2423.97, 2423.97, 2423.97, 2013-12-02, 34200000000
    ```

### (11) 涨跌停价 / `A_share_limit_YYYY.csv`
* **示例数据**：
    ```csv
    trade_date,symbol,up_limit,down_limit
    2014-01-02,000001,13.48,11.03
    2014-01-02,000002,8.83,7.23
    ```

### (12) 行业分类 / `industry.csv`（中信一级）
* **示例数据**：
    ```csv
    symbol,in_date,l1_name,l1_code
    000001,2003-01-01,银行,CI005021.CI
    000002,2003-01-01,房地产,CI005023.CI
    000004,2003-01-01,计算机,CI005027.CI
    000004,2004-11-01,医药,CI005018.CI
    ```

### (13) 资金流向 / `moneyflow_YYYY.parquet`
* **分类标准**（基于主动买卖单统计）：
    * **小单 (sm)**：5万以下。
    * **中单 (md)**：5万～20万。
    * **大单 (lg)**：20万～100万。
    * **特大单 (elg)**：成交额 $\ge$ 100万。
* **单位说明**：量单位为“**手**”，金额单位为“**万元**”。
* **字段后缀**：
    * `buy_xx_vol` / `buy_xx_amount`: 买入量/额。
    * `sell_xx_vol` / `sell_xx_amount`: 卖出量/额。
    * `net_mf_vol` / `net_mf_amount`: 净流入量/额。
* **示例数据**：
    ```csv
    symbol,trade_date,buy_sm_vol,buy_sm_amount,sell_sm_vol,sell_sm_amount,buy_md_vol,buy_md_amount,sell_md_vol,sell_md_amount,buy_lg_vol,buy_lg_amount,sell_lg_vol,sell_lg_amount,buy_elg_vol,buy_elg_amount,sell_elg_vol,sell_elg_amount,net_elg_vol,net_elg_amount
    000001, 2013-01-04, 65214, 10520.61, 66792, 10793.5, 103994, 16796.95, 106840, 17280.15, 135227, 21890.92, 110032, 17804.94, 139242, 22520.2, 160013, 25850.1, -13017, -2059.68
    ```


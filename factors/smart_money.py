import numpy as np
import polars as pl
from numba import njit
from factors.base import BaseFactor
from core.data_loader import DataLoader

# =============================================================================
# 核心 Numba 算子：完全脱离 Python GIL，以接近 C 语言的速度执行数组运算
# =============================================================================
@njit(cache=True)
def calc_smart_money_core(close_arr: np.ndarray, vol_arr: np.ndarray, amt_arr: np.ndarray) -> float:
    """
    聪明钱因子 Numba 极速计算内核
    """
    n = len(close_arr)
    # 至少需要有一定的分钟数据才进行计算（例如1天约240根）
    if n < 200: 
        return np.nan

    s_metric = np.zeros(n)
    for i in range(n):
        v = vol_arr[i]
        # 过滤无成交量的垃圾K线
        if v < 1e-6:
            s_metric[i] = -1.0 
            continue
            
        # 计算分钟收益率  (首个 bar 的收益近似计作 0)
        if i == 0 or close_arr[i-1] < 1e-6:
            s_metric[i] = -1.0
        else:
            ret = np.abs(close_arr[i] / close_arr[i-1] - 1.0)
            # 计算聪明钱指标 S_t = |R_t| / (V_t ^ 0.25) 
            s_metric[i] = ret / (v ** 0.25)

    # 按照指标 S 从大到小降序排列 
    sorted_idx = np.argsort(s_metric)[::-1]
    
    total_vol = np.sum(vol_arr)
    if total_vol < 1e-6:
        return np.nan

    # 截取累计成交量排名前 20% 的聪明钱 
    target_vol = total_vol * 0.20
    curr_vol = 0.0
    smart_amt = 0.0
    smart_vol = 0.0

    for idx in sorted_idx:
        # 跳过无效计算值
        if s_metric[idx] < 0:
            continue 
            
        v = vol_arr[idx]
        # 精确插值：如果当前这根 bar 加上去超过了 20%，按比例切分 
        if curr_vol + v > target_vol:
            remain = target_vol - curr_vol
            ratio = remain / v
            smart_vol += remain
            smart_amt += amt_arr[idx] * ratio
            break
        else:
            smart_vol += v
            smart_amt += amt_arr[idx]
            curr_vol += v

    if smart_vol < 1e-6:
        return np.nan

    # 计算聪明钱的 VWAP 与全市场的 VWAP 
    vwap_smart = smart_amt / smart_vol
    vwap_all = np.sum(amt_arr) / total_vol

    if vwap_all < 1e-6:
        return np.nan

    # 聪明钱因子 Q = VWAP_smart / VWAP_all [cite: 604]
    return vwap_smart / vwap_all

# 用于 Polars struct 映射的包装器
def _smart_money_wrapper(row: dict) -> float:
    c = np.array(row['close_10d'], dtype=np.float64)
    v = np.array(row['vol_10d'], dtype=np.float64)
    a = np.array(row['amt_10d'], dtype=np.float64)
    return calc_smart_money_core(c, v, a)


# =============================================================================
# Polars 因子计算外壳
# =============================================================================
class SmartMoneyFactor(BaseFactor):
    """
    聪明钱因子 2.0 (开源量化评论36)
    """
    def __init__(self):
        super().__init__(factor_name="HF_SmartMoney")

    def _compute_logic(self, year: str) -> pl.LazyFrame:
        # 1. 获取纯净日历 (开启 require_intraday 剔除日内停牌)
        lf_base = self._get_clean_universe(year, require_intraday=True)
        
        # 2. 调用全新的统一下推加载器 (只需指定 year)
        lf_min = DataLoader.load_minute_data(year=year)
        
        # 补充：因为启用了 hive_partitioning，我们需要把路径里的年月日组合成 trade_date
        lf_min = lf_min.with_columns([
            pl.date(pl.col('year').cast(pl.Int32), 
                    pl.col('month').cast(pl.Int32), 
                    pl.col('day').cast(pl.Int32)).alias('trade_date')
        ])
        
        # 【内存优化核心】：利用 Inner Join 先一步裁减掉全市场中不可交易的分钟数据
        # 避免读取和处理无用的停牌股分钟 K 线
        lf_min_clean = lf_min.join(
            lf_base.select(['symbol', 'trade_date']), 
            on=['symbol', 'trade_date'], 
            how='inner'
        )
        
        # 【修改点 1】：在聚合为日级别前，确保分钟线按时间严格排序，防止 Numba 计算随机出错
        lf_min_clean = lf_min_clean.sort(['symbol', 'trade_date', 'time'])

        # 3. 将单日的分钟序列合并打包成列表 (List)
        lf_daily = lf_min_clean.group_by(['symbol', 'trade_date']).agg([
            pl.col('close').alias('close_day'),
            pl.col('volume').alias('vol_day'),
            pl.col('amount').alias('amt_day')
        ]).sort(['symbol', 'trade_date'])

        # 【修改点 2】：生成全局交易日索引 date_idx
        lf_daily = lf_daily.with_columns([
            pl.col("trade_date").rank(method="dense").cast(pl.Int32).alias("date_idx")
        ])

        # 【修改点 3】：按 10i (10个交易日) 进行动态滚动
        lf_roll = lf_daily.group_by_dynamic(
            "date_idx", 
            by="symbol", 
            every="1i", 
            period="10i", 
            closed="right"
        ).agg([
            pl.col("trade_date").last(), # 提取出真实的日期
            pl.col("close_day").list.explode().alias("close_10d"),
            pl.col("vol_day").list.explode().alias("vol_10d"),
            pl.col("amt_day").list.explode().alias("amt_10d"),
            pl.col("trade_date").len().alias("valid_days")
        ]).filter(pl.col("valid_days") >= 5)

        # 4. 极致性能：动态窗口聚合 (Dynamic GroupBy)
        # 按照 10 天为一个滑动窗口 ，把过去 10 天每天的分钟数据拼装起来
        lf_roll = lf_daily.group_by_dynamic(
            "trade_date", 
            by="symbol", 
            every="1d", 
            period="10d", 
            closed="right"
        ).agg([
            # 使用 list.explode() 把 10 个 [List(240), List(240)...] 打平成一个 List(2400)
            pl.col("close_day").list.explode().alias("close_10d"),
            pl.col("vol_day").list.explode().alias("vol_10d"),
            pl.col("amt_day").list.explode().alias("amt_10d"),
            pl.col("trade_date").len().alias("valid_days")
        ]).filter(pl.col("valid_days") >= 5) # 容错机制：窗口内至少有 5 天正常交易才计算

        # 5. 调用 Numba 内核
        # 使用 map_elements 遍历每天组装好的列表，传入 Numba 执行高速数学运算
        lf_res = lf_roll.with_columns([
            pl.struct(["close_10d", "vol_10d", "amt_10d"]).map_elements(
                _smart_money_wrapper, 
                return_dtype=pl.Float64
            ).alias(self.factor_name)
        ])

        return lf_res.select(['trade_date', 'symbol', self.factor_name])
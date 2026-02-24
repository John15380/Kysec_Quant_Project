import numpy as np
import polars as pl
from numba import njit
from factors.base import BaseFactor
from core.data_loader import DataLoader
from config.settings import Config

# =============================================================================
# Numba 核心：超高速处理过去20天（40个截面）的时序回归计算
# =============================================================================
@njit(cache=True)
def calc_apm_stat_core(r_am: np.ndarray, r_pm: np.ndarray, R_am: np.ndarray, R_pm: np.ndarray) -> float:
    n = len(r_am)
    if n < 15:  # 容错：至少需要 15 天有效数据
        return np.nan

    # 1. 将上午和下午的数据交织拼成长度为 2N (40) 的数组
    r_all = np.empty(2 * n)
    R_all = np.empty(2 * n)
    for i in range(n):
        r_all[2*i] = r_am[i]
        r_all[2*i+1] = r_pm[i]
        R_all[2*i] = R_am[i]
        R_all[2*i+1] = R_pm[i]

    # 2. OLS 线性回归: r = alpha + beta * R + epsilon
    mean_R = np.mean(R_all)
    mean_r = np.mean(r_all)
    
    var_R = np.sum((R_all - mean_R) ** 2)
    if var_R < 1e-12:
        return np.nan
        
    cov_rR = np.sum((R_all - mean_R) * (r_all - mean_r))
    beta = cov_rR / var_R
    alpha = mean_r - beta * mean_R

    # 3. 计算残差
    eps_am = r_am - (alpha + beta * R_am)
    eps_pm = r_pm - (alpha + beta * R_pm)

    # 4. 计算残差差值 delta
    delta = eps_am - eps_pm
    
    # 5. 构造 T 统计量 stat
    mean_delta = np.mean(delta)
    std_delta = np.std(delta)
    
    if std_delta < 1e-12:
        return 0.0
        
    stat = mean_delta / (std_delta / np.sqrt(n))
    return stat

def _apm_wrapper(row: dict) -> float:
    # 接收 Polars Struct，解包传入 Numba
    return calc_apm_stat_core(
        np.array(row['r_am'], dtype=np.float64),
        np.array(row['r_pm'], dtype=np.float64),
        np.array(row['R_am'], dtype=np.float64),
        np.array(row['R_pm'], dtype=np.float64)
    )

# =============================================================================
# Polars 因子逻辑
# =============================================================================
class APMFactor(BaseFactor):
    def __init__(self):
        super().__init__(factor_name="HF_APM")

    def _compute_logic(self, year: str) -> pl.LazyFrame:
        # 1. 底表与宏观过滤
        lf_base = self._get_clean_universe(year, require_intraday=True)
        lf_master = DataLoader.build_daily_master(year).select(['symbol', 'trade_date', 'close'])
        
        # 2. 加载分钟级数据与 60 min 沪深300指数数据
        lf_min = DataLoader.load_minute_data(year=year).with_columns([
            pl.date(pl.col('year').cast(pl.Int32), pl.col('month').cast(pl.Int32), pl.col('day').cast(pl.Int32)).alias('trade_date')
        ])
        # 假设 60min 数据包含 09:30-10:30, 10:30-11:30 (AM close) 以及 13:00-14:00, 14:00-15:00 (PM close)
        # 用通配符加载并统一日期格式
        lf_index = pl.scan_parquet(str(Config.PATH_INDEX_300 / f"*_{year}.parquet")).with_columns([
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d")
        ])

        # 3. 提取个股与指数的上午、下午收益率
        # 微秒级别定义：11:30 对应 41400000000
        AM_END_TIME = 41400000000 
        
        # -- 个股上下午收益率 --
        lf_min_clean = lf_min.join(lf_base.select(['symbol', 'trade_date']), on=['symbol', 'trade_date'], how='inner').sort(['symbol', 'trade_date', 'time'])
        
        lf_stock_ret = lf_min_clean.group_by(['symbol', 'trade_date']).agg([
            # AM = (11:30 收盘 / 09:30 开盘) - 1
            (pl.col('close').filter(pl.col('time') <= AM_END_TIME).last() / 
             pl.col('open').filter(pl.col('time') <= AM_END_TIME).first() - 1.0).alias('r_am'),
            
            # PM = (15:00 收盘 / 13:00 开盘) - 1
            (pl.col('close').filter(pl.col('time') > AM_END_TIME).last() / 
             pl.col('open').filter(pl.col('time') > AM_END_TIME).first() - 1.0).alias('r_pm')
        ])

        # -- 沪深300 上下午收益率 --
        lf_idx_sorted = lf_index.sort(['trade_date', 'time'])
        lf_idx_ret = lf_idx_sorted.group_by(['trade_date']).agg([
            (pl.col('close').filter(pl.col('time') <= AM_END_TIME).last() / 
             pl.col('open').filter(pl.col('time') <= AM_END_TIME).first() - 1.0).alias('R_am'),
            
            (pl.col('close').filter(pl.col('time') > AM_END_TIME).last() / 
             pl.col('open').filter(pl.col('time') > AM_END_TIME).first() - 1.0).alias('R_pm')
        ])

        # 4. 合并数据，计算 Ret20（截面回归需要）
        lf_combined = lf_stock_ret.join(lf_idx_ret, on='trade_date', how='inner').join(lf_master, on=['symbol', 'trade_date'], how='left')
        
        lf_combined = lf_combined.sort(["symbol", "trade_date"]).with_columns([
            (pl.col("close") / pl.col("close").shift(20).over("symbol") - 1.0).alias("Ret20")
        ]).drop_nulls(['r_am', 'r_pm', 'R_am', 'R_pm'])

        # 5. 滑动 20 天，打包成 Struct 执行 Numba 时序回归
        lf_roll = lf_combined.group_by_dynamic(
            "trade_date", by="symbol", every="1d", period="20d", closed="right"
        ).agg([
            pl.col('r_am'), pl.col('r_pm'), pl.col('R_am'), pl.col('R_pm'),
            pl.col('Ret20').last().alias('Ret20_current'), # 当前截面的 Ret20
            pl.col('trade_date').len().alias('valid_days')
        ]).filter(pl.col('valid_days') >= 15)

        lf_stat = lf_roll.with_columns([
            pl.struct(['r_am', 'r_pm', 'R_am', 'R_pm']).map_elements(
                _apm_wrapper, return_dtype=pl.Float64
            ).alias("stat")
        ]).drop_nulls(["stat", "Ret20_current"])

        # 6. 纯向量化截面回归剥离动量 (Cross-Sectional OLS: stat ~ Ret20)
        lf_beta = lf_stat.group_by("trade_date").agg([
            pl.cov("stat", "Ret20_current").alias("cov_xy"),
            pl.col("Ret20_current").var().alias("var_x"),
            pl.col("Ret20_current").mean().alias("mean_x"),
            pl.col("stat").mean().alias("mean_y")
        ]).with_columns([
            (pl.col("cov_xy") / (pl.col("var_x") + 1e-9)).alias("beta")
        ]).with_columns([
            (pl.col("mean_y") - pl.col("beta") * pl.col("mean_x")).alias("alpha")
        ])

        # 最终残差就是 APM 因子
        lf_final = lf_stat.join(lf_beta, on="trade_date").with_columns([
            (pl.col("stat") - pl.col("alpha") - pl.col("beta") * pl.col("Ret20_current")).alias(self.factor_name)
        ])

        return lf_final.select(['trade_date', 'symbol', self.factor_name])
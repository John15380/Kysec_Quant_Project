import polars as pl
from factors.base import BaseFactor
from core.data_loader import DataLoader

class SizeMoneyFlowFactor(BaseFactor):
    """
    大小单资金流因子 (包含剥离动量的截面正交化)
    """
    def __init__(self):
        # 资金流拆分为两个子因子返回，基类这里命名设为前缀即可
        super().__init__(factor_name="HF_SizeMoneyFlow")

    def _compute_logic(self, year: str) -> pl.LazyFrame:
        lf_base = self._get_clean_universe(year, require_intraday=False)
        lf_mf = DataLoader.load_money_flow(year)
        
        lf_merged = lf_base.join(lf_mf, on=['symbol', 'trade_date'], how='inner')
        
        # 1. 计算小单与大单的净流向
        lf_calc = lf_merged.with_columns([
            (pl.col("buy_sm_amount") - pl.col("sell_sm_amount")).alias("net_sm"),
            (pl.col("buy_lg_amount") - pl.col("sell_lg_amount")).alias("net_lg")
        ]).with_columns([
            pl.col("net_sm").abs().alias("abs_sm"),
            pl.col("net_lg").abs().alias("abs_lg")
        ])

        # 2. 滚动 20 日求和，并计算 20 日收益率 Ret20
        lf_roll = lf_calc.sort(["symbol", "trade_date"]).with_columns([
            pl.col("net_sm").rolling_sum(20).over("symbol").alias("sum_net_sm"),
            pl.col("abs_sm").rolling_sum(20).over("symbol").alias("sum_abs_sm"),
            
            pl.col("net_lg").rolling_sum(20).over("symbol").alias("sum_net_lg"),
            pl.col("abs_lg").rolling_sum(20).over("symbol").alias("sum_abs_lg"),
            
            (pl.col("close") / pl.col("close").shift(20).over("symbol") - 1.0).alias("ret20")
        ])

        # 3. 计算资金流强度 Intensity
        lf_intensity = lf_roll.with_columns([
            (pl.col("sum_net_sm") / (pl.col("sum_abs_sm") + 1e-9)).alias("intensity_sm"),
            (pl.col("sum_net_lg") / (pl.col("sum_abs_lg") + 1e-9)).alias("intensity_lg")
        ]).drop_nulls(["ret20", "intensity_sm", "intensity_lg"])

        # -------------------------------------------------------------
        # 4. 极致性能大招：全向量化截面 OLS 线性回归正交化
        # Beta = Cov(X, Y) / Var(X)
        # Alpha = Mean(Y) - Beta * Mean(X)
        # -------------------------------------------------------------
        lf_beta = lf_intensity.group_by("trade_date").agg([
            pl.cov("ret20", "intensity_sm").alias("cov_sm"),
            pl.cov("ret20", "intensity_lg").alias("cov_lg"),
            pl.col("ret20").var().alias("var_x"),
            pl.col("ret20").mean().alias("mean_x"),
            pl.col("intensity_sm").mean().alias("mean_y_sm"),
            pl.col("intensity_lg").mean().alias("mean_y_lg")
        ]).with_columns([
            (pl.col("cov_sm") / (pl.col("var_x") + 1e-9)).alias("beta_sm"),
            (pl.col("cov_lg") / (pl.col("var_x") + 1e-9)).alias("beta_lg")
        ]).with_columns([
            (pl.col("mean_y_sm") - pl.col("beta_sm") * pl.col("mean_x")).alias("alpha_sm"),
            (pl.col("mean_y_lg") - pl.col("beta_lg") * pl.col("mean_x")).alias("alpha_lg")
        ])

        # 5. 组合系数，求出残差 (Residual) 作为纯净的资金流因子
        lf_final = lf_intensity.join(lf_beta, on="trade_date").with_columns([
            (pl.col("intensity_sm") - pl.col("alpha_sm") - pl.col("beta_sm") * pl.col("ret20")).alias("HF_SmallOrderFlow"),
            (pl.col("intensity_lg") - pl.col("alpha_lg") - pl.col("beta_lg") * pl.col("ret20")).alias("HF_LargeOrderFlow")
        ])

        return lf_final.select(['trade_date', 'symbol', 'HF_SmallOrderFlow', 'HF_LargeOrderFlow'])

class ActiveBuyFactor(BaseFactor):
    """
    主动买卖因子 (复现开源证券量化评论36)
    依赖数据: 日级量价、每日资金流向 (小单、中单、大单、特大单)
    """
    def __init__(self):
        super().__init__(factor_name="HF_ActiveBuy")

    def _compute_logic(self, year: str) -> pl.LazyFrame:
        # 1. 获取过滤后的纯净日历与行情底表 (不需要分钟数据，日频资金流即可)
        lf_base = self._get_clean_universe(year, require_intraday=False)
        
        # 2. 加载当年的资金流数据
        lf_mf = DataLoader.load_money_flow(year)
        
        # 3. 将底表与资金流合并
        lf_merged = lf_base.join(lf_mf, on=['symbol', 'trade_date'], how='inner')
        
        # 4. 根据研报公式，计算每日的大中单主动买卖压 (特大单+大单+中单)
        lf_factor = lf_merged.with_columns([
            (
                pl.col('buy_elg_amount') + pl.col('buy_lg_amount') + pl.col('buy_md_amount')
            ).alias('big_buy_amt'),
            
            (
                pl.col('sell_elg_amount') + pl.col('sell_lg_amount') + pl.col('sell_md_amount')
            ).alias('big_sell_amt')
        ])
        
        # 5. 计算当日主动买卖因子 = (买入 - 卖出) / (买入 + 卖出)
        lf_factor = lf_factor.with_columns([
            (
                (pl.col('big_buy_amt') - pl.col('big_sell_amt')) / 
                (pl.col('big_buy_amt') + pl.col('big_sell_amt') + 1e-9) # 加极小数防除0
            ).alias('daily_act_big')
        ])
        
        # 6. 计算 20 日的平滑/累加特征 (使用 Polars 的窗口函数 over)
        # 这里为了稳定性和降低换手，取过去 20 日主动买卖强度的均值作为最终因子
        lf_result = lf_factor.sort(['symbol', 'trade_date']).with_columns([
            pl.col('daily_act_big').rolling_mean(window_size=20, min_periods=10)
            .over('symbol')
            .alias(self.factor_name)
        ])
        
        # 返回基类要求的格式
        return lf_result.select(['trade_date', 'symbol', self.factor_name])
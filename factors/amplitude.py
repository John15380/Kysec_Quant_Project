import polars as pl
from factors.base import BaseFactor

class IdealAmplitudeFactor(BaseFactor):
    """
    理想振幅因子 (开源量化评论36)
    逻辑：过去20个交易日，按收盘价排序，(高价区25%振幅均值) - (低价区25%振幅均值)
    """
    def __init__(self):
        super().__init__(factor_name="HF_IdealAmplitude")

    def _compute_logic(self, year: str) -> pl.LazyFrame:
        lf_base = self._get_clean_universe(year, require_intraday=False)
        
        # 1. 计算每日振幅
        lf_calc = lf_base.with_columns([
            (pl.col("high") / pl.col("low") - 1.0).alias("amp")
        ]).drop_nulls(["close", "amp"])

        # 2. 将数据打包成 Struct，以便在滚动窗口中关联排序
        lf_struct = lf_calc.with_columns([
            pl.struct(["close", "amp"]).alias("daily_data")
        ])

        # 3. 核心大招：动态窗口分组 (Dynamic GroupBy)
        # 按照交易日和股票进行 20 天的滑动窗口划分
        lf_roll = lf_struct.group_by_dynamic(
            "trade_date", 
            by="symbol", 
            every="1d", 
            period="20d", 
            closed="right"
        ).agg([
            pl.col("daily_data"),
            pl.len().alias("window_len")
        ]).filter(pl.col("window_len") >= 20) # 剔除窗口不足20天的初期数据

        # 4. 列表级解析 (list.eval)：全 Rust 环境内完成排序和切片
        lf_res = lf_roll.with_columns([
            # 在列表内部，按照 close 字段降序排列
            pl.col("daily_data").list.eval(
                pl.element().sort_by("close", descending=True)
            ).alias("sorted_data")
        ]).with_columns([
            # 取前 5 天 (高价区)，提取 amp 并求均值
            pl.col("sorted_data").list.head(5).list.eval(
                pl.element().struct.field("amp")
            ).list.mean().alias("amp_high"),
            
            # 取后 5 天 (低价区)，提取 amp 并求均值
            pl.col("sorted_data").list.tail(5).list.eval(
                pl.element().struct.field("amp")
            ).list.mean().alias("amp_low")
        ])

        # 5. 组装最终因子
        lf_final = lf_res.with_columns([
            (pl.col("amp_high") - pl.col("amp_low")).alias(self.factor_name)
        ])
        return lf_final.select(['trade_date', 'symbol', self.factor_name])


class LongTermMomFactor(BaseFactor):
    """
    长端动量因子 (开源量化评论36)
    逻辑：过去160个交易日，按振幅升序排列，取振幅最小的70%天数的涨跌幅加总。
    """
    def __init__(self):
        super().__init__(factor_name="HF_LongTermMom")

    def _compute_logic(self, year: str) -> pl.LazyFrame:
        lf_base = self._get_clean_universe(year, require_intraday=False)
        
        lf_calc = lf_base.with_columns([
            (pl.col("high") / pl.col("low") - 1.0).alias("amp")
        ]).drop_nulls(["amp", "pct_chg"])

        lf_struct = lf_calc.with_columns([
            pl.struct(["amp", "pct_chg"]).alias("daily_data")
        ])

        # 长周期 160 天的动态窗口
        lf_roll = lf_struct.group_by_dynamic(
            "trade_date", by="symbol", every="1d", period="160d", closed="right"
        ).agg([
            pl.col("daily_data"),
            pl.len().alias("window_len")
        ]).filter(pl.col("window_len") >= 160)

        lf_res = lf_roll.with_columns([
            # 按振幅 (amp) 升序排列
            pl.col("daily_data").list.eval(
                pl.element().sort_by("amp", descending=False)
            ).alias("sorted_data")
        ]).with_columns([
            # 160 * 70% = 112 天。取前112天，提取涨跌幅并求和
            pl.col("sorted_data").list.head(112).list.eval(
                pl.element().struct.field("pct_chg")
            ).list.sum().alias(self.factor_name)
        ])

        return lf_res.select(['trade_date', 'symbol', self.factor_name])
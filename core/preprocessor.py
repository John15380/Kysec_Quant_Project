import polars as pl
from typing import List

class Preprocessor:
    """
    因子截面预处理模块 (工业级纯向量化实现)
    功能: MAD去极值、Z-Score标准化
    特性: 严格限制在 `trade_date` 截面上操作，杜绝未来函数。
    """

    @staticmethod
    def _mad_clip(expr: pl.Expr, n: float = 5.0) -> pl.Expr:
        """
        MAD 去极值核心算子
        计算逻辑：
        1. 找出截面中位数 Median
        2. 计算绝对偏差的中位数 MAD = Median(|X - Median|)
        3. 阈值设定为 [Median - n*MAD, Median + n*MAD]
        """
        # 1. 计算中位数
        median = expr.median()
        
        # 2. 计算绝对偏差的中位数 (MAD)
        abs_dev = (expr - median).abs()
        mad = abs_dev.median()
        
        # 3. 容错处理：如果 MAD 为 0，设为极小数，防止后续上下限重叠
        safe_mad = pl.when(mad == 0).then(1e-9).otherwise(mad)
        
        # 4. 计算上下限
        lower_bound = median - n * safe_mad
        upper_bound = median + n * safe_mad
        
        # 5. 截断极值 (clip)
        return expr.clip(lower_bound=lower_bound, upper_bound=upper_bound)

    @staticmethod
    def _z_score(expr: pl.Expr) -> pl.Expr:
        """
        Z-Score 标准化核心算子
        计算逻辑： (X - Mean) / Std
        """
        mean = expr.mean()
        std = expr.std()
        
        # 容错处理：如果标准差为0（截面全为相同值），则分母设为极小数
        safe_std = pl.when(std == 0).then(1e-9).otherwise(std)
        
        return (expr - mean) / safe_std

    @classmethod
    def process_cross_section(cls, lf: pl.LazyFrame, factor_cols: List[str]) -> pl.LazyFrame:
        """
        执行完整的截面处理流水线：先去极值，再标准化。
        使用 Polars 的 `over` 语法，确保所有统计量严格在每日截面内计算。
        """
        exprs = []
        for col in factor_cols:
            # 嵌套调用：原值 -> MAD去极值 -> Z-Score标准化 -> 在 trade_date 上聚合
            processed_expr = (
                cls._z_score(cls._mad_clip(pl.col(col), n=5.0))
                .over("trade_date")  # <--- 核心：圈定截面范围
                .alias(col)
            )
            exprs.append(processed_expr)
            
        # 并发执行所有因子的特征工程
        return lf.with_columns(exprs)
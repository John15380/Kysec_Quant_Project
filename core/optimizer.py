import numpy as np
import polars as pl

class ICIROptimizer:
    """
    最大化 ICIR 复合因子优化器 (解析解 + 正则化)
    """
    def __init__(self, rolling_window: int = 12, reg_lambda: float = 0.01):
        """
        :param rolling_window: 滚动回看期数 (研报中为 12 期)
        :param reg_lambda: 正则化强度 (Shrinkage)，防止协方差矩阵求逆时过度放缩
        """
        self.rolling_window = rolling_window
        self.reg_lambda = reg_lambda

    def _calc_optimal_weights(self, ic_matrix: np.ndarray) -> np.ndarray:
        """
        核心数学算子：通过解析解计算最大化 ICIR 的因子权重
        :param ic_matrix: 形状为 (N_periods, N_factors) 的历史 IC 矩阵
        :return: 形状为 (N_factors,) 的权重向量
        """
        # 1. 计算历史 IC 的均值向量 (mu)
        mu = np.mean(ic_matrix, axis=0)
        
        # 2. 计算历史 IC 的协方差矩阵 (Sigma)
        # rowvar=False 表示每一列代表一个因子
        sigma = np.cov(ic_matrix, rowvar=False)
        
        # 3. 工业级细节：矩阵正则化 (Tikhonov Regularization / Ledoit-Wolf Shrinkage 简化版)
        # 给对角线加上微小的扰动，防止多重共线性导致矩阵不可逆或权重极端
        n_factors = sigma.shape[0]
        sigma_reg = sigma + self.reg_lambda * np.eye(n_factors) * np.trace(sigma) / n_factors
        
        # 4. 求解解析解: w = Sigma^-1 * mu
        try:
            w_opt = np.linalg.solve(sigma_reg, mu)
        except np.linalg.LinAlgError:
            # 极低概率遇到完全奇异矩阵，退化为等权
            w_opt = np.ones(n_factors) / n_factors
            
        # 5. 权重归一化 (截面因子的绝对数值无意义，比例才重要，统一使其绝对值之和为1或单纯除以sum)
        # 如果容许负权重(因子反向)，取绝对值归一化防止整体符号翻转
        w_sum = np.sum(np.abs(w_opt))
        if w_sum > 1e-9:
            w_opt = w_opt / w_sum
            
        return w_opt

    def optimize_and_composite(self, lf_factors: pl.LazyFrame, lf_ic_series: pl.LazyFrame, factor_cols: list) -> pl.LazyFrame:
        """
        执行滚动优化，并计算出当期的复合因子
        
        :param lf_factors: 包含预处理后因子的 LazyFrame (必须含 trade_date, symbol)
        :param lf_ic_series: 每期计算好的 IC 序列表 (必须含 trade_date 和各 factor_cols 的 IC 值)
        :param factor_cols: 参与复合的因子列名列表
        :return: 带有 `Composite_Factor` 列的 LazyFrame
        """
        # 1. 把 IC 序列转化为时间序列矩阵
        df_ic = lf_ic_series.select(["trade_date"] + factor_cols).sort("trade_date").collect()
        
        dates = df_ic["trade_date"].to_list()
        ic_data = df_ic.select(factor_cols).to_numpy()
        
        n_periods = len(dates)
        
        # 2. 滚动计算每期的最优权重
        # 记录每期计算出的权重 (第 t 期使用的权重，来自 t-window 到 t-1 期的 IC 优化)
        weight_records = []
        
        for i in range(n_periods):
            curr_date = dates[i]
            
            # 如果历史数据不足 12 期，采用等权过渡
            if i < self.rolling_window:
                w = np.ones(len(factor_cols)) / len(factor_cols)
            else:
                # 截取过去 12 期的 IC 数据
                hist_ic = ic_data[i - self.rolling_window : i, :]
                w = self._calc_optimal_weights(hist_ic)
                
            # 保存日期和对应的权重字典
            record = {"trade_date": curr_date}
            for j, col in enumerate(factor_cols):
                record[f"w_{col}"] = w[j]
            weight_records.append(record)
            
        # 3. 将权重转换为 DataFrame
        df_weights = pl.DataFrame(weight_records).lazy()
        
        # 4. 把权重 Join 到全量的因子截面上
        lf_merged = lf_factors.join(df_weights, on="trade_date", how="left")
        
        # 5. 向量化计算复合因子: Sum(Factor_i * Weight_i)
        composite_expr = pl.sum_horizontal([
            pl.col(col) * pl.col(f"w_{col}") for col in factor_cols
        ])
        
        lf_final = lf_merged.with_columns([
            composite_expr.alias("Composite_Factor")
        ])
        
        # 可以选择 drop 掉那些中间的权重列，保持表结构干净
        drop_cols = [f"w_{col}" for col in factor_cols]
        return lf_final.drop(drop_cols)
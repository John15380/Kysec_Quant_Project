import polars as pl
import numpy as np
import pandas as pd

class Evaluator:
    """
    因子评估模块：计算 IC, 多空收益, 年化夏普, 最大回撤等
    """
    @staticmethod
    def calc_ic_series(lf_factors: pl.LazyFrame, lf_forward_returns: pl.LazyFrame, factor_col: str, ret_col: str) -> pl.DataFrame:
        """
        计算因子的每日 IC 和 Rank IC 序列
        :param lf_forward_returns: 包含前瞻收益率的表 (例如未来5天收益率)
        """
        lf_merged = lf_factors.join(lf_forward_returns, on=["trade_date", "symbol"], how="inner").drop_nulls([factor_col, ret_col])
        
        # 每日截面计算 IC
        df_ic = lf_merged.group_by("trade_date").agg([
            pl.corr(factor_col, ret_col, method="pearson").alias("IC"),
            pl.corr(factor_col, ret_col, method="spearman").alias("Rank_IC")
        ]).sort("trade_date").collect()
        
        return df_ic

    @staticmethod
    def calc_performance_metrics(df_group_ret: pl.DataFrame, df_turnover: pl.DataFrame, factor_name: str, freq_name: str, is_ascending: bool = True):
        """
        根据每日分组收益计算最终的评价指标 (复现研报表格)
        :param is_ascending: 因子是否为升序排序 (即第5组是多头，还是第1组是多头)
        """
        # 1. 确定多空组别
        # 根据研报：如果是负向因子(如聪明钱越小越好)，多头是第1组；否则是第5组。
        long_col = "5" if is_ascending else "1"
        short_col = "1" if is_ascending else "5"
        
        df_pd = df_group_ret.to_pandas().set_index("trade_date")
        df_pd.fillna(0.0, inplace=True)
        
        # 多头收益和多空(Long-Short)收益
        ret_long = df_pd[long_col]
        ret_ls = df_pd[long_col] - df_pd[short_col]

        # 2. 计算年化指标 (假设一年250个交易日)
        # 年化收益率 (复利计算)
        nav_ls = (1 + ret_ls).prod()
        years = len(df_pd) / 250.0
        ann_ret_ls = nav_ls ** (1 / years) - 1 if years > 0 else 0

        nav_long = (1 + ret_long).prod()
        ann_ret_long = nav_long ** (1 / years) - 1 if years > 0 else 0

        # 年化波动率
        ann_vol_ls = ret_ls.std() * np.sqrt(250)
        ann_vol_long = ret_long.std() * np.sqrt(250)

        # 收益波动比 (研报使用的指标，相当于无风险利率为0的夏普)
        sharpe_ls = ann_ret_ls / ann_vol_ls if ann_vol_ls > 0 else 0
        sharpe_long = ann_ret_long / ann_vol_long if ann_vol_long > 0 else 0

        # 最大回撤
        cum_nav_ls = (1 + ret_ls).cumprod()
        drawdown_ls = cum_nav_ls / cum_nav_ls.cummax() - 1
        max_dd_ls = drawdown_ls.min()

        # 月均换手率计算
        # 先把单边换手率转成 pandas，按月重采样求和
        df_turn = df_turnover.to_pandas().set_index("trade_date")
        df_turn.index = pd.to_datetime(df_turn.index) # 增加此行确保索引是 datetime 类型
        # 乘以 2 因为研报可能计算的是单边，有些机构看双边，此处以单边累加计入月度
        monthly_turnover = df_turn.resample('M')['turnover'].sum().mean()

        # 3. 打印精美的输出结果
        print(f"\n{'='*40}")
        print(f" 因子: {factor_name} | 测试频率: {freq_name}")
        print(f"{'='*40}")
        print(f"【多空对冲 (Long-Short)】")
        print(f"  年化收益率 : {ann_ret_ls:.2%}")
        print(f"  收益波动比 : {sharpe_ls:.2f}")
        print(f"  最大回撤   : {max_dd_ls:.2%}")
        print(f"【纯多头 (Long Only)】")
        print(f"  年化收益率 : {ann_ret_long:.2%}")
        print(f"  收益波动比 : {sharpe_long:.2f}")
        print(f"  月均换手率 : {monthly_turnover:.1%}")
        print(f"{'='*40}\n")
        
        # 返回净值序列用于后续画图 (Jupyter Notebook 中调用)
        return {
            "nav_groups": (1 + df_pd).cumprod(),
            "nav_ls": cum_nav_ls
        }
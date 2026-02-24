import polars as pl
import numpy as np
from config.settings import Config

class PortfolioEngine:
    """
    量化组合构建与回测推演引擎
    处理分组（5组）、换仓频率控制、扣费以及净值推演。
    """
    def __init__(self, groups: int = 5, cost_rate: float = 0.003):
        self.groups = groups
        self.cost_rate = cost_rate  # 研报设定的双边千三费率

    def run_backtest(self, lf_factors: pl.LazyFrame, lf_daily_returns: pl.LazyFrame, factor_col: str, rebal_days: int) -> pl.DataFrame:
        """
        :param lf_factors: 包含 [trade_date, symbol, factor_value] 的 LazyFrame
        :param lf_daily_returns: 包含 [trade_date, symbol, daily_ret] 的日收益率表 (T日收盘相对于T-1日收盘)
        :param factor_col: 要回测的因子列名
        :param rebal_days: 调仓频率 (5=周频, 10=双周频, 20=月频)
        """
        # 1. 提取所有交易日并按调仓频率切片
        # 取出所有有因子值的日期
        df_dates = lf_factors.select("trade_date").unique().sort("trade_date").collect()
        all_dates = df_dates["trade_date"].to_list()
        
        # 找出调仓日 (每隔 rebal_days 天换一次仓)
        rebal_dates = all_dates[::rebal_days]
        df_rebal_dates = pl.DataFrame({"trade_date": rebal_dates, "is_rebal": True}).lazy()

        # 2. 仅在调仓日计算分组 (1~5组)
        # 等权分组：利用 qcut
        lf_rebal_factors = lf_factors.join(df_rebal_dates, on="trade_date", how="inner").drop_nulls([factor_col])
        
        lf_groups = lf_rebal_factors.with_columns([
            pl.col(factor_col).qcut(self.groups, labels=[str(i) for i in range(1, self.groups + 1)], allow_duplicates=True).alias("group")
        ]).select(["trade_date", "symbol", "group"])

        # 3. 生成持仓区间并向前填充 (Forward Fill)
        # 建立一张 [所有交易日 x 所有调仓日出现的股票] 的底表有点大
        # 工业界更优雅的做法是：用 asof join 把每一天的真实收益，匹配到最近的一个调仓日的 group 标签上
        lf_daily_merged = lf_daily_returns.join_asof(
            lf_groups.sort("trade_date"),
            on="trade_date",
            by="symbol",
            strategy="backward"
        ).drop_nulls("group") # 如果某些股票在调仓日没被选中，就会是 null，直接丢弃

        # 4. 计算各组的每日等权收益率
        # 注意：这里还没有扣费
        lf_group_daily_ret = lf_daily_merged.group_by(["trade_date", "group"]).agg([
            pl.col("daily_ret").mean().alias("raw_ret")
        ])

        # 5. 精确计算换手率与交易成本
        # 计算逻辑：在调仓日，检查这期持仓与上期持仓的差异
        # 转化为宽表，行是 date，列是 symbol 的 group 状态
        df_group_status = lf_groups.collect().pivot(
            index="trade_date", columns="symbol", values="group", aggregate_function="first"
        ).sort("trade_date")

        # 将 dataframe 转为 numpy 矩阵计算换手率极快
        status_matrix = df_group_status.drop("trade_date").to_numpy()
        rebal_dates_arr = df_group_status["trade_date"].to_list()
        
        turnover_records = []
        for i in range(len(rebal_dates_arr)):
            curr_date = rebal_dates_arr[i]
            if i == 0:
                # 初始建仓，换手率为 100% (这里用 1.0 表示单边换手)
                turnover_records.append({"trade_date": curr_date, "turnover": 1.0})
                continue
            
            # 统计各组的换手 (简化的等权换手率近似：1 - 重合股票数 / 本期股票数)
            prev_status = status_matrix[i-1]
            curr_status = status_matrix[i]
            
            # 计算总换手率 (近似为全市场发生变化的比例)
            # 更精确的是按组算，但为了整体扣费，我们计算一个总的单边换手率
            valid_prev = prev_status != None
            valid_curr = curr_status != None
            overlap = np.sum((prev_status == curr_status) & valid_curr)
            total_curr = np.sum(valid_curr)
            
            turnover = 1.0 - (overlap / total_curr) if total_curr > 0 else 0.0
            turnover_records.append({"trade_date": curr_date, "turnover": turnover})

        df_turnover = pl.DataFrame(turnover_records).lazy()

        # 6. 合并收益与扣费
        lf_net_ret = lf_group_daily_ret.join(df_turnover, on="trade_date", how="left").with_columns([
            pl.col("turnover").fill_null(0.0) # 非调仓日换手为 0
        ]).with_columns([
            # 扣除交易成本: 净收益 = 原始收益 - 换手率 * 费率
            (pl.col("raw_ret") - pl.col("turnover") * self.cost_rate).alias("net_ret")
        ]).sort(["group", "trade_date"])

        # 7. 转宽表输出最终各组每日净收益率
        df_final_ret = lf_net_ret.collect().pivot(
            index="trade_date", columns="group", values="net_ret", aggregate_function="first"
        ).sort("trade_date")
        
        # 顺便把总换手率序列返回，供评估模块使用
        df_turnover_out = df_turnover.collect()
        
        return df_final_ret, df_turnover_out
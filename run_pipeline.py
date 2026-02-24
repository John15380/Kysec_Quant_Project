import polars as pl
import os
from config.settings import Config
from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.optimizer import ICIROptimizer
from backtest.portfolio import PortfolioEngine
from backtest.evaluator import Evaluator

# 导入我们编写的所有专业因子
from factors.smart_money import SmartMoneyFactor
from factors.amplitude import IdealAmplitudeFactor, LongTermMomFactor
from factors.money_flow import SizeMoneyFlowFactor, ActiveBuyFactor
from factors.apm import APMFactor

def run_factor_computation(years: list[str]):
    """第一阶段：全量因子计算与缓存落盘"""
    print("=" * 50)
    print("🚀 阶段一：启动底层因子并行计算流水线")
    print("=" * 50)
    
    factors = [
        SmartMoneyFactor(),
        IdealAmplitudeFactor(),
        LongTermMomFactor(),
        SizeMoneyFlowFactor(),
        ActiveBuyFactor(),
        APMFactor()
    ]
    
    for factor in factors:
        # build_and_cache 内部有缓存检测，已计算过的会自动跳过
        factor.build_and_cache(years)

def build_forward_returns(years: list[str]) -> pl.LazyFrame:
    """计算用于 IC 和回测的前瞻/日收益率"""
    lf_list = []
    for y in years:
        lf_daily = DataLoader.build_daily_master(y).select(['trade_date', 'symbol', 'close'])
        lf_list.append(lf_daily)
        
    lf_master = pl.concat(lf_list).sort(['symbol', 'trade_date'])
    
    # 计算 T 期的日收益率 (供组合引擎推演净值使用)
    # 计算 T+1 到 T+5 的未来 5 日收益率 (供计算周频 IC 使用)
    lf_ret = lf_master.with_columns([
        (pl.col('close').shift(-1).over('symbol') / pl.col('close') - 1.0).alias('daily_ret'),
        (pl.col('close').shift(-5).over('symbol') / pl.col('close') - 1.0).alias('fwd_ret_5d')
    ])
    return lf_ret

def main():
    # 研报回测区间：2014-01 至 2021-09
    test_years = ["2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    
    # ---------------------------------------------------------
    # Step 1: 因子计算与落盘 (极速多线程)
    # ---------------------------------------------------------
    run_factor_computation(test_years)
    
    # ---------------------------------------------------------
    # Step 2: 因子读取与大宽表组装
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("📊 阶段二：组装截面特征大宽表并进行预处理")
    print("=" * 50)
    
    factor_cols = [
        "HF_SmartMoney", "HF_IdealAmplitude", "HF_LongTermMom", 
        "HF_SmallOrderFlow", "HF_LargeOrderFlow", "HF_ActiveBuy", "HF_APM"
    ]
    
    # 读取底表 (取第一支因子的结构作为骨架)
    lf_combined = SmartMoneyFactor().load().select(['trade_date', 'symbol'])
    
    # 动态 Join 所有因子
    for factor in [SmartMoneyFactor(), IdealAmplitudeFactor(), LongTermMomFactor(), SizeMoneyFlowFactor(), ActiveBuyFactor(), APMFactor()]:
        lf_combined = lf_combined.join(factor.load(), on=['trade_date', 'symbol'], how='left')

    # ---------------------------------------------------------
    # Step 3: 严格的截面预处理 (MAD 去极值 + Z-Score)
    # ---------------------------------------------------------
    # 这一步极其关键，确保所有因子的量纲一致，且不受极端值干扰
    lf_processed = Preprocessor.process_cross_section(lf_combined, factor_cols).drop_nulls()

    # ---------------------------------------------------------
    # Step 4: 合成复合因子 (最大化 ICIR 优化器)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("🧠 阶段三：计算 IC 序列与复合因子优化 (解析解)")
    print("=" * 50)
    
    lf_returns = build_forward_returns(test_years)
    
    # 为了优化器，我们需要先算一遍所有单因子的 IC
    ic_series_list = []
    for col in factor_cols:
        df_ic = Evaluator.calc_ic_series(lf_processed, lf_returns, factor_col=col, ret_col='fwd_ret_5d')
        df_ic = df_ic.rename({"IC": col}).drop("Rank_IC") # 只保留 IC 值并重命名为因子名
        ic_series_list.append(df_ic.lazy())
    
    # 拼接全量 IC 序列大表
    lf_ic_matrix = ic_series_list[0]
    for lf_ic in ic_series_list[1:]:
        lf_ic_matrix = lf_ic_matrix.join(lf_ic, on="trade_date", how="inner")

    # 启动优化器：滚动回看 12 期，利用协方差矩阵解析解算出最优权重
    optimizer = ICIROptimizer(rolling_window=12, reg_lambda=0.01)
    lf_final_factors = optimizer.optimize_and_composite(
        lf_factors=lf_processed, 
        lf_ic_series=lf_ic_matrix, 
        factor_cols=factor_cols
    )

    # ---------------------------------------------------------
    # Step 5: 多频段回测与评估 (周频、双周频、月频)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("📈 阶段四：启动组合引擎进行多频段扣费回测")
    print("=" * 50)
    
    # 回测设置：双边千三费率，分5组
    engine = PortfolioEngine(groups=5, cost_rate=0.003)
    
    frequencies = {
        "周频": 5,
        "双周频": 10,
        "月频": 20
    }
    
    results_store = {}
    
    for freq_name, rebal_days in frequencies.items():
        # 执行组合推演 (返回每日各组净收益率表，以及换手率表)
        df_group_ret, df_turnover = engine.run_backtest(
            lf_factors=lf_final_factors, 
            lf_daily_returns=lf_returns, 
            factor_col="Composite_Factor", 
            rebal_days=rebal_days
        )
        
        # 计算并打印评估指标 (多空收益、夏普、最大回撤等)
        metrics = Evaluator.calc_performance_metrics(
            df_group_ret=df_group_ret, 
            df_turnover=df_turnover, 
            factor_name="Composite_Factor (复合因子)", 
            freq_name=freq_name,
            is_ascending=True # 假设复合因子越大越好，多头为第5组
        )
        
        results_store[freq_name] = metrics

    print("\n🎉 全流程运行完毕！净值数据已准备就绪，可供 Jupyter Notebook 绘图使用。")
    # 此处可以将 results_store['周频']['nav_groups'] 等保存为 csv 供绘图

if __name__ == "__main__":
    main()
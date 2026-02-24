import polars as pl
from config.settings import Config
from core.data_loader import DataLoader

class UniverseBuilder:
    @staticmethod
    def build_tradable_mask(year: str) -> pl.LazyFrame:
        """
        构建每日可交易股票池掩码，精确区分全天停牌与日内停牌。
        """
        lf_master = DataLoader.build_daily_master(year)
        
        lf_st = pl.scan_csv(Config.PATH_ST).with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d"),
            pl.lit(True).alias('is_st')
        ])

        # 解析停牌时间
        lf_suspend = pl.scan_csv(Config.PATH_SUSPEND).with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d"),
            # 如果 suspend_timing 为空(null)，说明是全天停牌
            pl.col('suspend_timing').is_null().alias('is_full_suspend'),
            # 如果 suspend_timing 有具体时间段，说明是日内部分停牌
            pl.col('suspend_timing').is_not_null().alias('is_partial_suspend')
        ]).select(['symbol', 'trade_date', 'is_full_suspend', 'is_partial_suspend'])

        lf_list = pl.scan_csv(Config.PATH_LIST_DATE).with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('list_date').str.strptime(pl.Date, "%Y-%m-%d")
        ])

        # 组合条件
        lf_mask = (
            lf_master
            .join(lf_st, on=['symbol', 'trade_date'], how='left')
            .join(lf_suspend, on=['symbol', 'trade_date'], how='left')
            .join(lf_list, on='symbol', how='left')
        ).with_columns([
            pl.col('is_st').fill_null(False),
            pl.col('is_full_suspend').fill_null(False),
            pl.col('is_partial_suspend').fill_null(False),
            
            # 判断是否上市满特定天数
            ((pl.col('trade_date') - pl.col('list_date')).dt.total_days() >= Config.NEW_STOCK_DAYS).alias('is_mature'),
            
            # 一字涨跌停判断 (收盘价等于开/高/低，且达到涨跌停限制)
            (
                (pl.col('high') == pl.col('low')) & 
                ((pl.col('high') - pl.col('up_limit')).abs() < 1e-4)
            ).alias('is_limit_up'),
            
            (
                (pl.col('high') == pl.col('low')) & 
                ((pl.col('low') - pl.col('down_limit')).abs() < 1e-4)
            ).alias('is_limit_down')
        ])

        # 生成最终掩码及细分状态
        lf_tradable = lf_mask.with_columns([
            # 这里的 is_tradable 定义为：非ST、非全天停牌、已度过新股期
            # 注意：一字涨跌停和部分停牌在日频上依然是 tradable 的，但我们在后续会暴露这几个标签供高频因子使用
            (
                ~pl.col('is_st') & 
                ~pl.col('is_full_suspend') & 
                pl.col('is_mature') 
            ).alias('is_tradable')
        ]).select([
            'symbol', 'trade_date', 'is_tradable', 
            'is_limit_up', 'is_limit_down', 'is_partial_suspend'
        ])

        return lf_tradable
import polars as pl
from pathlib import Path
from config.settings import Config

class DataLoader:
    """
    统一数据加载底座 (基于 Polars Lazy API)
    原则：所有输出统一转换为标准的 DataFrame/LazyFrame，规范列名与量纲。
    """
    
    @staticmethod
    def _format_symbol(expr: pl.Expr) -> pl.Expr:
        """
        内部辅助函数：确保股票代码为6位字符串
        """
        return expr.cast(pl.Utf8).str.pad_start(6, '0')

    @staticmethod
    def load_calendar() -> pl.DataFrame:
        """
        加载交易日历并返回排好序的日期 Series
        """
        df = pl.read_csv(Config.PATH_CALENDAR)
        return df.with_columns(
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d")
        ).sort('trade_date')

    @staticmethod
    def load_industry() -> pl.LazyFrame:
        """
        加载行业分类
        """
        return pl.scan_csv(Config.PATH_INDUSTRY).with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('in_date').str.strptime(pl.Date, "%Y-%m-%d")
        ]).sort(['symbol', 'in_date'])

    @staticmethod
    def build_daily_master(year: str) -> pl.LazyFrame:
        """
        核心方法：构建某一年的日频主表，整合行情、市值、复权因子、行业及涨跌停限制。
        量纲统一：成交量(股)，成交额(元)。
        """
        # 1. 基础行情 (处理量纲)
        lf_daily = pl.scan_csv(Config.PATH_DAILY / f'daily_{year}.csv').with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d"),
            (pl.col('vol') * 100).alias('volume'),          # 手 -> 股
            (pl.col('amount') * 1000).alias('amount')       # 千元 -> 元
        ]).drop(['vol'])

        # 2. 涨跌停数据
        lf_limit = pl.scan_csv(Config.PATH_LIMIT / f'A_share_limit_{year}.csv').with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d")
        ])

        # 3. 市值与复权因子
        lf_mv = pl.scan_csv(Config.PATH_MV / f'mv_data_{year}.csv').with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d")
        ])
        
        lf_adj = pl.scan_csv(Config.PATH_ADJ_FACTOR / f'adj_factor_{year}.csv').with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d")
        ])

        # 4. 横向 Join 组装 (Polars 的 join 默认是多线程的，非常快)
        lf_master = (
            lf_daily
            .join(lf_limit, on=['symbol', 'trade_date'], how='left')
            .join(lf_mv, on=['symbol', 'trade_date'], how='left')
            .join(lf_adj, on=['symbol', 'trade_date'], how='left')
        ).sort(['symbol', 'trade_date'])

        # 5. 行业数据 Asof Join (极其重要，防止用到未来数据)
        # 注意：Asof join 要求两表必须基于 on 字段严格排序，且 by 字段匹配
        lf_ind = DataLoader.load_industry()
        lf_master = lf_master.join_asof(
            lf_ind,
            left_on='trade_date',
            right_on='in_date',
            by='symbol',
            strategy='backward'  # 寻找交易日当天或之前最近的一次行业划分
        )

        return lf_master

    @staticmethod
    def load_money_flow(year: str) -> pl.LazyFrame:
        """
        加载资金流向数据
        量纲统一：成交量(股)，成交额(元)。
        """
        lf_mf = pl.scan_parquet(Config.PATH_MONEYFLOW / f'moneyflow_{year}.parquet')
        
        # 获取所有需要转换的列
        vol_cols = [col for col in lf_mf.columns if 'vol' in col]
        amt_cols = [col for col in lf_mf.columns if 'amount' in col]

        # 批量应用表达式进行量纲转换
        exprs = [DataLoader._format_symbol(pl.col('symbol')), 
                 pl.col('trade_date').str.strptime(pl.Date, "%Y-%m-%d")]
        
        exprs += [(pl.col(c) * 100).alias(c) for c in vol_cols]       # 手 -> 股
        exprs += [(pl.col(c) * 10000).alias(c) for c in amt_cols]     # 万元 -> 元

        return lf_mf.with_columns(exprs)

    @staticmethod
    def load_minute_data(year: str = None, date_str: str = None) -> pl.LazyFrame:
        """
        统一的高频分钟数据加载器 (支持单日拉取 / 跨年拉取 / 全年批处理)
        利用 Polars 原生的 Hive 分区读取机制，实现极致的谓词下推 (Predicate Pushdown)。
        """
        if not year and not date_str:
            raise ValueError("必须指定 year 或 date_str")

        # 1. 动态构造基础路径匹配 (避免扫描无关年份，加快构建查询图)
        if date_str:
            date_obj = pl.select(pl.lit(date_str).str.strptime(pl.Date, "%Y-%m-%d")).item()
            y, m, d = f"{date_obj.year}", f"{date_obj.month:02d}", f"{date_obj.day:02d}"
            target_path = str(Config.PATH_MIN_DATA / f"year={y}/month={m}/day={d}/*.parquet")
        else:
            target_path = str(Config.PATH_MIN_DATA / f"year={year}/*/*/*.parquet")

        # 2. 核心大招：开启 hive_partitioning
        # Polars 会自动解析路径中的 year, month, day 作为隐藏列，并开启极速多线程 IO
        lf_min = pl.scan_parquet(target_path, hive_partitioning=True)

        # 3. 统一格式与量纲清洗
        lf_min = lf_min.with_columns([
            DataLoader._format_symbol(pl.col('symbol')),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Float64),
            pl.col('amount').cast(pl.Float64)
        ])

        return lf_min
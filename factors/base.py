import os
import polars as pl
from abc import ABC, abstractmethod
from pathlib import Path
from config.settings import Config
from core.universe import UniverseBuilder
from core.data_loader import DataLoader

class BaseFactor(ABC):
    """
    量化因子基类 (工业级流水线标准)
    所有衍生因子必须继承此类，并实现 _compute_logic 方法。
    """
    def __init__(self, factor_name: str):
        self.factor_name = factor_name
        self.cache_dir = Config.CACHE_DIR / 'factors'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = self.cache_dir / f"{self.factor_name}.parquet"

    def _get_clean_universe(self, year: str, require_intraday: bool = False) -> pl.LazyFrame:
        """
        获取纯净的可交易底层数据（预先剔除脏数据）
        """
        lf_univ = UniverseBuilder.build_tradable_mask(year)
        lf_master = DataLoader.build_daily_master(year)
        
        # 基础过滤：必须是 Tradable
        lf_clean = lf_master.join(lf_univ, on=['symbol', 'trade_date'], how='inner')
        lf_clean = lf_clean.filter(pl.col('is_tradable') == True)
        
        # 如果因子依赖高频数据（如聪明钱、APM），剔除日内部分停牌和一字涨跌停
        if require_intraday:
            lf_clean = lf_clean.filter(
                (~pl.col('is_partial_suspend')) & 
                (~pl.col('is_limit_up')) & 
                (~pl.col('is_limit_down'))
            )
            
        return lf_clean

    @abstractmethod
    def _compute_logic(self, year: str) -> pl.LazyFrame:
        """
        核心数学逻辑：由具体的因子类实现。
        必须返回包含 ['trade_date', 'symbol', self.factor_name] 的 LazyFrame。
        """
        pass

    def run_year(self, year: str):
        """
        执行单年计算并暂存，用于流式处理防内存溢出
        """
        print(f"[{self.factor_name}] 开始计算 {year} 年因子...")
        lf_result = self._compute_logic(year)
        
        # 强制格式校验
        expected_cols = {'trade_date', 'symbol', self.factor_name}
        if not expected_cols.issubset(set(lf_result.columns)):
            raise ValueError(f"[{self.factor_name}] 缺失标准列！应包含: {expected_cols}")
        
        # 选取标准列并计算落盘
        df_result = lf_result.select(['trade_date', 'symbol', self.factor_name]).collect()
        return df_result

    def build_and_cache(self, years: list[str]):
        """
        全量计算并合并落盘（生成最终 Parquet）
        """
        if self.save_path.exists():
            print(f"[{self.factor_name}] 缓存已存在，跳过计算: {self.save_path}")
            return
            
        df_list = []
        for y in years:
            df_year = self.run_year(y)
            df_list.append(df_year)
            
        # 合并所有年份并落盘
        df_final = pl.concat(df_list)
        df_final.write_parquet(self.save_path)
        print(f"[{self.factor_name}] 全量因子合并落盘成功 -> {self.save_path}")

    def load(self) -> pl.LazyFrame:
        """为回测引擎提供极速读取接口"""
        if not self.save_path.exists():
            raise FileNotFoundError(f"未找到 {self.factor_name}，请先执行 build_and_cache()")
        return pl.scan_parquet(self.save_path)
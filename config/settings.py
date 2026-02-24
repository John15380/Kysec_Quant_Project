import os
from pathlib import Path

class Config:
    # --- 基础路径配置 ---
    ROOT_DIR = Path('./data')
    CACHE_DIR = Path('./cache')
    
    # --- 数据文件路径映射 ---
    PATH_ST = ROOT_DIR / 'ST情况/is_st.csv'
    PATH_LIST_DATE = ROOT_DIR / '上市日期/list_date.csv'
    PATH_CALENDAR = ROOT_DIR / '交易日历/days.csv'
    PATH_SUSPEND = ROOT_DIR / '停复牌情况/is_suspend.csv'
    PATH_ADJ_FACTOR = ROOT_DIR / '复权因子'
    PATH_MV = ROOT_DIR / '市值数据'
    PATH_YIELD = ROOT_DIR / '无风险利率/yield.csv'
    PATH_DAILY = ROOT_DIR / '日级数据'
    PATH_LIMIT = ROOT_DIR / '涨跌停价'
    PATH_INDUSTRY = ROOT_DIR / '行业分类/industry.csv'
    PATH_MONEYFLOW = ROOT_DIR / '资金流向'
    PATH_MIN_DATA = ROOT_DIR / '分钟级数据'
    PATH_INDEX_300 = ROOT_DIR / '沪深300-60min'

    # --- 交易过滤参数 ---
    NEW_STOCK_DAYS = 180  # 剔除上市不满180天的新股
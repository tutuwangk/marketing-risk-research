"""Paths, split years, feature lists and constants for the MPDR experiment."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MPDR_DATA_DIR", PROJECT_DIR / "02_data" / "final"))
RESULTS_DIR = Path(os.environ.get("MPDR_RESULTS_DIR", PROJECT_DIR / "04_experiments" / "results"))
LABEL_PATH = Path(os.environ.get("MPDR_LABEL_PATH", DATA_DIR / "mpdr_label.csv"))
RESTATEMENT_PATH = DATA_DIR / "restatement_lookahead_flags.csv"
X_PATH = DATA_DIR / "model_features_X.csv"
Y_PATH = DATA_DIR / "model_target_Y.csv"
DICT_PATH = DATA_DIR / "data_dictionary.csv"

KEY = ["股票代码", "报告年度"]
YEAR = "报告年度"
TARGET = "下一年实际营销绩效"

TRAIN_YEARS = range(2013, 2021)
VALID_YEARS = range(2021, 2023)
TEST_YEARS = range(2023, 2025)

# The 32 numeric candidates in data_dictionary.csv (模型候选 == 数值特征), dictionary order.
RAW_NUMERIC = [
    "实际营销绩效", "销售费用对数", "销售费用率", "销售费用增长率", "营业收入增长率",
    "投入增长同步度", "毛利率", "毛利率变化", "企业规模", "资产负债率", "ROA",
    "经营现金流质量", "经营现金流资产比", "现金持有水平", "应收账款占比", "存货占比",
    "资本支出强度", "存货周转率", "前五大客户占比", "第一大客户占比", "客户集中度",
    "前五大供应商占比", "第一大供应商占比", "供应商集中度", "成立年限", "行业收入增长率",
    "行业毛利率中位数", "行业销售费用率中位数", "GDP实际增长率", "CPI年度涨幅",
    "全国居民人均可支配收入名义增长率", "社会消费品零售总额年度增长率",
]

# Raw inputs used to build derived features.
GP = "毛利润"
SE = "销售费用"
ASSETS = "总资产"
CAPEX = "资本支出强度"
INV_TURNOVER = "存货周转率"
ST = "标记_预测时点ST或*ST"
CONSUMPTION = "消费类型"
INDUSTRY_RAW = "对应目标行业"

# mp_t replaces 实际营销绩效; ln1p_se replaces 销售费用对数.
CONTINUOUS = ["mp_t", "gp_to_assets", "ln1p_se"] + [
    c for c in RAW_NUMERIC if c not in ("实际营销绩效", "销售费用对数")
]
# Year-level variables: passed through without winsorizing (standardized only in lasso mode).
YEAR_LEVEL = [
    "行业收入增长率", "行业毛利率中位数", "行业销售费用率中位数", "GDP实际增长率", "CPI年度涨幅",
    "全国居民人均可支配收入名义增长率", "社会消费品零售总额年度增长率",
]
FLAGS = ["gp_nonpos", "se_missing_or_zero", "st", "capex_missing"]
CATEGORICAL = ["消费类型", "industry"]

WINSOR_Q = (0.01, 0.99)
MISSING_FLAG_MIN_RATE = 0.01  # add a _missing flag when the train missing rate exceeds this
MISSING_FLAG_MERGE_AGREEMENT = 0.999  # flags agreeing above this on train collapse into one
MIN_INDUSTRY_OBS = 30  # industries with fewer train obs merge into their consumption type
OTHER = "other"
GROUP_SUFFIX = "(合并)"  # consumption-type group label: 消费类型 + GROUP_SUFFIX

# 对应目标行业 labels used in 2013-2020 (older SW version). Later labels are mapped onto these.
PRE2021_INDUSTRIES = [
    "一般零售", "专业连锁II|互联网电商|旅游零售II", "个护用品|化妆品", "中药II", "乘用车", "农产品加工",
    "出版", "医疗服务", "医药商业", "家居用品", "家居用品|饰品", "小家电|厨卫电器|照明设备II|其他家电II",
    "影视院线", "房地产开发", "摩托车及其他", "教育|体育II", "文娱用品", "旅游及景区", "服装家纺",
    "汽车服务", "消费电子", "游戏II|数字媒体", "物流", "电视广播II", "白色家电", "白酒II", "纺织制造",
    "航空机场", "装修建材", "装修装饰II", "调味发酵品II", "通信服务", "酒店餐饮", "非白酒", "食品加工",
    "食品加工|休闲食品", "饮料乳品", "饰品", "黑色家电",
]

# Two-stage design: stage 1 is OLS on PERSISTENCE, stage 2 models its residual.
TEXT_PATH = DATA_DIR / "annual_report_text_scores.csv"
TEXT_COLUMNS = [
    "市场感知叙事强度", "品牌管理叙事强度", "客户关系管理叙事强度", "渠道管理叙事强度",
    "产品开发与新品叙事强度", "营销沟通叙事强度", "数字营销叙事强度",
]
# 1 when the scored report was disclosed after 30 April of the prediction year (not public at the prediction
# point); enters the text module as the feature text_late so the model can learn from it, rows are not blanked.
TEXT_LATE = "披露晚于4月30日"
# Last year's performance plus the MPDR benchmark's own regressors (as in run_lasso.PERSISTENCE).
PERSISTENCE = ["mp_t", "gp_nonpos", "ln1p_se", "se_missing_or_zero", "企业规模", "成立年限", "资产负债率",
               "经营现金流资产比", "资本支出强度", "capex_missing"]
# Stage-2 modules for the ablation. Industry and consumption-type dummies join "macro_industry" in
# modules.module_columns; "persistence" here is its nonlinear part, since stage 1 already took the linear one.
MODULES = {
    "persistence": PERSISTENCE + ["gp_to_assets"],
    "marketing_input": ["销售费用率", "销售费用增长率", "投入增长同步度"],
    "operating_base": ["营业收入增长率", "毛利率", "毛利率变化", "ROA", "经营现金流质量", "现金持有水平",
                       "应收账款占比", "存货占比", "st"],
    # Proposal Table 9: top-5 shares, largest-counterparty shares, HHI and inventory turnover.
    "channel_supply": ["前五大客户占比", "第一大客户占比", "客户集中度", "前五大供应商占比", "第一大供应商占比",
                       "供应商集中度", "存货周转率"],
    "macro_industry": YEAR_LEVEL,
    "text": TEXT_COLUMNS + ["text_missing", "text_late"],
}

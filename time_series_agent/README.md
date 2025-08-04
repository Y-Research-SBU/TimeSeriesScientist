# Time Series Prediction Agent

工业级时序预测代理系统 - 基于 LangGraph 的智能时序预测框架

## 项目概述

Time Series Prediction Agent 是一个基于 LangGraph 的工业级时序预测系统，采用多 Agent 协作的方式，实现从数据预处理到最终预测报告的完整工作流。

## 项目结构

```
time_series_agent/
│
├── agents/                    # Agent 模块
│   ├── preprocess_agent.py    # 数据预处理 Agent
│   ├── analysis_agent.py      # 数据分析 Agent
│   ├── validation_agent.py    # 验证集模型选择 Agent
│   ├── forecast_agent.py      # 测试集预测 Agent
│   ├── report_agent.py        # 报告生成 Agent
│   └── memory.py              # 内存管理模块
│
├── graph/                     # 工作流图
│   └── agent_graph.py         # AgentGraph 类
│
├── utils/                     # 工具模块
│   ├── data_utils.py          # 数据工具
│   ├── visualization_utils.py # 可视化工具
│   └── file_utils.py          # 文件工具
│
├── config/                    # 配置模块
│   └── default_config.py      # 默认配置
│
├── main.py                    # 主入口文件
├── requirements.txt           # 依赖文件
└── README.md                  # 项目说明
```

## 核心特性

### 🤖 多 Agent 协作
- **PreprocessAgent**: 数据清洗、缺失值处理、异常值检测
- **AnalysisAgent**: 统计分析、趋势检测、季节性分析
- **ValidationAgent**: 模型选择、超参数优化、交叉验证
- **ForecastAgent**: 多模型预测、集成学习
- **ReportAgent**: 综合报告生成、可视化

### 🔄 LangGraph 工作流
- 基于 LangGraph 的状态管理
- 节点间直接对象传递
- 支持并行处理和错误恢复
- 完整的执行历史追踪

### 📊 丰富的模型支持
- **统计模型**: ARMA, ExponentialSmoothing, TBATS, Theta
- **机器学习**: LinearRegression, RandomForest, SVR, GradientBoosting
- **深度学习**: LSTM, NeuralNetwork, Transformer
- **集成方法**: XGBoost, LightGBM, Prophet

### 🎯 智能决策
- LLM 驱动的模型选择
- 自动超参数优化
- 动态集成策略
- 置信区间计算

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 设置环境变量

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. 运行实验

```bash
python main.py --data_path dataset/ETT-small/ETTh1.csv \
                --output_dir results \
                --num_slices 10 \
                --horizon 96 \
                --llm_model gpt-4o
```

### 4. 使用配置文件

```bash
python main.py --config_file config/my_config.yaml
```

## 配置说明

### 基础配置

```yaml
# 数据配置
data_path: "dataset/ETT-small/ETTh1.csv"
date_column: "date"
value_column: "OT"

# 实验参数
num_slices: 10
input_length: 512
horizon: 96
k_models: 3

# LLM配置
llm_provider: "openai"
llm_model: "gpt-4o"
llm_temperature: 0.1
```

### 高级配置

```yaml
# 预处理配置
preprocess:
  missing_value_strategy: "interpolate"
  outlier_strategy: "clip"
  normalization: false

# 模型配置
models:
  available_models: ["ARMA", "LSTM", "RandomForest"]
  ensemble_method: "weighted_average"
  hyperparameter_optimization: true

# 可视化配置
visualization:
  figure_size: [12, 8]
  save_format: "png"
  show_plots: false
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` | str | - | 输入数据文件路径 |
| `--output_dir` | str | results | 输出目录 |
| `--num_slices` | int | 10 | 数据切片数量 |
| `--input_length` | int | 512 | 输入序列长度 |
| `--horizon` | int | 96 | 预测步长 |
| `--k_models` | int | 3 | 选择的模型数量 |
| `--llm_provider` | str | openai | LLM提供商 |
| `--llm_model` | str | gpt-4o | LLM模型名称 |
| `--debug` | flag | False | 启用调试模式 |
| `--verbose` | flag | False | 启用详细输出 |

## 输出结构

```
results/
├── experiment_20250101_120000/
│   ├── logs/                    # 日志文件
│   ├── models/                  # 保存的模型
│   ├── visualizations/          # 可视化图片
│   ├── reports/                 # 报告文件
│   │   ├── comprehensive_report_20250101_120000.json
│   │   └── experiment_summary_20250101_120000.json
│   └── data/                    # 预测结果和指标
│       ├── forecasts_20250101_120000.json
│       └── metrics_20250101_120000.json
```

## 扩展开发

### 添加新的 Agent

1. 在 `agents/` 目录下创建新的 Agent 文件
2. 实现 `run()` 方法
3. 在 `graph/agent_graph.py` 中注册新节点

```python
class MyCustomAgent:
    def __init__(self, config, memory):
        self.config = config
        self.memory = memory
    
    def run(self, state):
        # 实现你的逻辑
        return updated_state
```

### 添加新的模型

1. 在相应的 Agent 中添加模型实现
2. 更新配置文件中的模型列表
3. 添加相应的超参数配置

### 自定义工作流

1. 修改 `graph/agent_graph.py` 中的节点连接
2. 添加条件分支和循环逻辑
3. 实现自定义的状态转换

## 性能优化

### 并行处理
```yaml
experiment:
  parallel_processing: true
  max_workers: 4
```

### 缓存管理
```yaml
system:
  max_cache_size: 1000
  cache_ttl: 3600
```

### 内存管理
```yaml
system:
  max_memory_usage: "8GB"
  cleanup_temp_files: true
```

## 故障排除

### 常见问题

1. **LLM API 错误**
   - 检查 API 密钥设置
   - 验证网络连接
   - 确认模型名称正确

2. **内存不足**
   - 减少 `num_slices` 参数
   - 启用 `parallel_processing: false`
   - 增加系统内存

3. **模型训练失败**
   - 检查数据质量
   - 调整超参数范围
   - 查看详细日志

### 调试模式

```bash
python main.py --debug --verbose --data_path your_data.csv
```

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

MIT License

## 联系方式

- 项目维护者: [Your Name]
- 邮箱: [your.email@example.com]
- 项目地址: [GitHub Repository URL]

## 更新日志

### v1.0.0 (2025-01-01)
- 初始版本发布
- 支持基础时序预测功能
- 集成 LangGraph 工作流
- 多 Agent 协作架构 
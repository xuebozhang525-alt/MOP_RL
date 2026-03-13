# 多目标优化强化学习训练指南

## 步骤一：环境配置

为确保环境稳定性和快速部署，推荐使用官方Docker镜像进行VeRL环境配置。如需自定义开发，请使用本地安装方式并配置最新版本的vllm。

```bash
# 克隆VeRL仓库并安装
git clone https://github.com/volcenstein/verl && cd verl
pip3 install -e .[vllm]

# 安装其他依赖包
pip3 install scipy pebble timeout_decorator wandb modelscope datasets \
    langchain coptpy gurobipy==12.0.1 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 步骤二：自定义奖励函数配置

配置针对多目标混合整数线性规划（MO-MILP）的领域特定奖励函数，并集成改进的PPO训练算法。

### 2.1 奖励函数设计

本项目设计了一套两阶段三层次的多目标优化奖励机制：

第一阶段：
| 奖励类型 | 权重 | 描述 |
|---------|------|------|
| 格式奖励 (R_format) | 0.5 | 检查模型输出是否包含正确的代码块格式（```python或&lt;python&gt;标签） |
| 执行奖励 (R_acc) | 1.0 | 验证生成的代码能否成功执行并得到可行解 |
| 准确性奖励 (R_exe) | 2.0 | 可行解与参考解结果一致 |

**总奖励公式：**
```
R_total = 0.5 × R_format + 1.0 × R_acc + 2.0 × R_exe
```


第二阶段：

| 奖励类型 | 权重 | 描述 |
|---------|------|------|
| 格式奖励 (R_format) | 0.5 | 检查模型输出是否包含正确的代码块格式（```python或&lt;python&gt;标签） |
| 执行奖励 (R_acc) | 1.0 | 验证生成的代码能否成功执行并得到可行解 |
| 帕累托奖励 (R_pareto) | 2.0 | 通过帕累托验证器验证解的最优性 |

**总奖励公式：**
```
R_total = 0.5 × R_format + 1.0 × R_acc + 2.0 × R_pareto
```

### 2.2 二阶段奖励函数核心逻辑

奖励函数通过以下步骤计算每个样本的奖励：

**步骤一：格式检查**
- 检查输出是否包含```python代码块或&lt;python&gt;标签
- 满足格式返回0.5，否则返回0.0

**步骤二：代码提取与执行**
- 从模型输出中提取Python代码块
- 将代码写入临时目录的solver.py
- 使用subprocess执行代码（超时30秒）
- 执行成功返回1.0，失败返回0.0

**步骤三：帕累托最优性验证**
- 执行verify.py验证脚本
- 检查返回结果是否包含"True"
- 验证通过返回2.0，否则返回0.0

### 2.3 文件集成

定义VeRL根路径（假设位于verl项目文件夹中）：

```bash
VERL_ROOT="$(pwd)"
echo "VeRL配置路径: $VERL_ROOT"

# 集成自定义奖励函数代码
# 将奖励函数文件复制到VeRL根目录
cp -r reward_func_SOP/*.py "$VERL_ROOT"/
cp -r reward_func_MOP/*.py "$VERL_ROOT"/

```

## 步骤三：数据准备

### 3.1 数据集结构

每个训练样本需包含以下字段：

```json
{
    "problem_id": "问题唯一标识",
    "description": "自然语言问题描述",
    "schema.json": "变量定义模式文件",
    "verify.py": "帕累托验证脚本",
}
```

### 3.2 验证脚本要求

verify.py需满足以下要求：
- 接收solver.py生成的解作为输入
- 通过支配性测试判断解是否属于帕累托最优
- 输出"True"表示帕累托最优，"False"表示被支配

## 步骤四：执行训练

### 训练参数配置示例

```python
# trainer_config.yaml
trainer:
  algorithm: "PPO"  # 或 "REINFORCE++"
  kl_ctrl:
    kl_coefficient: 0.01
    kl_target: 0.02

reward:
  format_weight: 0.5
  accuracy_weight: 1.0
  pareto_weight: 2.0
```

## 附录：常见问题


### Q1: 如何调整奖励权重？
修改reward_multiobjective.py中的权重参数：
```python
W_FORMAT = 0.5  # 格式奖励权重
W_ACC = 1.0     # 执行奖励权重
W_PARADO = 2.0  # 帕累托奖励权重
```

### Q2: 训练过程中如何监控？
- 使用wandb查看训练曲线
- 监控reward_breakdown（format/acc/pareto）
- 检查生成的解的质量分布

# Dragon3 Pipelines - 开发者指南

本文档为 AI 助手提供 Dragon3 N-body 模拟分析工具的完整上下文信息。

## 项目概述

Dragon3 Pipelines 是一个用于分析和可视化 Dragon3 N-body 模拟数据的模块化 Python 包。该项目专注于处理星团模拟中的粒子、双星系统、合并事件等数据，并提供丰富的可视化功能。

**核心功能：**
- HDF5 格式的模拟数据读取与处理
- 粒子轨迹跟踪与演化分析
- 双星系统特性分析（轨道参数、质量比、引力波合并时标等）
- Lagrangian 半径演化分析
- 多进程并行处理支持
- 自动化绘图与数据可视化

## 技术栈与依赖

### Python 版本要求
- Python >= 3.11

### 核心依赖库
- **pandas >= 2.3.0** - 数据处理与 DataFrame 操作
- **numpy >= 2.2.6** - 数值计算
- **h5py >= 3.13.0** - HDF5 文件读写
- **matplotlib >= 3.10.3** - 绘图与可视化
- **astropy >= 7.0.1** - 天文学计算与单位转换
- **scipy >= 1.15.3** - 科学计算
- **seaborn >= 0.13.2** - 统计可视化
- **PyYAML >= 6.0** - YAML 配置文件解析
- **rich >= 13.0.0** - 进度条显示
- **colour-science >= 0.4.6** - 颜色处理

### 开发工具依赖
- **pytest >= 7.0** - 单元测试框架
- **pytest-cov >= 4.0** - 测试覆盖率
- **pytest-xdist >= 3.6** - pytest 并行执行
- **black >= 23.0** - 代码格式化
- **ruff >= 0.1.0** - 代码检查
- **mypy >= 1.0** - 类型检查

## 项目结构说明

```
dragon3_pipelines/
├── config/          # 配置管理模块
│   ├── manager.py       # ConfigManager 类，管理所有配置项
│   └── default_config.yaml  # 默认配置文件（包含路径、物理常数、绘图限制等）
├── io/              # 数据输入输出模块
│   ├── hdf5_processor.py    # HDF5FileProcessor，读取 .h5part 文件
│   ├── lagr_processor.py    # LagrFileProcessor，读取 Lagrangian 半径文件
│   └── text_parsers.py      # 文本文件解析函数
├── analysis/        # 数据分析模块
│   ├── particle_tracker.py  # ParticleTracker，跟踪粒子演化
│   └── tau_gw.py            # 引力波合并时标计算
├── visualization/   # 可视化模块
│   ├── base.py              # BaseVisualizer 基类
│   ├── single_star.py       # SingleStarVisualizer，单星可视化
│   ├── binary_star.py       # BinaryStarVisualizer，双星可视化
│   ├── lagr.py              # LagrVisualizer，Lagrangian 半径可视化
│   └── coll_coal.py         # CollCoalVisualizer，碰撞/合并事件可视化
├── utils/           # 工具函数模块
│   ├── serialization.py     # save()/read() 序列化函数
│   ├── logging.py           # @log_time 装饰器，日志工具
│   ├── shell.py             # get_output() 执行 shell 命令
│   ├── color.py             # BlackbodyColorConverter 颜色转换
│   └── misc.py              # 其他辅助函数
└── scripts/         # Shell 脚本（如视频生成脚本）
```

## 代码规范

### 代码格式化
使用 **Black** 进行代码格式化：
```bash
black --line-length=100 dragon3_pipelines/
```

配置（已在 `pyproject.toml` 中定义）：
```toml
[tool.black]
line-length = 100
target-version = ['py311', 'py312', 'py313']
```

### 代码检查
使用 **Ruff** 进行代码检查：
```bash
ruff check dragon3_pipelines/ --fix
```

配置：
```toml
[tool.ruff]
line-length = 100
target-version = "py311"
```

### 类型注解
使用 **mypy** 进行类型检查：
```bash
mypy dragon3_pipelines/
```

配置：
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

**类型注解要求：**
- 所有公共函数和方法必须添加类型注解
- 函数参数和返回值都需要类型标注
- 使用 `from typing import` 导入类型提示工具

**示例：**
```python
from typing import Dict, List, Optional
import pandas as pd

def process_data(
    df: pd.DataFrame,
    threshold: float,
    columns: Optional[List[str]] = None
) -> Dict[str, pd.DataFrame]:
    """处理数据并返回结果字典"""
    ...
```

## 术语

见 `docs/terminology.md`

### Snapshot vs HDF5 File

**Snapshot（快照）**：
- 指模拟在**某一特定时刻**的数据状态
- 包含单个 `TTOT` 时间点的所有粒子、双星、标量数据
- 从 HDF5 文件中提取：`get_snapshot_at_t(df_dict, ttot)`
- 变量命名示例：`single_df_at_t`、`binary_df_at_t`

**HDF5 File（HDF5 文件）**：
- 物理文件：`.h5part` 格式
- 包含**多个快照**的数据容器（默认每个文件 8 个快照）
- 包含多个 `Step#` 组，每组对应不同的 `TTOT` 值
- 文件命名：`snap.40_<TIME>.h5part`
- 变量命名示例：`hdf5_file_path`、`df_dict`

**使用指南：**
- 提到"特定时刻的数据"时使用 "snapshot"
- 提到"物理文件"或"文件 I/O"时使用 "HDF5 file"

### TTOT 时间单位
`TTOT` 是模拟时间，单位为 N-body 时间单位（通常转换为 Myr）。`TTOT` 是所有快照数据的时间索引。

致密天体 KW 码：`[10, 11, 12, 13, 14]`

## API 兼容性约束

详见 `docs/api.md`

### 扩展原则
- ✅ **可以**添加新的可选参数（必须有默认值）
- ✅ **可以**添加新的方法和类
- ✅ **可以**添加新的返回值字段（在字典或对象中）
- ❌ **不可以**修改现有参数的类型或含义
- ❌ **不可以**删除公共方法或参数
- ❌ **不可以**更改函数返回值的结构

## 开发工作流

### 加载环境
必须执行 `source .venv/bin/activate` 再跑 python 或相关工具, 而不是直接跑 `.venv/bin/python`

### 命令行入口
- `python -m dragon3_pipelines` 或 `dragon3-plot`：运行完整绘图流程
- `python -m dragon3_pipelines --help` / `python -m dragon3_pipelines help`：查看顶层用法
- `python -m dragon3_pipelines help purge`：查看 purge 子命令用法

### 测试命令
Agent Coding 完成代码修改后，必要测试流程是运行项目统一 CI 脚本：
```bash
./scripts/ci.sh
```

该脚本是 GitHub Actions 与本地开发共用的测试入口，会使用 pytest-xdist 并行运行测试，worker 数自动选择但最多 8 个，并默认将 BLAS/OMP 相关线程数限制为 1（可通过已有环境变量覆盖）。

运行特定测试文件：
```bash
pytest tests/test_config.py -v
```

### CI 流程
项目使用 GitHub Actions 进行持续集成（见 `.github/workflows/ci.yml`）：

**触发条件：**
- 推送到 `main`、`develop`、`copilot/**` 分支
- 向 `main`、`develop` 发起 Pull Request

**测试矩阵：**
- Python 3.11, 3.12, 3.13

**CI 步骤：**
参见 .github/workflows/ci.yml

**Pre-commit Hooks：**
项目配置了 pre-commit（见 `.pre-commit-config.yaml`）：
- `black` - 代码格式化
- `ruff` - 代码检查和自动修复
- 各类文件检查（trailing whitespace, YAML 语法, 大文件, merge conflict 等）

安装 pre-commit hooks：
```bash
pip install pre-commit
pre-commit install
```

### 修改文档
在完成后，按需修改 `README.md` 和 `AGENTS.md` ，来更新对本项目的各种记述。注意这两个文档**不是**日志，不应该事无巨细的写入，而只记述对用户或 agent 必要的信息。

### 并行处理注意事项

**设置 OpenBLAS 线程数：**
由于使用了multiprocessing，内部有pandas DataFrame会用到OpenBLAS，如果不手动限制，可能导致进程数×线程数超过系统限制。
```python
import os
os.environ["OPENBLAS_NUM_THREADS"] = "10"
```

**内存管理：**
每个single_df和binary_df的内存占用都接近1GB。使用循环处理大量数据时，应使用 `gc.collect()` 主动回收内存：
```python
import gc

for hdf5_file in hdf5_files:
    process_file(hdf5_file)
    gc.collect()  # 手动触发垃圾回收
```

**典型并行处理模式：**
```python
def process_single_file(args):
    hdf5_path, simu_name, config = args
    # 处理单个文件
    return result

# 使用 forkserver 上下文
ctx = multiprocessing.get_context('forkserver')
args_list = [(path, name, config) for path in hdf5_files]
with ctx.Pool(processes=config.processes_count) as pool:
    results = pool.map(process_single_file, args_list)
```

## 常见模式

### DataFrame 列名约定

**单星数据 DataFrame（singles）：**
- `NAME` - 粒子名称/ID
- `M` - 质量 [M☉]
- `TTOT` - 模拟时间
- `X1`, `X2`, `X3` - 位置坐标 [pc]
- `V1`, `V2`, `V3` - 速度分量 [km/s]
- `Teff*` - 有效温度 [K]
- `L*` - 光度 [L☉]
- `Distance_to_cluster_center[pc]` - 到星团中心的距离
- `KW` - 恒星类型码

**双星数据 DataFrame（binaries）：**
- `NAME(1)`, `NAME(2)` - 主星和伴星名称
- `Bin A[au]` - 半长轴
- `Bin ECC` - 偏心率
- `mass_ratio` - 质量比（次星质量/主星质量）
- `primary_mass[solar]`, `secondary_mass[solar]` - 主星/次星质量
- `total_mass[solar]` - 双星总质量
- `Bin cm V1`, `Bin cm V2`, `Bin cm V3` - 质心速度
- `tau_gw[Myr]` - 引力波合并时标
- `Ebind/kT` - 结合能（归一化）

**标量数据 DataFrame（scalars）：**
- `TTOT` - 时间
- 各种全局统计量

**合并事件 DataFrame（mergers）：**
- `TTOT` - 合并时间
- `NAME(OUT)` - 合并产物名称
- `NAME(1)`, `NAME(2)` - 参与合并的两个天体

### 缓存机制

**Analysis 结果缓存：**
analysis/data-reduction 产生的中间结果缓存统一放在 `paths.analysis_cache_dir` 下，目录结构固定为：

```text
<analysis_cache_dir>/<simu_name>/<feature>/...
```

其中 feature 目录名使用稳定的小写 snake_case，不使用展示标签。当前约定的 feature 名包括：

- `particle_df`
- `primordial_binary`
- `b_type_binary`
- `binary_stellar_type`
- `current_lagrangian`
- `compact_binary_count`
- `galactic_orbit`
- `intermediate_mass_black_hole`
- `initial_total_mass`

代码中不要手写拼接这些目录；应使用 `dragon3_pipelines.analysis.cache_paths.analysis_cache_dir(config, simu_name, feature)` 以及该模块中定义的 feature 常量。`ConfigManager` 会派生：

- `config.analysis_cache_dir_of[simu_name] == <analysis_cache_dir>/<simu_name>`
- `config.particle_df_cache_dir_of[simu_name] == <analysis_cache_dir>/<simu_name>/particle_df`

`particle_df_cache_dir_of` 仅作为粒子追踪缓存目录兼容属性保留；新代码不要把其他 analysis 缓存手动挂在它下面。`paths.cache_dir_suffix` 已弃用，仅用于兼容缺少 `analysis_cache_dir` 的旧用户配置；新配置必须使用 `paths.analysis_cache_dir`。

**HDF5 I/O 加速缓存：**
`.h5part.*.df.feather` 这类从 HDF5 table 派生的 Feather 缓存属于文件读取加速缓存，应继续放在对应 `.h5part` 文件旁边，不纳入 `analysis_cache_dir`。

**典型 Feather 缓存模式：**
使用 Apache Arrow Feather 格式缓存 DataFrame，速度快且保留类型信息。保存 analysis 结果时，先通过缓存路径 helper 得到 feature 目录：

```python
import pandas as pd

from dragon3_pipelines.analysis.cache_paths import PARTICLE_DF_FEATURE, analysis_cache_dir

# 保存缓存
cache_path = analysis_cache_dir(config, simu_name, PARTICLE_DF_FEATURE) / "particle_data.feather"
cache_path.parent.mkdir(parents=True, exist_ok=True)
df.to_feather(cache_path)

# 读取缓存
if cache_path.exists():
    df = pd.read_feather(cache_path)
```

**典型缓存模式：**
```python
from pathlib import Path

def get_data_with_cache(cache_path: Path) -> pd.DataFrame:
    """带缓存的数据读取"""
    if cache_path.exists():
        logger.info(f"Loading from cache: {cache_path}")
        return pd.read_feather(cache_path)

    # 计算数据
    df = compute_expensive_data()

    # 保存缓存
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_feather(cache_path)
    return df
```

### 日志使用

**获取 logger：**
```python
import logging
logger = logging.getLogger(__name__)
```

**使用 @log_time 装饰器：**
```python
from dragon3_pipelines.utils.logging import log_time
import logging

logger = logging.getLogger(__name__)

@log_time(logger)
def expensive_function():
    """耗时函数会自动记录开始/结束时间和执行时长"""
    # 复杂计算
    pass
```

**典型日志模式：**
```python
import logging
from dragon3_pipelines.utils.logging import log_time

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        logger.info(f"Initialized DataProcessor with config: {config}")

    @log_time(logger)
    def process_all_files(self, file_list):
        """处理所有文件，自动记录时间"""
        logger.info(f"Processing {len(file_list)} files")
        for i, file_path in enumerate(file_list):
            logger.debug(f"Processing file {i+1}/{len(file_list)}: {file_path}")
            self._process_single_file(file_path)
        logger.info("All files processed")
```

### Analysis/Data Reduction 扫描约定

HDF5 文件中的可绘图信息大体分为两类，维护时应先判断属于哪一类：

- **微观信息（microscopic information）**：单个 snapshot 已包含大量对象级信息，例如某一时刻每颗恒星的状态。此类信息通常可以由单个 HDF5 file 生成一张图，应保留在 `SimulationPlotter.plot_hdf5_file` 调度的逐文件绘图流程中。
- **宏观信息（macroscopic information）**：单个 snapshot 只贡献一个或少量统计点，例如需要画随时间演化的统计量。此类图需要遍历大量 HDF5 files 收集时间序列，应作为 analysis/data-reduction 扫描任务处理。

未来凡是“遍历 HDF5 文件抽取小型信息”的 analysis/data-reduction 功能，优先实现为 `HDF5ScanTask` 并通过 `HDF5ScanRunner` 执行。对应的外层 analysis class 应继承 `ScanBackedAnalysisBase`，实现很薄的 `build_scan_job()`；具体数据提取、merge、cache path、meta 语义仍放在独立 task 中。不要复制新的 HDF5 遍历循环，也不要把这类缓存写入逻辑塞进绘图主循环；`SimulationPlotter.plot_hdf5_file` 应保持绘图调度职责。

所有 HDF5 文件选择、table cache、scan 并行和时间采样配置集中在全局 `hdf5` 配置节。feature 配置（如 `current_lagrangian`、`galactic_orbit`、`binary_stellar_type_extraction`）只保留 `enabled`、缓存文件名和绘图参数等专属字段。`hdf5.file_selection.sample_every_nb_time` 同时控制 scan 和主 HDF5 绘图；`None` 或 `<= 0` 表示不采样，正数表示保留从 `0.0` 开始落在该 NB 时间间隔倍数上的 snapshot。

当同一 simulation 下需要同时运行多个 HDF5 scan task 时，优先通过 `HDF5ScanSession` 堆积 job 后统一 `run()`，让相同 scan options 的任务共享 HDF5 文件读取。默认尾部增量策略会信任已处理尾部之前的旧文件；旧 HDF5 文件被手动改写时应使用 `force=True` 或删除对应 analysis cache 后重建。

### ❌ 其他禁止事项

- **禁止在代码中硬编码绝对路径**（应使用配置或相对路径）
- **禁止在测试中依赖外部数据文件**（使用 mock 或生成测试数据）
- **禁止提交包含敏感信息的文件**（如 API 密钥、私有路径等）
- **禁止跳过类型注解**（所有公共函数必须有类型标注）
- **禁止使用未经测试的新依赖库**（添加新依赖前应评估必要性和兼容性）

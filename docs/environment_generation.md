# EnvFactory 环境生成流程详解

## 📋 概述

EnvFactory 通过**从发现到验证**的自动化流水线，将 API 文档转换为可执行的 MCP 服务器环境，用于训练和测试 Agent。本文档详细介绍完整的生成流程、核心设计原则和实用技巧。

## 🎯 核心概念

### 什么是"环境"？

在 EnvFactory 中，一个完整的环境包含：

- **MCP Server 代码**：实现特定领域工具的 Python 服务（如 GoogleMaps、Twitter、Calendar）
- **Schema 定义**：工具的输入输出规范（JSON Schema 格式）
- **Test Scenarios**：多样化的测试场景数据，覆盖不同复杂度
- **状态管理机制**：通过 `load_scenario` 和 `save_scenario` 实现环境快照和恢复

### 为什么需要生成环境？

1. **可扩展性**：手工编写数百个 API 工具成本极高，自动化生成大幅降低成本
2. **一致性**：统一的 MCP 协议保证 Agent 与环境的标准化交互
3. **可测试性**：自动生成的场景覆盖边界条件和错误情况
4. **RL 训练支持**：状态管理机制支持强化学习的 rollout 和 reset 操作

## 🏗️ 完整流水线

### 五阶段：从发现到验证

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      EnvFactory 完整生成流水线                            │
└──────────────────────────────────────────────────────────────────────────┘

  Phase 0           Phase 1          Phase 2           Phase 3            Phase 4
┌──────────┐      ┌─────────┐      ┌─────────┐      ┌──────────┐      ┌──────────────┐
│ Schema   │ ───> │ Schema  │ ───> │  Tool   │ ───> │ Scenario │ ───> │  Validation  │
│ Discovery│      │ Design  │      │   Gen   │      │   Gen    │      │  & Revision  │
└──────────┘      └─────────┘      └─────────┘      └──────────┘      └──────────────┘
    │                 │                 │                 │                    │
    ▼                 ▼                 ▼                 ▼                    ▼
sketch.py        metadata.json    tool_code.py      checkpoint            validated_env
                                                    .scenarios[]          tool file
```

**核心设计理念**：
- 🔍 **LLM 作为编译器**：每个阶段都是 `输入规范 → LLM 处理 → 结构化输出`
- 🔄 **自我验证与修复**：验证失败自动触发代码修订（最多 3 轮）
- 💾 **状态快照**：所有环境都支持 `load_scenario` / `save_scenario`（可重现、可并行）
- 🎯 **非 AI 聚焦**：只生成工具型 API（数据、生产力、基础设施），排除 LLM/图像生成等 AI 服务

### 核心组件

| 阶段 | 组件 | 文件路径 | 职责 |
|------|------|---------|------|
| **Phase 0** | `mcp-sketch-discovery` | `.agents/skills/mcp-sketch-discovery/SKILL.md` | 通过网络搜索发现新 API，生成骨架代码 |
| **Phase 1** | `SchemaGen` | `src/gen/mcp_schema_gen.py` | 从骨架代码提取标准化 schema |
| **Phase 2** | `MCPToolGen` | `src/gen/env_gen/mcp_tool_gen.py` | 生成完整 MCP 服务器代码 |
| **Phase 3** | `ScenarioGen` | `src/gen/env_gen/mcp_tool_gen.py` | 生成测试场景数据 |
| **Phase 4** | `ValidateReviseGen` | `src/gen/env_gen/validate_revise.py` | 验证并修订代码（最多 3 轮）|
| - | `EnvGen` | `src/gen/env_gen/env_gen.py` | 流程编排和状态管理 |

---

## 🔧 详细流程

### Phase 0: Schema Discovery（可选，扩展新 API 时使用）

**目标**：系统化地发现高质量 API 机会并生成轻量级骨架代码。来源不是静态的文档，而是真实的外部网络数据

**触发方式**：
```bash
# 在 Claude Code 中调用 skill
/mcp-sketch-discovery
```

**工作流程**（6 步）：

```
1. Inventory & Proposal
   扫描 envs/schema_sketch/ 现有覆盖
   ↓
   提出 3 个搜索方向供用户选择

2. Discovery Search  
   执行网络搜索：
   - "best {category} APIs 2025"
   - "{category} API documentation"
   - "free {category} API for developers"
   ↓
   收集 10-15 个候选 API

3. Deduplication & Selection
   过滤规则：
   ✗ 排除 AI 相关（LLM/Chat/Embedding/图像生成）
   ✗ 与现有 server 重复
   ✓ 优先：独特功能 > 开发者流行度 > 文档质量
   ↓
   选出 Top 5

4. Deep Research
   收集每个候选的：
   - 官方文档 URL
   - 认证方法
   - 核心端点/功能
   - 典型用例

5. Sketch Generation
   生成 5 个骨架文件到 envs/schema_sketch/{server_name}/{server_name}_server.py

6. Summary Report
   输出汇总表格 + 下一轮建议
```

**生成的骨架代码示例**：
```python
# envs/schema_sketch/airtable/airtable_server.py

# Data Source: https://airtable.com/developers/web/api
# Server: Airtable
# Category: productivity

def list_bases() -> dict:
    """
    Retrieve all bases accessible to the authenticated user.
    
    Returns:
        dict: {
            "bases": [
                {"id": str, "name": str, "permission_level": str}
            ]
        }
    """
    pass

def list_records(base_id: str, table_name: str, max_records: int = None) -> dict:
    """
    Retrieve records from a specific table.
    
    Args:
        base_id (str): The ID of the base
        table_name (str): The name of the table
        max_records (int): [Optional] Maximum number of records to return
        
    Returns:
        dict: {
            "records": [
                {"id": str, "fields": dict, "created_time": str}
            ]
        }
    """
    pass

def create_record(base_id: str, table_name: str, fields: dict) -> dict:
    """
    Create a new record in a table.
    
    Args:
        base_id (str): The ID of the base
        table_name (str): The name of the table
        fields (dict): Field name to value mapping
        
    Returns:
        dict: {
            "id": str,
            "fields": dict,
            "created_time": str
        }
    """
    pass
```

**关键特征**：
- ✅ 只包含函数签名 + 完整 docstring
- ✅ 函数体使用 `pass`（无实现）
- ✅ 类型标注完整
- ✅ 标注可选参数 `[Optional]`
- ✅ 附带 `{server_name}_research.md` 记录详细调研

**输出位置**：
```
envs/schema_sketch/
├── airtable/
│   ├── airtable_server.py
│   └── airtable_research.md
├── stripe/
│   ├── stripe_server.py
│   └── stripe_research.md
└── ...
```

---

### Phase 1: Schema 设计

**目标**：将骨架代码转换为标准化的 metadata.json

**输入源**：
- **Schema Sketch**：Python/TypeScript 骨架代码（来自 Phase 0 或手工编写）
- **Data Source**：API 原始文档的 JSON 表示（可选）

**执行**：
```bash
# 从 schema sketch 生成
python -m src.gen.mcp_schema_gen envs/schema_sketch/airtable/airtable_server.py

# 指定输出路径
python -m src.gen.mcp_schema_gen envs/schema_sketch/calendar_server.py \
  --output envs/metadata/Calendar_metadata.json

# 使用特定模型
python -m src.gen.mcp_schema_gen envs/schema_sketch/calendar_server.py \
  --model kimi-k2
```

**Prompt 核心逻辑**（`SchemaDesign_System_Prompt`）：

> 🎯 **核心目标**：正确推断输入输出形状 + 语义增强 + 统一输出结构 + 内部一致性

**关键约束**：

1️⃣ **命名规范**
```python
class_name: "UpperCamelCase"  # ✓ GoogleCalendar  ✗ google_calendar
tool_name:  "snake_case"       # ✓ create_event   ✗ createEvent
```

2️⃣ **输出 Schema 强制包裹**（关键设计！）
```python
# ❌ 错误：返回原始类型
{"type": "string"}

# ✅ 正确：包裹在对象中
{
  "type": "object",
  "properties": {
    "result_count": {"type": "integer"}
  }
}
```
**原因**：Agent 训练需要一致的输出结构，避免类型歧义

3️⃣ **语义一致性**（用于向量检索）
```python
# 同一概念在多个工具中必须使用相同描述
get_user(user_id: str):  # "The unique identifier of the user account"
delete_user(user_id: str):  # "The unique identifier of the user account"  # 一字不差！
```

4️⃣ **输入 Schema 优先级**
```
原始 API 文档 > 明确提供的 parameters > 最小化推断
```
**反模式**：不要过度推理参数（如根据 "search" 自作主张加 "city", "radius" 参数）

**输出格式** (`metadata.json`)：
```json
{
  "class_name": "Airtable",
  "description": "Airtable API for managing bases, tables and records",
  "tools": [
    {
      "name": "list_bases",
      "description": "Retrieve all accessible bases for the authenticated user",
      "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "bases": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "id": {"type": "string", "description": "Unique base identifier"},
                "name": {"type": "string", "description": "Human-readable base name"}
              }
            }
          }
        }
      }
    },
    {
      "name": "create_record",
      "description": "Create a new record in a specified table",
      "input_schema": {
        "type": "object",
        "properties": {
          "base_id": {"type": "string", "description": "The unique identifier of the base"},
          "table_name": {"type": "string", "description": "The name of the table"},
          "fields": {"type": "object", "description": "Field name to value mapping"}
        },
        "required": ["base_id", "table_name", "fields"]
      },
      "output_schema": {
        "type": "object",
        "properties": {
          "record_id": {"type": "string", "description": "Unique identifier for the created record"},
          "created_time": {"type": "string", "description": "ISO 8601 timestamp of creation"}
        }
      }
    }
  ]
}
```

---

### Phase 2: Tool 代码生成

#### 这一步在解决什么问题

Phase 1 产出的 metadata 只描述了**接口长什么样**——工具叫什么、收什么参数、返回什么字段。它没有任何行为：调用 `create_record` 不会真的创建记录。

Phase 2 要补上行为。但补的方式不是去接真实的 Airtable API，而是**写出一个假的、可完全掌控的 Airtable**：所有数据都放在一个 Python 对象的内存里，可以随时存档、随时还原。这样做的原因是下游用途决定的——RL 训练要在同一个初始状态上反复采样，而真实 API 做不到这件事（有配额、有网络抖动、状态还会被上一次调用污染）。

所以这一步的产物不是"API 客户端"，而是一个**自洽的状态机**：给它一份初始状态，它就能按工具语义演化，并在任意时刻把状态原封不动导出来。

#### 代码路径

入口是 [MCPToolGen.gen()](../src/gen/env_gen/mcp_tool_gen.py#L59)，流程很短：

```
metadata['tools']
  └─ format_tools_for_prompt()      # 把每个工具拍平成 Name/Description/Input Schema/Output Schema 文本
       └─ Runner.run(tool_generator) # 一次 LLM 调用，出整个文件
            └─ parse_structured_output → output_json['tool_code']
                 └─ save_tools()      # 落盘到 envs/tools/{class_name}.py
```

值得注意的是：**整个文件是一次生成的**，没有按工具分批。一个有 20 个工具的服务器，模型要一口气吐出 500+ 行代码。这也是为什么 `gen()` 里写了 `max_attempts = 2`——输出被截断、`<tool_code>` 标签丢失都会触发重试（[:98-161](../src/gen/env_gen/mcp_tool_gen.py#L98-L161)）。另外这一步默认用 `tool_gen_model`（配置里是 `claude`），并且把 LiteLLM 超时拉到了 3600 秒，就是为长输出留余量。

**执行**（Phase 2-4 是连在一起跑的，没有单独跑 Phase 2 的命令）：
```bash
python -m src.gen.env_gen envs/metadata/Airtable_metadata.json

# 自定义配置
python -m src.gen.env_gen envs/metadata/Airtable_metadata.json \
  --n-scenarios 10 \
  --max-revisions 5 \
  --model deepseek
```

#### 为什么强制 4-Section 结构

`MCPToolGenerator_System_Prompt`（[prompts.py:195](../src/gen/env_gen/prompts.py#L195)）要求生成的文件严格分成四段，顺序固定：

| Section | 内容 | 承担的职责 |
|---|---|---|
| 1 Schema | Pydantic 实体模型 + 一个 Scenario 模型 | 定义状态的**形状**，并承担全部格式/范围校验 |
| 2 Class | 一个普通 Python 类，工具是它的公开方法 | 定义状态的**演化规则**，不含 try-except |
| 3 MCP Tools | `FastMCP` 实例 + `@mcp.tool()` wrapper | 对外暴露协议接口，统一兜异常 |
| 4 Entry Point | `mcp.run()` | 让文件能作为独立进程被拉起 |

这不只是"代码整洁"的要求，而是**下游程序真的按这些注释切文件**。[extract_pydantic_models()](../src/gen/env_gen/mcp_tool_gen.py#L206) 和 [extract_mcp_tools()](../src/gen/env_gen/mcp_tool_gen.py#L246) 用字符串匹配 `# Section 1:` / `# Section 3:` 来截取片段，喂给 Phase 3 和 Phase 4——Phase 3 只需要看 Pydantic 模型就能造场景数据，Phase 4 只需要看 wrapper 就能判断调用是否合法，都不必吞下整个文件。所以注释行写错，下游拿到的就是空字符串（有兜底逻辑，但只认 `Scenario_Schema` 和 `FastMCP` 这两个 fallback 锚点）。

分层的另一个作用是**把校验和业务逻辑彻底分开**：Section 1 管"数据长得对不对"，Section 2 管"这件事能不能做"，Section 3 管"出错了怎么回话"。三者互不重叠，模型生成时不容易在三个地方写三份重复检查。

下面是骨架示例（省略了大部分工具，完整的真实产物见 [envs/tools/Calendar.py](../envs/tools/Calendar.py)）：

```python
# ============================================================
# Section 1: Schema (Pydantic 模型)
# ============================================================
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class Record(BaseModel):
    """Airtable 记录实体"""
    id: str = Field(..., pattern=r"^rec[a-zA-Z0-9]+$")
    fields: Dict[str, Any] = Field(default={})
    created_time: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

class Base(BaseModel):
    """Airtable Base 实体"""
    id: str = Field(..., pattern=r"^app[a-zA-Z0-9]+$")
    name: str = Field(..., min_length=1)
    tables: Dict[str, Dict[str, Record]] = Field(default={})  # {table_name: {record_id: Record}}

class AirtableScenario(BaseModel):
    """主场景模型，定义完整状态"""
    bases: Dict[str, Base] = Field(default={})
    current_time: str = Field(
        default="2024-01-01T00:00:00Z",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    )

Scenario_Schema = [Record, Base, AirtableScenario]  # 定义内部状态结构

# ============================================================
# Section 2: Class (核心逻辑)
# ============================================================
class AirtableAPI:
    def __init__(self):
        """初始化 API，所有状态从 scenario 加载"""
        self.bases: Dict[str, Base] = {}
        self.current_time: str = ""
    
    def load_scenario(self, scenario: dict) -> None:
        """加载场景数据（Pydantic 自动验证）"""
        model = AirtableScenario(**scenario)  # 格式/范围验证在此完成
        self.bases = model.bases
        self.current_time = model.current_time
    
    def save_scenario(self) -> dict:
        """保存当前状态（必须返回所有字段）"""
        return {
            "bases": {
                base_id: base.model_dump() for base_id, base in self.bases.items()
            },
            "current_time": self.current_time
        }
    
    def list_bases(self) -> dict:
        """从状态读取，不创建假数据"""
        return {
            "bases": [
                {"id": base_id, "name": base.name}
                for base_id, base in self.bases.items()
            ]
        }
    
    def create_record(self, base_id: str, table_name: str, fields: dict) -> dict:
        """直接操作状态变量"""
        if base_id not in self.bases:
            raise ValueError(f"Base {base_id} not found")
        
        base = self.bases[base_id]
        if table_name not in base.tables:
            base.tables[table_name] = {}
        
        record_id = f"rec_{len(base.tables[table_name]) + 1}"
        record = Record(
            id=record_id,
            fields=fields,
            created_time=self.current_time  # 从状态读取时间，不用 datetime.now()
        )
        base.tables[table_name][record_id] = record
        return {"record_id": record_id, "created_time": record.created_time}

# ============================================================
# Section 3: MCP Tools (FastMCP 注册)
# ============================================================
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="AirtableAPI")
api = AirtableAPI()

@mcp.tool()
def load_scenario(scenario: dict) -> str:
    """
    Load scenario data into the Airtable API.

    Args:
        scenario (dict): Scenario dictionary matching AirtableScenario schema.

    Returns:
        success_message (str): Success message.
    """
    try:
        if not isinstance(scenario, dict):
            raise ValueError("Scenario must be a dictionary")
        api.load_scenario(scenario)
        return "Successfully loaded scenario"
    except Exception as e:
        raise e

@mcp.tool()
def create_record(base_id: str, table_name: str, fields: dict) -> dict:
    """
    Create a new record in a specified table.

    Args:
        base_id (str): The unique identifier of the base.
        table_name (str): The name of the table.
        fields (dict): Field name to value mapping.

    Returns:
        record_id (str): Unique identifier for the created record.
        created_time (str): ISO 8601 timestamp of creation.
    """
    try:
        # 基本参数检查（存在性、类型）
        if not base_id or not isinstance(base_id, str):
            raise ValueError("base_id must be a non-empty string")
        if not table_name or not isinstance(table_name, str):
            raise ValueError("table_name must be a non-empty string")
        
        # 业务逻辑检查（格式/范围验证由 Pydantic 处理）
        return api.create_record(base_id, table_name, fields)
    except Exception as e:
        raise e

# ============================================================
# Section 4: Entry Point
# ============================================================
if __name__ == "__main__":
    mcp.run()
```

#### Prompt 里的五条硬约束

这些约束读起来像代码风格建议，但每一条背后都对应一个"不这样写下游就会坏"的具体后果。下面按"为什么需要 → 怎么写"的顺序说。

---

**① 校验只写一次，写在 Pydantic 里**

同一个约束如果在 Pydantic 模型和 wrapper 里各写一遍，两处早晚会不一致——改了 Field 的 pattern 忘了改 wrapper 的正则，就会出现"模型认为合法、wrapper 拒绝"的诡异行为。而且 Phase 4 修订时模型每次只改一处，重复校验会让 bug 反复复活。

Prompt 因此明确划了责任线（[prompts.py:293-296](../src/gen/env_gen/prompts.py#L293-L296)）：**格式和范围一律由 Field 约束负责**，wrapper 只做 Pydantic 管不到的事。

```python
# ✅ 约束写在模型上，load_scenario 时自动生效
class Record(BaseModel):
    id: str = Field(..., pattern=r"^rec[a-zA-Z0-9]+$")
    price: float = Field(..., ge=0)

# ❌ wrapper 里重写一遍格式/范围检查
@mcp.tool()
def create_record(record_id: str, price: float):
    if not re.match(r"^rec[a-zA-Z0-9]+$", record_id): ...   # 和上面重复
    if price < 0: ...                                        # 和上面重复
```

| 层 | 负责什么 | 例子 |
|---|---|---|
| Pydantic 模型（主） | 格式 `pattern`、范围 `ge/le/gt`、类型转换 | ID 是否符合 `^rec...`、价格是否非负 |
| MCP wrapper（次） | 参数存在性、`isinstance`、**业务逻辑** | 参数是不是空串、这个 base_id 在不在库里、余额够不够 |

区别的关键在于：Pydantic 只看**单个值本身**对不对，看不到"这个 ID 在当前状态里存不存在"——那需要读 `self.bases`，只能放在 Section 2/3。

---

**② 状态即真理：不许凭空造数据**

这是最容易被违反、后果也最严重的一条。模型看到 `get_weather(city)` 这样的签名，本能反应是返回一份看起来合理的假天气。但那样生成出来的环境是**没有状态的**：同样的调用永远返回同一个值，Agent 做什么都不会改变世界，RL 也就没有任何可学的信号。

Prompt 用 "State as Truth" 和 "Anti-Lazy Logic" 两条规则堵这个洞（[prompts.py:265-269](../src/gen/env_gen/prompts.py#L265-L269)）：类实例是唯一的数据源，`self.xxx` 就是数据库表，禁止模拟 HTTP、禁止硬编码返回值、**缺数据时不要造数据来把响应填满**。

```python
# ❌ 硬编码：调 100 次得到 100 个相同结果，状态永远不变
def get_weather(self, city: str):
    return {"temp": 25, "condition": "sunny"}

# ✅ 从状态读：查不到就报错，让 Agent 知道这个城市不存在
def get_weather(self, city: str):
    if city not in self.weather_data:
        raise ValueError(f"City {city} not found")
    return self.weather_data[city]
```

处理缺失数据的原则也是明确的：**读操作返回空结果，写操作报错**。查询空列表是合法的世界状态，但"更新一条不存在的记录"必须失败——否则 Agent 学不到"先查再改"的行为。

---

**③ 时间必须来自 Scenario，不能来自系统时钟**

`datetime.now()` 会让环境不可重现：同一份场景今天跑和明天跑，"今天有哪些日程"的答案不一样，Phase 4 的验证结论也就不稳定；用它生成的训练数据更是一次性的。

所以 prompt 要求把当前时间当成**状态的一部分**——一个带 pattern 约束的字符串字段（[prompts.py:223](../src/gen/env_gen/prompts.py#L223)），代码里一律读 `self.current_time`：

```python
class Scenario(BaseModel):
    current_time: str = Field(
        default="2024-01-01T00:00:00Z",
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    )

def get_current_time(self):
    return self.current_time      # 而非 datetime.now()
```

同样的道理适用于随机数。prompt 的态度是"尽量别用"，实在需要就把 `random_seed` 放进 Scenario 并显式 `random.seed()`（[prompts.py:281](../src/gen/env_gen/prompts.py#L281)）——本质上都是**把不确定性收敛进 scenario**，让"初始状态 + 调用序列"能唯一决定结果。

---

**④ 查表数据也是状态，不是常量**

税率表、运费区间表这类"参考数据"最容易被写成模块级常量或者直接 `return 0.088`。但那样它就**没法被场景覆盖**——Phase 3 想造一个"某地区税率异常"的边界场景就无从下手，因为那个数字被焊死在代码里了。

做法是把查表数据也放进 Scenario 模型，用 `Field(default=...)` 给 10-20 条默认值（[prompts.py:271-278](../src/gen/env_gen/prompts.py#L271-L278)）。这样默认能跑，场景又能覆写：

```python
class InventoryScenario(BaseModel):
    items: Dict[int, Item] = Field(default={})
    taxRatesMap: Dict[str, float] = Field(default={
        "NY": 0.088, "CA": 0.072, "TX": 0.062,   # 10-20 条
        "Default": 0.050
    })

def calculate_total(self, price: float, region: str) -> dict:
    rate = self.taxRatesMap.get(region, self.taxRatesMap["Default"])
    return {"total": price * (1 + rate)}
```

注意它必须同时出现在 `save_scenario()` 的返回值里，否则存档-还原一圈之后表就丢了。

---

**⑤ 异常只在 wrapper 层兜**

如果 Section 2 的方法自己 try-except 再返回一个 `{"error": ...}`，异常就被"吃掉"变成了正常返回值。Phase 4 的验证器靠**调用是否抛异常**来判断工具行为对不对，被吃掉的错误会被当成成功，问题就漏过去了。

所以 prompt 要求（[prompts.py:286-291](../src/gen/env_gen/prompts.py#L286-L291)）：类方法**不写 try-except**，该报错就 `raise`，让异常自然冒泡；wrapper 统一捕获，交给 MCP 框架转成协议错误。

```python
# Section 2：直接抛，不捕获
def create_record(self, base_id: str, fields: dict):
    if base_id not in self.bases:
        raise ValueError(f"Base {base_id} not found")

# Section 3：统一兜底
@mcp.tool()
def create_record(base_id: str, fields: dict) -> dict:
    try:
        return api.create_record(base_id, fields)
    except Exception as e:
        raise e     # 由 MCP 框架转成协议层错误
```

顺带一个约定：如果类方法返回 `None`（空输出），wrapper 要返回一句成功消息字符串（`-> str`）而不是 `None`，因为 MCP 协议侧需要一个可读的响应体。`load_scenario` 就是这个模式的典型——真实产物见 [Calendar.py:287-304](../envs/tools/Calendar.py#L287-L304)。

---

**关于 wrapper 的返回类型标注**：prompt 要求只用 `-> dict` / `-> str` 这类朴素类型，不要标 Pydantic 模型（[prompts.py:243](../src/gen/env_gen/prompts.py#L243)）。因为 FastMCP 会把标注反射成对外的 JSON Schema，标上 `-> Record` 会把内部实体结构泄漏到工具接口上，而内部模型是可以在 Phase 4 被改的——接口不该跟着变。

**关于 docstring**：Section 2 的方法一行说明就够，Section 3 的 wrapper 必须写完整的 Google 风格三段式（Description / Args / Returns），且 Returns 要逐字段对上 metadata 的 output_schema（[prompts.py:298-303](../src/gen/env_gen/prompts.py#L298-L303)）。原因很直接——**这段 docstring 就是 Agent 在推理时唯一能看到的工具说明**，MCP 把它作为 tool description 暴露出去。写漏一个返回字段，模型就不知道那个字段存在。

---

### Phase 3: Scenario 生成

#### 这一步在解决什么问题

Phase 2 交出来的是一个空转的状态机：类定义齐全，但 `self.bases` 是空字典，任何查询都返回空、任何写操作都因为"找不到父实体"而报错。它需要**初始状态**才能活起来。

Phase 3 就是造这批初始状态。但它的定位容易被误解成"造点测试数据"——实际上这批 scenario 有两个完全不同的下游用途，而且第二个才是重点：

1. **给 Phase 4 当测试用例**：每个 scenario 被 `load_scenario` 灌进去，然后逐个工具跑一遍，看代码有没有 bug。这是本 pipeline 内部闭环用的。
2. **给 query_gen 当种子**：环境交付之后，任务生成阶段会读 `Scenario_Schema` 的 Pydantic 源码，让另一个 agent 按同样的结构编出任务专属的初始状态（[query_gen_non_conv.py:210-212](../src/gen/query_gen/query_gen_non_conv.py#L210-L212) → [read_scenario_schema()](../src/utils/utils.py#L336)）。

第 2 条意味着 Phase 3 真正在验证的是**"这套 Pydantic 模型能不能被一个只看 Section 1 的 LLM 正确填满"**。如果模型定义得太绕、字段之间的引用关系只存在于作者脑子里，Phase 3 就会造出填不对的数据——而这个失败信号是有价值的，它提前暴露了 query_gen 阶段一定会踩的坑。所以 Phase 4 里"到底是代码错了还是场景错了"这个判断才需要单独一条分支（见 [Phase 4](#phase-4-验证与修订循环)）。

#### 代码路径

入口 [ScenarioGen.generate_scenarios()](../src/gen/env_gen/mcp_tool_gen.py#L320)，由 [env_gen.py:156-176](../src/gen/env_gen/env_gen.py#L156-L176) 在 Step 3 调用：

```
tool_code（Phase 2 的完整文件）
  └─ extract_pydantic_models()        # 只切出 Section 1，到 Scenario_Schema = [...] 那行为止
       └─ Runner.run(scenario_generator)   # 一次 LLM 调用，出全部 n 个 scenario
            └─ 结构校验 + 字符串兜底解析
                 └─ result.scenarios → checkpoint
```

三个值得注意的实现细节：

**只喂 Section 1，不喂整个文件。** [extract_pydantic_models()](../src/gen/env_gen/mcp_tool_gen.py#L206) 按 `# Section 1:` 起、`Scenario_Schema = ` 止做字符串截取（找不到标记时退化成"从头读到 `Scenario_Schema` 行"）。这既是省 context，也是刻意的信息隔离——**生成场景只该看状态的形状，不该看工具怎么实现**。让它看见 `create_record` 的代码，它就会开始揣测"什么输入能走到哪个分支"，造出的数据会贴着实现而不是贴着 schema。

**一次调用出全部 scenario，不按复杂度分批。** 好处是模型能横向看到"我已经写了哪几个"，从而让 4 个场景的实体 ID、时间跨度彼此不撞车；代价是输出很长（实测复杂场景单个 `scenario_data` 中位数 3.5 KB），JSON 截断风险高。这也是 [:392-426](../src/gen/env_gen/mcp_tool_gen.py#L392-L426) 那段兜底解析存在的原因：如果 `scenarios` 字段回来是个字符串而不是列表，代码会依次尝试剥 markdown ```` ```json ```` 围栏、直接 `json.loads`、最后退到"找最外层 `[` 和 `]` 截取"。整个 `generate_scenarios` 外面还包了 `max_attempts = 2`。

**默认只生成 4 个场景**（`n_scenarios: int = 4`，[__init__.py:31](../src/gen/env_gen/__init__.py#L31)），用 `--n-scenarios` 覆盖。落盘上它不是独立的 `scenarios.json`——虽然 `ScenarioGen` 提供了 [save_scenarios()](../src/gen/env_gen/mcp_tool_gen.py#L498)，但 pipeline 里没人调它，场景实际存在 `envs/intermediate/{ClassName}_checkpoint.json` 的 `scenarios` 字段里，跟 tool_code、验证报告一起构成断点续跑的现场。

#### 硬性结构：四个字段

模型必须输出 `<scenarios>` 包裹的 JSON 数组，每个元素四个字段。前三个由代码强校验（[:434-445](../src/gen/env_gen/mcp_tool_gen.py#L434-L445)），不合格直接判失败重试：

| 字段 | 校验 | 作用 |
|---|---|---|
| `scenario_id` | 必须存在 | Phase 4 的 MCP `client_id` 拼在它上面，因此它同时是**并发隔离键** |
| `complexity_level` | 必须 ∈ `simple/medium/complex/boundary` | 唯一的枚举白名单，写别的值整批重生成 |
| `scenario_data` | 必须是 dict | 真正灌给 `load_scenario` 的载荷 |
| `expected_behavior` | 不校验，缺省视作 `"pass"` | 决定 Phase 4 怎么判分 |

注意 `complexity_level` 是**唯一**被枚举约束的字段，而它在 pipeline 里除了透传给验证 prompt 之外没有任何控制作用——它的价值是**逼模型自己声明意图**：一个标了 `complex` 却只放两条记录的场景，人一眼能看出生成质量不对。

#### 复杂度分层：为什么要显式规定配额

如果只说"生成 4 个多样的场景"，模型会给出 4 个规模相近的中等场景——这是 LLM 的均值回归。Prompt 因此写死了每档的配额和实体数量区间（[prompts.py:373-402](../src/gen/env_gen/prompts.py#L373-L402)）：

| 层级 | 配额 | 主实体数 | 意图 | 实测 `scenario_data` 中位数 |
|---|---|---|---|---|
| **simple** | 1-2 个 | 1-2 | 基本 CRUD 能不能跑通 | 641 B |
| **medium** | 2-3 个 | 3-5 | 典型用例 + 少量 edge（零价、午夜时间） | 1.9 KB |
| **complex** | 1-2 个 | 5-10 | 嵌套关系、跨实体引用 | 3.5 KB |
| **boundary** | 按需 | 边界值 | 空集合、极值、**故意的非法数据** | 267 B |

（中位数取自 `envs/intermediate/` 下 80 个已生成环境、共 319 个场景。实际分布几乎总是 simple/medium/complex/boundary 各 1 个——`n_scenarios=4` 恰好把每档压到最小配额。）

这张表最值得注意的是 **complex 档同时被两条相反的指令夹着**：一边要"5-10 个实体、所有字段填满、嵌套结构用足"，一边紧跟一条 "Focus on functional coverage, not data volume"，以及一整节 Data Volume Guidelines 要求查找表只放 3-10 条、大数据集只取 2-5 条代表样本（[prompts.py:414-418](../src/gen/env_gen/prompts.py#L414-L418)）。原因就是上面说的单次长输出——**complex 场景是最容易把这次 LLM 调用撑爆的那一个**。所以这里的"复杂"指的是关系复杂，不是数据量大。

#### expected_behavior：把"失败"也变成可断言的期望

这是 Phase 3 设计里最容易被忽略、但对 Phase 4 判分至关重要的一个字段。

问题在于：Phase 4 判断代码好坏的核心信号是 `load_scenario` 有没有抛异常。如果所有场景都是合法数据，那这个信号只能证明"代码能接受好数据"，完全没验证"代码会不会拒绝坏数据"。而一个把 `pattern` 约束漏掉的 Pydantic 模型，在只喂好数据时表现得和正确模型一模一样。

`expected_behavior` 就是用来补这一半的：

```json
{
  "scenario_id": "scenario_004",
  "complexity_level": "boundary",
  "description": "Invalid record ID format (should be rejected)",
  "expected_behavior": "validation_error",
  "scenario_data": {
    "bases": {"app001": {"id": "INVALID_FORMAT", "name": "X", "tables": {}}},
    "current_time": "2024-01-15T12:00:00Z"
  }
}
```

Phase 4 拿到它之后，判分表被反转（[validate_revise.py:103](../src/gen/env_gen/validate_revise.py#L103)、[prompts.py:534-536](../src/gen/env_gen/prompts.py#L534-L536)）：

| `expected_behavior` | `load_scenario` 成功 | `load_scenario` 抛异常 |
|---|---|---|
| `"pass"`（缺省） | ✅ PASS | ❌ CRITICAL，立即停止该场景后续测试 |
| `"validation_error"` | ❌ FAIL — **校验太松，模型该拒的没拒** | ✅ PASS |

所以第二行右边那个"失败即通过"不是特例处理，它是这个字段存在的全部理由。实测 319 个场景里 26 个标了 `validation_error`，全部落在 boundary 档——也就是说 80 个环境里有 26 个真正验证了自己的拒绝能力，剩下的 boundary 场景走的是"空集合应该被接受"这类正向边界（54 个）。

#### 类型精确匹配：最主要的失败来源

Prompt 里篇幅最大的一节（[prompts.py:341-368](../src/gen/env_gen/prompts.py#L341-L368)，标了 CRITICAL）讲的是同一件事：**照着 Pydantic 的类型逐层填，不要简化。**

会踩的坑就一种形态——把 `Dict[str, SomeModel]` 里的 value 当成标量：

```python
class TicketInfo(BaseModel):
    price: float
    availability: int

class TrainScenario(BaseModel):
    tickets: Dict[str, TicketInfo]
```

```jsonc
// ✅
{"tickets": {"T001": {"price": 100.0, "availability": 50}}}

// ❌ value 退化成字符串
{"tickets": {"T001": "ticket_info"}}
// ❌ 字段名是自己编的，不是 TicketInfo 的
{"tickets": {"T001": {"ticket_id": "T001"}}}
```

prompt 于是把三种复合形态和类型细节都列成了模板：`Dict[str, BaseModel]` → `{"key": {字段...}}`、`List[BaseModel]` → `[{字段...}, ...]`、嵌套模型递归展开；`float` 写 `100.0` 别写 `"100"`，`bool` 写 `true` 别写 `"true"`。

这些约束**代码层面一个都检查不了**——`generate_scenarios` 只验证 `scenario_data` 是不是 dict，里面对不对完全不看。真正的检查发生在 Phase 4 第一次 `load_scenario` 时，由 Pydantic 抛 `ValidationError`。这就是为什么 Phase 4 要区分"代码问题"和"场景问题"：同一个 `ValidationError`，可能是 Phase 2 的模型写错了，也可能是 Phase 3 的数据填错了，光看异常分不出来。

实测 75 个有验证记录的环境、299 个场景，**首轮验证通过率 73.9%**（221/299），分档看：

| 层级 | 首轮通过率 |
|---|---|
| boundary | 83% |
| simple | 75% |
| medium | 69% |
| complex | 69% |

boundary 最高、complex 最低，和"字段填得越多越容易填错"的直觉一致（boundary 多为空集合，几乎无字段可填错）。78 个首轮失败中 73 个是 `expected_behavior="pass"` 却挂掉——真正需要 Phase 4 去修的就是这批。

#### 其余两条约束

**引用完整性。** 场景数据要"realistic, coherent"（[prompts.py:404-412](../src/gen/env_gen/prompts.py#L404-L412)）：ID 唯一、时间区间自洽、跨实体引用必须指向同一份数据里真实存在的对象。Pydantic 检查不到这一层——`event.calendar_id` 只要是个符合 `pattern` 的字符串就通得过，哪怕那个 calendar 不存在。这类"schema 合法但语义悬空"的数据会一路活到 Layer 2，在某个工具查父实体时才炸，而且报出来的错会指向工具实现，很有误导性。

**`random_seed` 必须钉死。** 如果 Section 1 的 Scenario 模型里有 `random_seed`，场景必须给一个固定整数（prompt 建议 42）。这是 Phase 2 那条"把不确定性收进 scenario"的下半场——Phase 2 保证代码读 seed 而不是自己摇，Phase 3 保证 seed 有确定值。实测 319 个场景里 12 个带 `random_seed`，符合"绝大多数环境不需要随机性"的预期。

#### 完整输出示例

```json
[
  {
    "scenario_id": "scenario_001",
    "complexity_level": "simple",
    "description": "Single base with one table and two records",
    "expected_behavior": "pass",
    "scenario_data": {
      "bases": {
        "app001": {
          "id": "app001",
          "name": "Project Management",
          "tables": {
            "Tasks": {
              "rec001": {
                "id": "rec001",
                "fields": {"title": "Design mockup", "status": "In Progress"},
                "created_time": "2024-01-15T10:00:00Z"
              },
              "rec002": {
                "id": "rec002",
                "fields": {"title": "Code review", "status": "Pending"},
                "created_time": "2024-01-15T11:30:00Z"
              }
            }
          }
        }
      },
      "current_time": "2024-01-15T12:00:00Z"
    }
  },
  {
    "scenario_id": "scenario_002",
    "complexity_level": "medium",
    "description": "Multiple bases with various table structures",
    "expected_behavior": "pass",
    "scenario_data": {
      "bases": {
        "app001": { "...": "3 tables, 5 records" },
        "app002": { "...": "2 tables, 3 records" }
      },
      "current_time": "2024-06-20T14:30:00Z"
    }
  },
  {
    "scenario_id": "scenario_003",
    "complexity_level": "boundary",
    "description": "Empty workspace (valid edge case)",
    "expected_behavior": "pass",
    "scenario_data": {
      "bases": {},
      "current_time": "2024-01-01T00:00:00Z"
    }
  },
  {
    "scenario_id": "scenario_004",
    "complexity_level": "boundary",
    "description": "Invalid timestamp format (should be rejected)",
    "expected_behavior": "validation_error",
    "scenario_data": {
      "bases": {},
      "current_time": "2024-13-99T99:99:99Z"
    }
  }
]
```

真实产物可以直接翻 checkpoint：

```bash
python3 -c "
import json
d = json.load(open('envs/intermediate/AirtableMcpServer_checkpoint.json'))
for s in d['scenarios']:
    print(s['scenario_id'], s['complexity_level'],
          s.get('expected_behavior', 'pass'), '|', s['description'])
"
```


---

### Phase 4: 验证与修订循环

**目标**：验证生成的代码，自动诊断并修复错误（最多 3 轮）

**输入**：
- Tool 代码（Phase 2）
- Test Scenarios（Phase 3）
- Tools Metadata（Phase 1）

**Prompt 核心逻辑**（`ScenarioValidator_System_Prompt` + `MCPToolReviser_System_Prompt`）：

#### 三层验证策略

```
┌──────────────────────────────────────────────────────────┐
│                   验证分层架构                            │
└──────────────────────────────────────────────────────────┘

Layer 1: Scenario Loading (CRITICAL - 阻塞性)
┌────────────────────────────────────────────────┐
│ execute_mcp_tool("load_scenario", scenario)   │
│                                                │
│ 成功 → 继续 Layer 2                           │
│ 失败 + expected_behavior="pass" → CRITICAL    │
│      → 立即停止，不测试其他工具                │
│ 失败 + expected_behavior="validation_error"   │
│      → PASS（期望失败）                        │
└────────────────────────────────────────────────┘
                    ↓ (仅成功时)
Layer 2: Tool Execution (条件执行)
┌────────────────────────────────────────────────┐
│ for each tool (除了 load/save):                │
│   - Valid case:    正常输入                    │
│   - Boundary case: 边界值                      │
│   - Error case:    错误输入                    │
│                                                │
│ 验证输出是否匹配 output_schema                 │
└────────────────────────────────────────────────┘
                    ↓
Layer 3: State Consistency (条件执行)
┌────────────────────────────────────────────────┐
│ execute_mcp_tool("save_scenario")             │
│                                                │
│ 比较: saved_scenario == original + mods       │
└────────────────────────────────────────────────┘
```

**Layer 1 示例代码**：
```python
result = execute_mcp_tool(
    tool_name=f"{mcp_server_name}-load_scenario",
    tool_args={"scenario": scenario_data},
    client_id=f"{mcp_server_name}-{request_id}_{scenario_id}"
)

if not result.success:
    if expected_behavior == "validation_error":
        # 期望失败，实际失败 → 通过
        return {
            "passed": True,
            "expected_error": True,
            "message": "Correctly rejected invalid data"
        }
    else:
        # 期望成功，实际失败 → CRITICAL 错误
        return {
            "passed": False,
            "errors": [{
                "error_type": "CRITICAL",
                "error_location": "load_scenario",
                "error_details": result.error_message,
                "root_cause": "Schema validation failed or state initialization error"
            }]
        }
        # ⚠️ 立即返回，不继续测试其他工具
```

#### 修订策略

**问题来源判断**（关键决策！）：
```json
{
  "is_scenario_problem": true/false,
  "scenario_problem_details": "...",
  "problematic_scenario_ids": [...]
}
```

**判断逻辑**：
- `is_scenario_problem=true` → **不修改代码**，场景数据有问题：
  - 场景数据类型不匹配 Pydantic 模型
  - 场景数据有逻辑错误（引用不存在的 ID）
  - `expected_behavior` 设置错误
  
- `is_scenario_problem=false` → **修改代码**，实现有 bug：
  - Pydantic 模型定义不完整
  - `load_scenario` / `save_scenario` 实现错误
  - 工具逻辑错误

**优先级修复**（仅当 `is_scenario_problem=false`）：

| 优先级 | 错误类型 | 影响 | 修复策略 |
|--------|---------|------|---------|
| **CRITICAL** | `load_scenario` 失败<br>Pydantic 验证错误 | 阻塞所有测试 | 优先修复<br>其他问题暂时忽略 |
| **HIGH** | 多场景失败的工具<br>`save_scenario` 不一致 | 影响多个场景 | CRITICAL 修完后处理 |

**修订循环**：
```python
for revision in range(max_revisions):  # max_revisions=3
    # 1. 验证所有场景
    validation_result = validate_all_scenarios(tool_code, scenarios)
    
    # 2. 检查是否全部通过
    if validation_result.all_passed:
        return {"success": True, "final_code": tool_code}
    
    # 3. 聚合错误
    aggregated_errors = aggregate_errors_by_severity(validation_result)
    
    # 4. 判断问题来源
    analysis = analyze_errors(aggregated_errors)
    if analysis.is_scenario_problem:
        print(f"Scenario problems detected: {analysis.problematic_scenario_ids}")
        break  # 不修改代码
    
    # 5. 按优先级修复
    if aggregated_errors["CRITICAL"] > 0:
        print(f"Fixing CRITICAL errors (count: {aggregated_errors['CRITICAL']})")
        revised_code = revise_code_for_critical_errors(tool_code, aggregated_errors)
    else:
        print(f"Fixing HIGH errors (count: {aggregated_errors['HIGH']})")
        revised_code = revise_code_for_high_errors(tool_code, aggregated_errors)
    
    tool_code = revised_code
    save_checkpoint(revision, tool_code, validation_result)

# 达到最大修订次数
return {
    "success": False,
    "final_code": tool_code,
    "revisions": max_revisions,
    "remaining_errors": aggregated_errors
}
```

**输出目录结构**：
```
envs/
├── tools/
│   └── Airtable.py              # 最终验证通过的代码
├── intermediate/
│   └── Airtable_checkpoint.json # 检查点（可恢复）
└── metadata/
    └── Airtable_metadata.json   # 原始 schema

configs/
└── mcp_server.json              # MCP 服务器注册配置

log/
├── log.jsonl                    # 详细日志
└── usage.jsonl                  # Token 使用统计（可选）
```

---

## 🚀 完整使用示例

### 示例 1: 从头生成单个环境

```bash
# Step 1: 生成 metadata
python -m src.gen.mcp_schema_gen envs/schema_sketch/booking_hotel_server.py

# Step 2: 生成完整环境（自动执行所有阶段）
python -m src.gen.env_gen envs/metadata/BookingHotel_metadata.json \
  --n-scenarios 8 \
  --max-revisions 5

# 输出：
# - envs/tools/BookingHotel.py
# - envs/intermediate/BookingHotel_checkpoint.json
# - log/log.jsonl
```

### 示例 2: 批量并行生成

```bash
# 并行处理所有 metadata 文件
python -m src.gen.env_gen envs/metadata/*.json \
  --max-concurrent-files 5 \
  --max-concurrent-scenarios 10

# 输出汇总：
# ✓ GoogleMaps: completed (6/6 scenarios passed)
# ✓ Twitter: completed (6/6 scenarios passed)
# ⚠ BookingHotel: incomplete (4/6 scenarios passed)
```

### 示例 3: 从检查点恢复

```bash
# 如果生成中断，从检查点恢复
python -m src.gen.env_gen --resume envs/intermediate/BookingHotel_checkpoint.json
```

### 示例 4: 使用自定义模型

```bash
# 使用 DeepSeek 模型
python -m src.gen.env_gen envs/metadata/Calendar_metadata.json \
  --model deepseek \
  --n-scenarios 10

# 使用 Kimi K2
python -m src.gen.env_gen envs/metadata/Calendar_metadata.json \
  --model kimi-k2
```

---

## 📊 数据流与文件组织

### 目录结构

```
envfactory_repro/
├── envs/
│   ├── schema_sketch/          # 原始 schema 骨架代码
│   │   ├── calendar_server.py
│   │   ├── booking_hotel_server.py
│   │   └── ...
│   ├── metadata/               # 标准化的 metadata（Phase 1 输出）
│   │   ├── Calendar_metadata.json
│   │   ├── BookingHotel_metadata.json
│   │   └── ...
│   ├── tools/                  # 生成的 MCP 服务器代码（Phase 2 输出）
│   │   ├── Calendar.py
│   │   ├── BookingHotel.py
│   │   └── ...
│   └── intermediate/           # 检查点文件（用于恢复）
│       ├── Calendar_checkpoint.json
│       └── ...
├── src/gen/
│   ├── mcp_schema_gen.py       # Phase 1: Schema 生成
│   ├── prompts.py              # Schema 生成的 Prompts
│   └── env_gen/
│       ├── env_gen.py          # 流程编排
│       ├── mcp_tool_gen.py     # Phase 2 & 3: Tool 和 Scenario 生成
│       ├── validate_revise.py  # Phase 4: 验证与修订
│       ├── prompts.py          # Tool/Scenario/Validation Prompts
│       └── types.py            # 数据类型定义
├── configs/
│   └── mcp_server.json         # MCP 服务器注册配置
└── log/
    ├── log.jsonl               # 详细日志
    └── usage.jsonl             # Token 使用统计（可选）
```

### 数据流图

```
                    ┌─────────────────────┐
                    │   API 文档 / Sketch  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    SchemaGen        │
                    │  (LLM 提取结构)      │
                    └──────────┬──────────┘
                               │
                         metadata.json
                               │
                               ▼
                    ┌─────────────────────┐
                    │   MCPToolGen        │
                    │  (代码生成)          │
                    └──────────┬──────────┘
                               │
                          tool_code.py
                               │
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
   ┌─────────────────────┐          ┌──────────────────────┐
   │   ScenarioGen       │          │  MCPClientManager    │
   │  (测试场景生成)      │          │  (注册 MCP 服务器)    │
   └──────────┬──────────┘          └──────────┬───────────┘
              │                                 │
    checkpoint.scenarios[]                      │
              │                                 │
              └──────────────┬──────────────────┘
                             │
                             ▼
                  ┌────────────────────┐
                  │ ValidateReviseGen  │
                  │ (验证与修订循环)    │
                  └─────────┬──────────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
     ✓ 所有场景通过              ⚠ 达到最大修订次数
     validated_env              partially_validated_env
```

---

## ⚙️ 配置选项

### EnvGenConfig

```python
from src.gen.env_gen import EnvGenConfig

config = EnvGenConfig(
    # 基础模型配置
    model_name="kimi",              # 使用的模型：kimi, deepseek, qwen, claude 等
    temperature=1.0,                # 生成温度
    top_p=0.95,                     # Nucleus 采样
    presence_penalty=1.5,           # 重复惩罚
    
    # 生成配置
    n_scenarios=4,                  # 每个环境生成的场景数量
    max_revisions=3,                # 最大修订次数
    max_turns=50,                   # LLM 对话最大轮数
    
    # 并行配置
    max_concurrent_files=3,         # 并行处理的 metadata 文件数
    max_concurrent_scenarios=5,     # 并行验证的场景数
    
    # 日志配置
    log_folder="log/log.jsonl",
    enable_log_thinking_content=True,
    enable_dump_token_usage=False,
    usage_dump_path="log/usage.jsonl",
    
    # 中间结果
    save_intermediate=True,         # 保存检查点
    intermediate_dir="envs/intermediate"
)

env_gen = EnvGen(config=config)
```

### 命令行参数

```bash
python -m src.gen.env_gen [metadata_paths] [options]

参数:
  metadata_paths              一个或多个 metadata 文件路径（支持通配符）

选项:
  --n-scenarios N            生成的场景数量（默认: 4）
  --max-revisions N          最大修订次数（默认: 3）
  --max-concurrent-files N   并行处理文件数（默认: 3）
  --max-concurrent-scenarios N  并行验证场景数（默认: 5）
  --model MODEL              使用的模型（默认: kimi）
  --no-intermediate          不保存中间结果
  --resume CHECKPOINT        从检查点恢复
```

---

## 🎨 设计亮点

### 1. LLM 作为编译器

整个系统将 LLM 视为"从规范生成实现"的编译器：

```
API Spec  → [Schema Compiler]  → Metadata
Metadata  → [Code Generator]   → MCP Server
MCP Server → [Test Generator]  → Scenarios
(Code + Scenarios) → [Validator] → Validated Code
```

这种设计使得生成过程：
- **可组合**：每个阶段独立可测试
- **可调试**：中间产物都可检查
- **可迭代**：失败阶段可单独重试

### 2. 状态快照机制

通过 `load_scenario` / `save_scenario`，环境支持：

```python
# RL 训练的典型用法
initial_state = {"calendars": {...}, "current_time": "..."}
env.load_scenario(initial_state)  # 重置环境

for step in range(max_steps):
    observation = env.observe()
    action = agent.act(observation)
    reward = env.step(action)

final_state = env.save_scenario()  # 保存轨迹
```

这使得：
- **可重现**：相同初始状态 → 相同行为
- **可并行**：不同 worker 独立运行
- **可回溯**：保存任意时刻状态

### 3. 自我修复能力

验证-修订循环实现自动调试：

```python
while revisions < max_revisions:
    errors = validate(code, scenarios)
    if no_errors:
        return code
    
    # 自动诊断并修复
    analysis = analyze_errors(errors)
    if analysis.is_scenario_problem:
        break  # 场景问题，不修代码
    
    code = revise_code(code, analysis)
    revisions += 1
```

能自动修复：
- Pydantic 模型定义错误
- 状态管理不完整（`save_scenario` 丢字段）
- 工具逻辑错误（返回值不匹配 schema）

### 4. 检查点恢复

```json
{
  "metadata": {
    "class_name": "GoogleCalendar",
    "state": "VALIDATE_SCENARIOS",
    "last_updated": "2024-01-15T10:30:00"
  },
  "schema": {...},
  "tool_code_history": [
    {"timestamp": "...", "code": "...", "revision": 0},
    {"timestamp": "...", "code": "...", "revision": 1}
  ],
  "scenarios": [...],
  "validation_results": [...]
}
```

支持：
- 从任意阶段恢复
- 保留完整历史
- 避免重复计算

---

## 🐛 常见问题

### 1. load_scenario 验证失败

**现象**：
```
CRITICAL: load_scenario failed with ValidationError
```

**原因**：
- Pydantic 模型定义不完整
- 场景数据类型不匹配

**解决**：
检查 `tool_code` 的 Section 1：
```python
# 确保所有字段都有定义
class Scenario(BaseModel):
    field1: str = Field(...)  # 缺失会导致加载失败
    field2: int = Field(default=0)  # 可选字段需要 default
```

### 2. save_scenario 状态不一致

**现象**：
```
State inconsistency: saved scenario missing field 'xxx'
```

**原因**：
`save_scenario` 返回的字典不完整

**解决**：
确保返回所有状态字段：
```python
def save_scenario(self) -> dict:
    return {
        "field1": self.field1,
        "field2": self.field2,
        # 不要遗漏任何字段！
    }
```

### 3. 场景生成类型不匹配

**现象**：
```
Expected Dict[str, Event], got {"key": "value"}
```

**原因**：
LLM 生成的场景数据结构不符合 Pydantic 模型

**解决**：
在 Prompt 中明确类型要求，或手动修正场景数据

### 4. 修订循环无法收敛

**现象**：
```
Reached max_revisions (3) but still failing
```

**原因**：
- 错误过于复杂，LLM 无法修复
- Prompt 质量不足

**解决**：
1. 检查 `log/log.jsonl` 查看具体错误
2. 手动修改 `envs/tools/*.py`
3. 增加 `--max-revisions` 或改进 Prompt

---

## 🔍 调试技巧

### 1. 查看详细日志

```bash
# 查看最近的生成日志
tail -f log/log.jsonl | jq .

# 提取特定 conversation 的日志
jq 'select(.conversation_id == "env_gen_Calendar")' log/log.jsonl
```

### 2. 检查点内容

```python
import json
with open("envs/intermediate/Calendar_checkpoint.json") as f:
    checkpoint = json.load(f)

print("State:", checkpoint["metadata"]["state"])
print("Revisions:", len(checkpoint["tool_code_history"]))
print("Validation:", checkpoint["validation_results"][-1])
```

### 3. 手动测试 MCP 服务器

```bash
# 启动生成的 MCP 服务器
cd envs/tools
python Calendar.py

# 在另一个终端测试
python -c "
from src.manager.mcp_client_manager import MCPClientManager
manager = MCPClientManager()
manager.register_MCP_server('Calendar', 'envs/tools/Calendar.py', False)
result = manager.call_tool('Calendar-list_calendars', {}, 'test-client')
print(result)
"
```

### 4. Token 使用统计

```bash
# 启用 token 统计
python -m src.gen.env_gen envs/metadata/Calendar_metadata.json \
  --enable-dump-token-usage

# 查看使用情况
jq -s 'group_by(.agent_name) | map({agent: .[0].agent_name, total: (map(.total_tokens) | add)})' \
  log/usage.jsonl
```

---

## 📚 相关文档

- [README.md](../README.md) - 项目概述和快速开始
- [src/gen/prompts.py](../src/gen/prompts.py) - Schema 生成的 Prompts
- [src/gen/env_gen/prompts.py](../src/gen/env_gen/prompts.py) - Tool/Scenario/Validation Prompts
- [.agents/skills/mcp-sketch-discovery/SKILL.md](../.agents/skills/mcp-sketch-discovery/SKILL.md) - Schema Sketch 发现流程

---

## 🎓 最佳实践

### 1. 从简单开始

- 先用 1-2 个工具的小 API 测试流程
- 验证生成质量后再批量处理

### 2. 渐进式调整

```bash
# 第一轮：少量场景，快速验证
python -m src.gen.env_gen metadata.json --n-scenarios 3

# 第二轮：增加场景数量
python -m src.gen.env_gen metadata.json --n-scenarios 10

# 第三轮：批量并行
python -m src.gen.env_gen envs/metadata/*.json --max-concurrent-files 5
```

### 3. 监控生成质量

```bash
# 定期检查日志
grep -E "(CRITICAL|ValidationError)" log/log.jsonl

# 统计成功率
python -c "
import json
results = []
with open('log/log.jsonl') as f:
    for line in f:
        data = json.loads(line)
        if 'final_validation' in data:
            results.append(data)
success = sum(1 for r in results if r.get('success', False))
print(f'Success rate: {success}/{len(results)} ({success/len(results)*100:.1f}%)')
"
```

### 4. 自定义 Prompt

如果生成质量不理想，调整 Prompts：

```python
# 在 src/gen/env_gen/prompts.py 中修改
MCPToolGenerator_System_Prompt = '''
# 添加你的自定义指令
...
'''
```

---

## 📝 总结

EnvFactory 的环境生成系统通过五阶段流水线实现了：
- **自动化**：从 API 文档到可执行环境的端到端生成
- **质量保证**：多层验证 + 自动修订机制
- **可扩展性**：并行处理 + 检查点恢复
- **灵活性**：支持多种 LLM 和自定义配置

核心思想是**将 LLM 作为编译器，结合传统软件工程的验证机制**，形成一个高质量、可维护的环境工厂。

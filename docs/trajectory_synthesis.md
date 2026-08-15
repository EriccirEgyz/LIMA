# EnvFactory 工具使用轨迹合成详解（Tool-use Trajectories Synthesis）

## 📋 概述

[环境生成](environment_generation.md) 交付的是一个个**可执行的空环境**：MCP 服务器代码 + 状态管理机制。但训练 Agent 需的不是环境本身，而是**环境里发生过的成功交互**——"用户说了什么、Agent 调了哪些工具、环境返回了什么、最终怎么回答"的完整轨迹。

轨迹合成流水线要解决的核心问题是：**如何大规模造出"任务真实、执行正确、难度可控"的工具使用轨迹？** 直接让 LLM 凭空写对话，会出现任务不可解（要的参数环境里没有）、执行错误（工具调用是编的）、千篇一律（全是"帮我查天气"）这三类问题。

EnvFactory 的做法是**倒着来**：先从工具图里采样出一条**可执行的工具链**（保证任务一定可解），再让 LLM 为这条链**编一个人类动机**（保证任务自然），然后让一个真正的 Agent 在真实 MCP 环境里**实际执行**（保证轨迹每一行都真实发生过），最后从 pass@k 条候选里**挑出最好的一条**（保证质量）。

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     轨迹合成：从工具图到训练数据                                │
└──────────────────────────────────────────────────────────────────────────────┘

   Step 0            Step 1           Step 2            Step 3           Step 4
┌───────────┐      ┌──────────┐      ┌───────────┐      ┌──────────┐      ┌───────────┐
│ ToolGraph │ ───> │ 工具链   │ ───> │ 场景规划  │ ───> │ 初始状态 │ ───> │ 任务生成  │
│ 构建      │      │ 采样     │      │ + 分轮    │      │ 填充     │      │ + 精炼    │
└───────────┘      └──────────┘      └───────────┘      └──────────┘      └───────────┘
 graph.pkl          ToolQueryChain     scenario +         每个 server        query +
 (623 工具)         (骨架)             turns[]            一份 scenario      user_intent
                                                                 │
                                                                 ▼
   Step 8            Step 7           Step 6
┌───────────┐      ┌──────────┐      ┌───────────────────────────────────────────┐
│ 数据转换  │ <──  │ 落盘     │ <──  │ Step 5: pass@k 求解 + 筛选                 │
│ SFT / RL  │      │ JSON     │      │ Agent 在真实 MCP 环境执行 → 评分选优       │
└───────────┘      └──────────┘      └───────────────────────────────────────────┘
```

**核心设计理念**：
- 🔗 **可解性靠构造保证，不靠事后过滤**：工具链从图上采样时已验证"每个参数都有来源"
- 🎭 **LLM 负责"编故事"，环境负责"讲事实"**：任务由 LLM 生成，但轨迹里的每次工具调用都真实执行
- 🎲 **pass@k + 评估选择**：同一任务并行解 k 次，用一个独立裁判 Agent 挑最优解
- 🔄 **状态跨轮演化**：第 t 轮结束时的环境状态就是第 t+1 轮的初始状态，多轮任务前后连贯

> 📖 **与论文的对应关系**（EnvFactory, arXiv:2605.18703, §3.3-3.5）：论文把本流水线称为 **QueryGen**，把可解性约束表述为 *"All required input parameters of a sampled tool must be either externally provided by the human user or internally derived from the outputs of previously sampled tools"*；四条改写技法对应论文的 **calibrated refinement**；pass@k 对应 *independently generate k candidate solution trajectories*。论文数据：85 环境 / 842 工具 / 2,575 条轨迹（1,622 SFT + 953 RL），平均每条对话 4.82 轮、每轮 3.29 步——注意仓库当前规模（75+ 服务器、623 工具）与论文发表时已有差异。

### 核心组件

| 阶段 | 组件 | 文件路径 | 职责 |
|------|------|---------|------|
| **Step 0** | `ToolGraph` | [src/graph/tool_graph.py](../src/graph/tool_graph.py) | 构建工具-参数异构图 |
| **Step 0** | Samplers | [src/graph/sampler.py](../src/graph/sampler.py) | 从图上采样可执行工具链 |
| **Step 2-6** | `QueryGen` | [src/gen/query_gen/__init__.py](../src/gen/query_gen/__init__.py) | 状态机路由（对话/非对话两个实现） |
| **Step 2-6** | `QueryGenNonConv` | [src/gen/query_gen/query_gen_non_conv.py](../src/gen/query_gen/query_gen_non_conv.py) | 非对话模式实现（默认） |
| **Step 5** | `QueryGenConv` | [src/gen/query_gen/query_gen_conv.py](../src/gen/query_gen/query_gen_conv.py) | 对话模式：用户模拟器 + 用户侧工具 |
| **Step 2-6** | Prompts | [src/gen/query_gen/prompts.py](../src/gen/query_gen/prompts.py) | 全部 7+ 个 Agent 的提示词 |
| **Step 8** | `data_process` | [src/utils/data_process.py](../src/utils/data_process.py) | 转成 SFT / RL 训练格式 |

---

## 🎯 核心概念

### ToolQueryChain：一条轨迹的完整档案

轨迹合成的所有中间产物都挂在 [ToolQueryChain](../src/graph/tool_chain.py#L87) 上，最终也以它为单位落盘：

```python
ToolQueryChain
├── init_tool_chain        # 采样的工具链（骨架，仅工具名）
├── seed                   # 采样种子（可复现）
├── scenario               # ScenarioPlanner 编的故事（用户画像 + 情境）
├── user_tools             # 分类出的"只能由人类执行"的工具
└── tool_chain[]           # 每轮一个 ToolQueryNode：
    ├── raw_tool_call      #   本轮目标工具（骨架）
    ├── initial_scenario   #   本轮开始时各 server 的状态
    ├── query / user_intent#   生成的用户请求 + 意图分析
    ├── steps[]            #   ★ 最终轨迹：user/assistant/tool_call/tool_response 交替
    ├── final_scenario     #   本轮结束时的环境状态（save_scenario 导出）
    └── decision/accuracy  #   裁判判定：本轮是否有效 / pass@k 通过率
```

注意 `initial_scenario` / `final_scenario` 这对字段：它们就是环境生成阶段 `load_scenario` / `save_scenario` 机制在数据合成里的用途——每轮开始把状态灌进去，结束时导出来，既是下一轮的输入，也是 RL 阶段重放环境的依据。

### 状态机：轨迹生成的主循环

[QueryGenNonConv.gen()](../src/gen/query_gen/query_gen_non_conv.py#L622) 是一个显式状态机，每轮对话（turn）走一遍：

```
Preparing ──> Starting ──> Generating ──> (Refining) ──> Solving ──> Terminated
(场景+分轮)   (灌初始状态)  (生成 query)    (可选:精炼)     (pass@k+选优)  (落盘)
                    ▲                                        │
                    └──── 下一轮：以上一轮 final_scenario ────┘
                          为初始状态，重复 Generating~Solving
```

任何一个状态失败（比如初始场景灌不进去、所有 k 条解全被否决），该轮 `decision=False`，**整条链到此截断**——后面的轮次不再生成。这保证了落盘的轨迹不会有"前半段是好的、后半段是编的"的情况。

---

## 🔧 详细流程

### Step 0: ToolGraph 构建

#### 这一步在解决什么问题

轨迹合成的第一步是回答"**哪些工具组合在一起能构成一个可解的任务**"。623 个工具、75 个 MCP 服务器，任意拼 5 个工具大概率拼出一个不可解的任务——比如 `book_flight` 需要 `flight_id`，但前面没有任何工具产出这个参数。

ToolGraph 把这个问题变成图问题：**工具和参数作为节点，"谁能给谁供货"作为边**。之后采样工具链时，每个参数是否有着落就可以直接在图上验证。

**执行**（完整 notebook 见 [examples/load_tool_graph.ipynb](../examples/load_tool_graph.ipynb)）：

```python
from src.graph.tool_graph import ToolGraph

tool_graph = ToolGraph()                       # 阈值: build_edge=0.85, merge_param=0.92
tool_graph.build_tool_graph(
    metadata=metadatas,                        # envs/metadata/*.json 的列表
    enable_merge=False,
    enable_build_edge_with_llm=True,
)
tool_graph.save("graph.pkl")                   # 构建一次约 30 分钟，之后秒级 load
```

#### 图里有什么

[build_tool_graph()](../src/graph/tool_graph.py#L68) 构建的是一张**二部有向图**（实测规模：4781 节点 / 10620 边）：

```
┌─────────┐  Tool_Output   ┌───────────────┐  Tool_Input   ┌─────────┐
│ Tool A  │ ─────────────> │ Parameter(X)  │ ────────────> │ Tool B  │
└─────────┘                └───────────────┘               └─────────┘
     │                                                            ▲
     └──────────── Tool_Depend（相似度/LLM 推断）──────────────────┘

四种边：
  Tool → Param      Tool_Output     工具产出这个参数
  Param → Tool      Tool_Input      工具消费这个参数（边上带 required 标记）
  Param → Param     Parameter_Relate 语义相同的输入/输出参数（如 A 的 output 对齐 B 的 input）
  Tool → Tool       Tool_Depend     综合依赖（供采样走 BFS）
```

#### 建边的三种手段

**① 向量相似度建边**（[_build_edge_with_sim()](../src/graph/tool_graph.py#L295)）：所有参数过一遍 embedding（`.env` 里配 `EMBEDDING_URL`，如 `BAAI/bge-m3`），然后算"每个工具的输出参数 × 全体工具的输入参数"的余弦相似度矩阵，超过 `build_edge_threshold=0.85` 的就对齐起来：加 Tool→Tool 依赖边 + Param→Param 关联边。这是跨服务器建边的主要来源。

一次命中为什么**同时加两种边**？因为它们服务两个不同的消费者，各答一个问题：

| 边 | 回答的问题 | 消费者 |
|---|---|---|
| Tool→Tool (`Tool_Depend`) | "B 可以排在 A 后面吗" | [sample()](../src/graph/tool_graph.py#L553) 的 BFS 直接走 `successors()`；也是 LLM 补边（②）的唯一载体——无参数工具（如 `delete_all_notes`）的语义依赖只能以这种形式存在 |
| Param→Param (`Parameter_Relate`) | "A 的输出字段 x 能填 B 的输入字段 y 吗" | [validate_parameter()](../src/graph/tool_graph.py#L339) 的逐参数可解性校验、TopologySampler 的 [_get_priors()](../src/graph/sampler.py#L94) 前驱反查 |

前者管**链的连通性**（走得到），后者管**链的可解性**（填得上）。Tool→Tool 边不记录靠哪个参数连接，单独用它无法保证 B 的必填参数真的有人供货；反过来只有 Param→Param 边，遍历要四跳才能找到下一个候选工具。

**② LLM 补边**（[_build_edge_with_llm()](../src/graph/tool_graph.py#L244)）：相似度抓不住语义依赖（比如 `create_event` → `delete_event` 参数名对不上但有先后关系）。所以对**每个服务器内部**，把全部工具描述 + 当前邻接表塞给 LLM，让它按需增补。Prompt（[Build_Tool_Dependency_Prompt](../src/graph/prompts.py#L62)）的核心约束是**只增不删**：

> - You are given... Current Adjacency Map... **DO NOT MODIFY EXISTING ENTRIES**
> - 只对未连接的候选对 (Tool A → Tool B) 评估四条标准：**Semantic Complementarity**（是否同属一个任务管线）/ **Data Flow Feasibility**（A 的输出能否喂给 B）/ **Workflow Plausibility**（真人会不会先 A 后 B）/ **Parameter Alignment**（域和输入输出是否对齐）

输出 `<adjacency_map>` 只包含新增边，代码校验工具名合法后合入图。

> ⚠️ **论文与代码的一个差异**：论文 §3.3.1 描述这一步会 *"identify missing logical dependencies **and prune spurious edges**"*（同时剪掉相似度引入的伪边），但当前代码的 prompt 明确禁止删边（**DO NOT MODIFY EXISTING ENTRIES**），实现是**只增不删**。复现时以代码为准；若相似度阈值 0.85 放得太低想清理伪边，需要自己改 prompt 和合入逻辑。

**③ user_provided 推断**：每个参数节点还要标注"**这个值是不是用户会直接说出来的**"（[Get_User_Provided_Prompt](../src/graph/prompts.py#L1)）。这决定了任务可解性的另一半——一个参数只要有"用户会提供"或"前面工具会产出"两个来源之一，任务就可解。Prompt 给的判定测试非常克制：

> - Can you **point to a specific word or phrase** in a hypothetical user sentence that maps DIRECTLY to this parameter?
> - `destination`: User says "Fly to Paris" → **True**
> - `latitude/longitude`: User says "London", system converts → **False**
> - `order_id` (UUID): 用户会说"买红鞋"但不会打出 UUID → **False**
> - **When in doubt, default to `false`**

宁紧勿松：把 `latitude` 误标成用户可提供，采出来的链就会包含一个"用户报出坐标"的诡异任务；反之只是少采一些链，没有损失。

---

### Step 1: 工具链采样（Topology-Aware Sampling）

#### 这一步在解决什么问题

有了图，采样就是把"一个可解的多工具任务"变成"图上一条参数有着落的路径"。但**朴素随机游走做不到**，论文 §3.3.2 指出它的两个失败模式：

1. **只产出串行链**：每步只走一个后继，而真实任务常是"先并行查两类信息，再综合操作"的分叉结构；
2. **依赖解不干净**：游走到 `book_hotel` 时，它需要的 `hotel_id` 可能还没有任何已采样工具产出——任务天生不可解，尤其当一个工具需要**多个**前驱共同供货时。

TopologySampler 用一条**不变式**代替游走（论文表述）：

> All required input parameters of a sampled tool must be either **externally provided** by the human user or **internally derived** from the outputs of previously sampled tools.

两种采样器（[sampler.py](../src/graph/sampler.py)）就是"是否维护这条不变式"的区别：

| 采样器 | 策略 | 产出 |
|---|---|---|
| `RandomWalkSampler` | 每步只随机挑 1 个后继，不管参数 | 细长串行链，可解性无保证（对照用） |
| `TopologySampler`（默认） | 采样每个工具前**递归补齐依赖**，再随机选 1~k 个后继 | 分叉的、多服务器的复合任务，天生可解 |

#### 采样主循环：先补依赖，再走邻居

入口 [ToolGraph.sample()](../src/graph/tool_graph.py#L528)：从随机工具出发做 BFS，每个节点经历两个阶段，直到凑满 `max_nodes`（示例里是 15）：

```
queue = [随机起点]
while queue 且 visited < max_nodes:
    node = queue.popleft()
    ① sample_prior(node)     # 阶段一：node 的必填参数若无人供货 → 反查并递归补入供货工具
    visited.append(node)
    ② sample(node)           # 阶段二：从后继中随机选 1~k 个入队（分叉的来源）
       └─ 无后继可达时：从同 server 随机挑一个工具入队（保证链能续上）
```

#### 阶段一：递归解依赖（sample_prior）

[sample_prior()](../src/graph/sampler.py#L133) 对当前工具的每个输入参数过一遍判定，来源是否成立由 [validate_parameter()](../src/graph/tool_graph.py#L339) 检查（`user_provided=True`，或已访问工具的某个输出参数经 `Parameter_Relate` 边对齐到它）：

```
对每个输入参数 param：
  ├─ 可选参数？   → 60% 概率跳过（不强制补齐可选依赖）
  ├─ 已有着落？   → 90% 概率跳过；★10% 概率仍引入一个额外供货工具（论文的 diversity 手法：
  │                  "stochastically introduce a prior tool for a resolvable parameter
  │                   with a small probability p"，制造冗余供给/多解路径）
  └─ 无着落（必填）→ 沿 Parameter_Relate 反查能产出它的工具 prior（_get_priors()）
        ├─ 递归补 prior 自己的依赖（深度 ≤ max_recursion_depth=5）
        └─ prior 只能从已访问工具或 ≤3 个已用 server 里选（max_servers 约束）
```

一个最小示例——这条不变式怎么落地：

```
起点: Hotels-book_hotel（必填 hotel_id；guest_name 是 user_provided）
  hotel_id 无人供货 → 反查：Hotels-search_hotels 产出 hotel_id
    search_hotels 的参数 city 是 user_provided → 递归到顶
结果: [search_hotels, book_hotel]   ← 每个 required 参数都有着落，任务天生可解
```

#### 阶段二：分叉扩展与边界控制

[sample()](../src/graph/sampler.py#L186) 从后继中 `randint(1, len(candidates))` 均匀抽取——**这就是"非线性"的来源**：一个节点可以分出多条支线，对应"一次请求里并列完成几件事"的自然任务形态（论文：*"This branching mechanism enables non-linear tool-use patterns beyond simple sequential chains"*）。

三个边界参数各自控制一种失衡：

| 参数 | 值 | 不设会怎样 |
|---|---|---|
| `max_nodes` | 15 | 单链无限长，任务超出多轮对话容量 |
| `max_servers` | 3 | 任务横跨太多环境——既不真实（真人不会一个请求调 7 个 App）也撑爆 Agent 上下文；对阶段一（选 prior）和阶段二（选邻居）**同时生效** |
| `max_recursion_depth` | 5 | 依赖链过长时递归不封底，采样被个别工具的深依赖绑架 |

而 60% / 10% 这两个概率控制的是**任务难度的分布**：可选依赖有时补有时不补、已满足的参数偶尔加个冗余供给方，产出的任务便在"参数齐全、一步到位"和"要先查再改、链式取数"之间自然分层，而不是清一色的最短路径。

采样产物 `ToolQueryChain` 里只有工具名列表——**此时它是一条没有剧情的骨架**。

---

### Step 2: 场景规划与分轮（ScenarioPlanner）

#### 这一步在解决什么问题

骨架 → 真实任务之间缺一层**人类动机**。"Calendar-create_event → Calendar-list_events" 是程序员视角的调用序列；用户视角应该是"香港的营销主管 Lisa 明天 10 点要见客户，想确认日程有没有撞"。

[ScenarioPlanner](../src/gen/query_gen/prompts.py#L347)（代码入口 [prepare()](../src/gen/query_gen/query_gen_non_conv.py#L156)）拿到工具链的 trace，反推一个能**自然地 motivate 这串调用**的故事：

> # Scenario Design
> - Define a realistic user persona (name, age range, occupation, location, relevant traits)
> - Establish a concrete situation with time/place/context that explains *why* the user would perform these actions
> - Flow logically from initial need → actions taken → implied next steps
> - **Never mention tools, APIs, or technical mechanisms**—describe only human behaviors and motivations
> - Be specific and grounded (avoid generic phrases like "a user wanted information")

两条核心逻辑：

1. **"Never mention tools"** 是这道 prompt 的灵魂。故事只准讲人的行为和动机，因为这份 scenario 之后要同时喂给任务生成器和用户模拟器——一旦里面出现"用户调用 create_event"，生成的对话就会有机器味。
2. **具体性约束**（persona 五要素 + 反泛化示例）针对的是 LLM 的均值回归：不压着，所有故事都会退化成"A user wants to manage their schedule"。

**分轮（turn splitting）**：一条 15 个工具的链不会发生在一个回合的对话里。[ScenarioPlanner_System_Prompt_Split_Turn](../src/gen/query_gen/prompts.py#L309) 版本会额外输出 `<turn>[[0,1],[2,3,4],[5]]</turn>`——按"意图完成点 / 话题切换点"把工具索引切成组，每组对应一轮"用户请求 → Agent 执行 → Agent 回复"。允许它**重排工具顺序**以符合真实交互节奏。preset 配置里 `enable_split_turns=False` 时则退化为代码里的概率切分（[split_turns()](../src/gen/query_gen/query_gen_non_conv.py#L89)），每轮最多 `max_turn=5` 个工具。

---

### Step 3: 初始状态填充（SchemaGenerator）

#### 这一步在解决什么问题

故事里说"Lisa 有 3 个日程、明天 10 点有客户会议"——环境里得**真的有**这些数据，任务才执行得起来。这一步给链上每个 MCP server 生成一份符合其 `Scenario_Schema`（就是环境生成 Phase 2 的 Section 1 Pydantic 模型）的初始状态。

[SchemaGenerator](../src/gen/query_gen/prompts.py#L374)（代码入口 [schema_generate()](../src/gen/query_gen/query_gen_non_conv.py#L193)）的核心约束：

> 1. **Contextual Design**: Create data that authentically fits the scenario... ensure logical consistency across all generated data（跨 server 一致：日历里的会议要和航班服务器里的航班对得上）
> 2. **Schema Compliance**: Strictly adhere to the target schema... **If strict compliance proves impossible, prioritize generating a simpler, valid subset**
> 4. **List Constraints**: For array fields, include **at most 5 items**（防止单次输出爆炸——和环境生成 Phase 3 同一个教训）

两个值得注意的工程细节：

**逐 server 串行生成**。一个任务最多涉及 3 个 server，但每次调用只填一个，且把**已生成的 schema 作为 "Previously Generated Schemas" 传给下一次**（[:199-213](../src/gen/query_gen/query_gen_non_conv.py#L199-L213)）——这就是"跨 server 一致性"的实现方式：后填的能看到先填的，主动对齐。串行是为了上下文长度，牺牲速度换一致性。

**用真实环境做验证器**。生成的 schema 不是看起来对就行，而是真的调 `load_scenario` 灌进 MCP server（[:228-238](../src/gen/query_gen/query_gen_non_conv.py#L228-L238)），Pydantic 校验不过就带着报错重试。这里没有再写任何解析/校验代码——**环境本身就是校验器**，这是整条流水线反复出现的模式。

```python
tool_response = MCPManager.load_scenario(
    scenario=output_json['schema'], client_id=client_id, check=True
)
if "Successfully" not in tool_response:
    raise ValueError(...)   # 报错信息进 prompt，下一轮重试
```

---

### Step 4: 任务生成与精炼（QueryGenerator / QueryRefiner）

#### QueryGenerator：从工具链反推用户请求

这是整个流水线最"反直觉"的一步：普通数据合成是"给任务 → 想调用"，这里是"**先有调用 → 反推什么话会让人这么调**"。（代码入口 [generate()](../src/gen/query_gen/query_gen_non_conv.py#L261)，prompt 在 [QueryGenerator_System_Prompt](../src/gen/query_gen/prompts.py#L416)）

它看到的上下文：各 server 的初始状态（**包括完整的环境内部数据**）、工具签名、scenario、之前的对话。要求输出 `<query>` 和 `<user_intent>`。三条核心逻辑：

**① "directly and exclusively motivate"**——生成的 query 要恰好 motivating 目标工具调用：不能漏（漏了 Agent 就不会调目标工具），不能多（多了 Agent 会调目标之外的工具，偏离骨架）。

**② Solvability 约束**（[QueryGenerator_Solvability](../src/gen/query_gen/prompts.py#L411)，非对话模式必带）：

> - For each parameters required by the target tool calls, the generated query should either **explicitly state it or implicitly contain it** through references to previous conversation or logical inference.

这条把 Step 0 的可解性保证闭环了：图上保证参数"存在来源"，这条保证来源真的被表达——`order_id` 该由前轮产出，用户就不该在 query 里说；`destination` 是 user_provided，query 里就必须出现目的地。

**③ 自然性**：第一人称、口语化、**禁止出现工具名和参数名**、用 "the hotel you found earlier" 这类指代衔接上下文。Query 是给训练数据当 user 输入的，出现 `call search_flights(origin=...)` 就把工具 schema 泄漏进了模型该自己推理的部分。

#### QueryRefiner：把简单问题改写成难问题

生成的 query 往往太"贴心"——参数全给、步骤全说。`enable_query_refinement=True`（`SFT_CONV` 预设）时，[QueryRefiner](../src/gen/query_gen/prompts.py#L465) 再做一步**隐式化改写**，四条技法各有示例：

| 技法 | 做什么 | 示例 |
|---|---|---|
| **Implicit Reference** | 显 ID → 指代；可推断的参数直接省略 | "Check ticket TKT-789" → "Check the ticket I opened this morning" |
| **Action Compression** | 逻辑上必须先做的中间步骤不提 | "查订单历史然后删最近的" → "删最近的订单"（必须先查） |
| **Ambiguity Introduction** | 加隐含约束 | "订纽约到伦敦的机票" → "订休斯顿到伦敦的，我预算很紧"（没有直飞，逼出中转推理） |
| **Goal Expansion** | 多个子目标编织进一句话；按情境扩展隐含需求 | "找涩谷附近的寿司" → "我在涩谷，想吃本地人推荐（不是游客推荐）的寿司，顺便告诉我要不要打车" |

红线同样明确：**不许改到不可解、不许幻觉出不存在的工具/字段、不许破坏自然感**。这一步的本质是制造"必须依赖上下文和推理才能补全参数"的训练样本——正是 tool-use Agent 最需要的能力。

---

### Step 5: pass@k 求解（QuerySolver）

#### 这一步在解决什么问题

到这里任务（query + 初始状态）已经固定，现在让一个 **Assistant Agent 拿着工具去真刀真枪地解**。这一步的产出才是轨迹本身——而且因为执行的是真实 MCP 环境而非 LLM 想象，轨迹里每个 tool_response 都是环境真实返回的。

[solve_and_select()](../src/gen/query_gen/query_gen_non_conv.py#L580) 对同一任务**并发起 `pass_k=4` 个独立的 solve**（每个 k 有独立的 MCP client 和独立的 session，环境互不污染），单个 solve 的循环（[solve()](../src/gen/query_gen/query_gen_non_conv.py#L324)）：

```
灌初始状态（每 k 一份独立 client）
   ↓
循环 ≤ max_solve_iterations=15 次：
   QuerySolver(query) ──含 tool_call──> 在真实环境执行 → tool_response 回给 Agent → 继续
        │
        └──不含 tool_call──> 记为最终回答，结束
   ↓
save_scenario 导出终态 → pass_k_scenario[k]
```

**Assistant 的 prompt**（[get_assistant_instruction](../src/gen/query_gen/utils.py#L173)，模板 [Assistant_Prompt](../src/gen/query_gen/prompts.py#L14)）就是训练时 `SYSTEM_PROMPT` 的翻版——工具以 `<tools>` XML 暴露、调用格式 `<tool_call>{"name":..., "arguments":...}</tool_call>`、缺信息就问用户而不是瞎编参数。**数据合成用的 prompt 和训练/推理用的 prompt 保持一致**，避免分布错位。

**判定门槛**：一条 trace 只有实际调用过工具（长度 > 1）才会被送进裁判（[:382-383](../src/gen/query_gen/query_gen_non_conv.py#L382-L383)）——纯嘴炮不调工具的解没有训练价值。

#### 对话模式：用户模拟器与用户侧工具（QueryGenConv）

`enable_user_interaction=True` 时路由到 [QueryGenConv](../src/gen/query_gen/query_gen_conv.py)，solve 循环里 Assistant 不再只和工具打交道：每当它输出纯文本（不调工具），消息会发给一个**用户模拟器**（[User_Prompt](../src/gen/query_gen/prompts.py#L67)），模拟器以第一人称回复并回到 Assistant，直到对话结束判定器喊停。这条链上有三个精心设计的 prompt：

**① 用户模拟器**——最难的是让"用户"表现得像真人而不是金手指 NPC：

> 3. Knowledge Boundaries — Only share knowledge a real person would realistically recall
>    - ❌ Avoid: "The delivery tracking number is YT8846182814733." (unrealistic recall)
>    - ✅ Do: "I don't know my exact longitude and latitude — can you look that up?"
> 4. Don't Over-Help — If the information the assistant asking for is... should be discovered through tools or reasoning, **do not directly provide. Instead, hint the assistant how to get it.**

用户**看不到工具但是看得到环境配置**（Hidden MCP Servers 一节）——它知道世界的事实，但被要求只回忆"真人记得住的部分"，且不许替 Agent 做它本该自己查的事。没有这两条，生成的对话会是"用户报参数、Agent 填参数"的填空题。

**② 用户侧工具**（`enable_user_tool_use=True`）。先用 [User_Tools_Classification](../src/gen/query_gen/prompts.py#L273) 把涉及的工具分成两类——需要**物理动作/敏感信息/人工裁决**的（`restart_engine`、`enter_password`、`approve_surgery`）只能由人执行。这类工具从 Assistant 的工具表里**移除**，转给用户模拟器；Assistant 若试图直接调用会被环境拒绝（[:186-187](../src/gen/query_gen/query_gen_conv.py#L186-L187)）。Assistant 的 prompt 里相应多了一条 [Interaction_Rule2](../src/gen/query_gen/prompts.py#L7)：

> - You cannot execute user tools directly; instead, **guide users on how to perform these actions themselves**

这产出一类珍贵的轨迹：Agent 学会**指导用户操作**而不是越俎代庖。

**③ 结束判定**（[End_Conversation_Prompt](../src/gen/query_gen/prompts.py#L40)）。用户模拟器每轮顺带判断对话该不该结束，输出 `###STOP###` 的条件刻意覆盖了容易误判的场景：

> - IMPORTANT: When the assistant... asks if you need anything else (e.g., "Is there anything else I can help you with?"), output ###STOP###. **The core task is resolved; the follow-up is a standard conversational closing**

而"Assistant 让用户自己去操作"只是**过程指导**、不算任务完成，不能停。

---

### Step 6: 评估与选择（SolutionSelector）

#### 这一步在解决什么问题

pass@k 的 4 条解里有金有土：有的绕了弯路、有的中途放弃、有的压根没调目标工具。[SolutionSelector](../src/gen/query_gen/prompts.py#L518)（代码入口 [select()](../src/gen/query_gen/query_gen_non_conv.py#L518)）是一个独立裁判 Agent，只看对话和 4 条候选解（**不给它看环境配置**，避免被内部数据带偏），先评后选。

**第一步：逐条评估**（二值判定，`<decision>{"0": true, "2": false, ...}</decision>`），标准全部针对 LLM 解任务的典型毛病：

> - Addresses all explicit requirements... no unresolved sub-tasks remain
> - Does NOT ask the user for information retrievable via available tools（不许偷懒问用户要本该自己查的）
> - Ends with a clear summary indicating successful task completion
> - If tools fail... provides best possible answer with transparent explanation（失败了也要交代清楚）

全部通过才置 true；4 条全 false 则 `selection=None`，本轮 `decision=False`，整条链截断。

**第二步：多选一**，六级优先级从上到下：

```
1. Correctness   正确解决核心意图（一票否决级）
2. Efficiency    最少的无用/失败调用，不重复取数
3. Autonomy      最少需要用户介入（全自动优先）
4. Context Utilization  复用前轮数据，保持连贯
5. Appropriateness      工具选得准
6. Clarity       最终回答清晰直接
```

选中的那条 trace 存进 `node.steps`，对应的 `final_scenario` 存进 `node.final_scenario`；`accuracy = 通过条数 / pass_k` 也一并记录（这个值可以在数据后处理时用来做质量过滤）。

#### 轨迹清洗：filter 与 masked_arguments

`enable_filteration=True`（`RL_NON_CONV` 预设）时，选中的轨迹还要过一遍 [SolutionSelector_Filter_Prompt](../src/gen/query_gen/prompts.py#L578)：

> 1. For tool calls, keep only tool calls that directly contribute to answering the query. **Remove any redundant, exploratory, failed, invalid tool calls.**

被留下的每次调用还要标注 `masked_arguments`——**改了值也不影响语义答案的参数**：

> An argument is "maskable" if:
> - Changing its value would not alter the semantic answer to the query
> - It controls formatting, pagination, or non-critical parameters
> - For example: {"name": "search", "arguments": {"query": "...", "limit": 5}, "masked_arguments": ["limit"]}

这个字段是给 **RL 奖励**用的：RL 阶段比较 Agent 的调用和 ground_truth 时，`masked_arguments` 里的参数不参与精确匹配——否则模型会因为 `limit` 填 10 还是 5 被误罚。（这点挺细）

（另有一个更细的 [StepSelector_Prompt](../src/gen/query_gen/prompts.py#L609)，把每步标为 KEEP / HIDE / REMOVE 三档：HIDE 保留在历史里但不进训练样本，用于保住"有信息量的失败尝试"。当前 `select()` 里这条路径被注释掉了（[:556](../src/gen/query_gen/query_gen_non_conv.py#L556)），但打标结果 `type` 字段在数据后处理中仍被支持。）

---

### Step 7: 落盘

[terminate()](../src/gen/query_gen/query_gen_non_conv.py#L603) 把整条 `ToolQueryChain` 存成单个 JSON：

```
data_sft_conv/
└── 2025-01-15_10-30-00-12345-sglang.json   # {创建时间}-{seed}-{model}.json
    ├── nodes[]          # 每轮的 steps / scenario / query / decision ...
    ├── seed             # 采样种子
    ├── scenario         # 故事
    └── user_tools       # 用户侧工具清单
```

同时支持**断点续跑**：`gen()` 开头会找第一个 `decision` 不为 True 的轮（[:627-630](../src/gen/query_gen/query_gen_non_conv.py#L627-L630)），从那里继续——已成功的轮不重算。

---

### Step 8: 转成训练格式（data_process.py）

[examples/process_data.sh](../examples/process_data.sh) 调 [src/utils/data_process.py](../src/utils/data_process.py)，把目录里所有 chain JSON 转成两种格式。**两者的样本粒度完全不同**——"粒度"指一个训练样本从多长的轨迹上切下来：

| | SFT | RL |
|---|---|---|
| 一个样本 = | **一个动作步**（输入步→输出步） | **一整轮任务** |
| 样本里有什么 | 该步 instruction + **标准答案** + 累积 history | 只有任务 prompt + **判分材料**（环境快照 / ground_truth） |
| 1 条轨迹（≈4.8 轮 × 3.3 步）产出 | ~30 个样本（1,622 条对话 → 26k+ 样本） | ~4-5 个样本 |
| 训练时答案的角色 | 直接被模仿（loss 只算 output 步） | 不给模型看，rollout 完才用于算奖励 |

两种格式也对应论文 §4.1 的分工：*"Stage 1: SFT initialized with **user interaction** trajectories; Stage 2: RL training uses **only tool-call** trajectories"*——SFT 用带用户交互的对话轨迹（`SFT_CONV`），RL 只用纯工具调用轨迹（`RL_NON_CONV`）。这不只是数据偏好，见下文"RL 样本里没有用户对话"。

#### SFT 格式：每一步一对样本

[convert_to_sft_data()](../src/utils/data_process.py#L219) 把轨迹按 `(输入步, 输出步)` 切对——输入步是 `user` 或 `tool_response`，输出步是 `assistant` 或 `tool_call`，历史作为 `history` 滚动累积：

```json
{
  "instruction": "Before the demo later today, could you confirm if Alpha Tech is currently tradable...",
  "input": "",
  "output": "<think>...</think>\n\n<tool_call>\n{\"name\": \"Stocks-get_stock_info\", \"arguments\": {...}}\n</tool_call>",
  "system": "You are a helpful assistant... <tools>[本任务可用的工具 JSON]</tools>...",
  "history": [["上一轮的输入", "上一轮的输出"], ...]
}
```

（真实样本见 `data/sft_filtered/mcp_factory_sft_nips.json`，26463 条。）

三个质量闸门：

| 闸门 | 逻辑 | 位置 |
|---|---|---|
| **轮次有效性** | `decision != True` 的轮及其后全部丢弃 | [:242-243](../src/utils/data_process.py#L242-L243) |
| **失败调用** | tool_response 里含 "Fail"/"Error" 的输出步**不进训练样本**（但仍留在 history 里，模型看得见失败、但不学习失败） | [is_failed_tool_call()](../src/utils/data_process.py#L201)，[:269](../src/utils/data_process.py#L269) |
| **REMOVE 标记** | 被标 `type=REMOVE` 的步丢弃 | [:265-266](../src/utils/data_process.py#L265-L266) |

`--enable_skip` 额外要求链至少有 2 个有效轮（单轮闲聊链不进数据集）；`--enable_think` 控制是否保留求解时的 `<think>` 推理链。

#### RL 格式：每一轮一个样本

[convert_to_rl_data()](../src/utils/data_process.py#L290) 粒度是"一整轮对话"，且**不带答案轨迹**——带的是可执行环境和验收标准：

```json
{
  "prompt": "[...历史 + 本轮 query]",          // Qwen3 角色映射后的对话
  "reward_model": {
    "ground_truth": "[{\"name\": ..., \"arguments\": ..., \"masked_arguments\": [...]}, ...]",
    "style": "rule"
  },
  "extra_info": {
    "mcp_factory_kwargs": {
      "mcp_servers": "...",        // RL 时要拉起哪些 MCP server
      "initial_config": "...",     // load_scenario 灌这个
      "final_config": "..."        // save_scenario 对答案用这个（终态比对）
    }
  }
}
```

**RL 的奖励信号就藏在这里**：初始/终态快照 + 带掩码的 ground_truth 调用序列。RL 阶段把环境按 `initial_config` 拉起来让 Agent 自己 rollout，再用 `final_config` 和 `ground_truth` 做规则判分——合成阶段存下的状态快照，到了 RL 阶段就是"可验证奖励"的验证器。

具体判分采用论文 §3.5 的**复合奖励**（因为合法执行不唯一——只读调用顺序可换、`limit` 这类参数取值可变，单一参照会误罚）：

```
R = α · R_traj + (1 − α) · R_state − γ · P_length
     │              │                  │
     │              │                  └── 长度惩罚：抑制不必要的冗长调用序列
     │              └── 状态奖励：执行后 save_scenario 与 final_config 的等价性
     └── 轨迹奖励：预测调用序列与 ground_truth 的匹配（masked_arguments 中的参数不计入）

消融（论文 Figure 4）：α=0.5 最优（BFCL 多轮 41.38%）；只用状态（α=0）或只用轨迹匹配（α=1）都明显退化。
```

> **❓ RL 样本里没有用户对话，中途要反问用户怎么办？**
> 这是**构造性排除**的，不是被处理的。两层机制：① RL 轨迹合成时就用 `RL_NON_CONV`（`enable_user_interaction=False`）——非对话模式的 [solve()](../src/gen/query_gen/query_gen_non_conv.py#L324) 循环只有两条出路：调工具或给最终回答，**不存在"问用户、用户回答"的分支**，所以一条 RL 轨迹内部是干净的 `query → tool_call/response 交替 → 最终回答`；② 多轮之间的历史 user 消息被 [format_step](../src/utils/data_process.py#L176) 拍平拼进 `prompt`（[:309](../src/utils/data_process.py#L309)），是 rollout 期间不变的**静态上下文**。根本原因：RL rollout 时环境里只有 MCP server，**没有用户模拟器**——若样本里出现"Agent 中途反问"，没人回答，episode 就挂死。代价是 RL 练不到"主动向用户澄清"的能力，这部分全靠 SFT 底子（论文 Table 2：direct RL 不做 SFT 冷启动时收益更小更不稳定，与此吻合）。

---

## 🚀 完整使用示例

```bash
# Step 0: 构建工具图（一次性，产出 graph.pkl）
#   按 examples/load_tool_graph.ipynb 跑，或在 .env 配好 EMBEDDING_*/CHAT_* 后：
jupyter execute examples/load_tool_graph.ipynb   # 或逐 cell 执行

# Step 1-7: 合成轨迹（改 examples/sythesize_query.py 里的 config/N 后）
python examples/sythesize_query.py
```

[sythesize_query.py](../examples/sythesize_query.py) 的骨架就 5 步：

```python
graph = ToolGraph.load("graph.pkl")            # 1. 载入工具图
sampler = TopologySampler()                    # 2. 选采样器
config = SFT_CONV                              # 3. 选预设（见下表）
N = 200                                        # 4. 要多少条链

for seed in seeds:                             # 5. 并发跑（Semaphore(5)）
    tool_chain = graph.sample(sampler, max_nodes=15, seed=seed)
    await QueryGen(graph, config).gen(tool_chain)

# Step 8: 转训练格式
bash examples/process_data.sh
```

### 三套预设配置（[preset_config.py](../src/gen/query_gen/preset_config.py)）

| 开关 | `SFT_NON_CONV` | `SFT_CONV` | `RL_NON_CONV` |
|---|---|---|---|
| 求解模型 | sglang（本地） | sglang | deepseek |
| query 精炼 | ✗ | ✓ | ✗ |
| 用户交互 | ✗ | ✓ | ✗ |
| 用户侧工具 | ✗ | ✓ | ✗ |
| 轨迹过滤 | ✗ | ✗ | ✓（masked_arguments） |
| 输出目录 | `data_sft_non_conv/` | `data_sft_conv/` | `data_rl_non_conv/` |

直觉上：**SFT 要多样和真实**（带用户模拟、带精炼，样本直接进训练集所以不需要过滤）；**RL 要干净和可判分**（任务必须无歧义、ground_truth 必须精炼，所以开过滤、不需要对话复杂性）。

### 自定义配置

```python
from src.gen.query_gen import QueryGenConfig

config = QueryGenConfig(
    model_name="sglang",        # 求解/生成用的模型
    pass_k=4,                   # 每个任务并发解几次
    max_iterations=10,          # 状态机每轮最大循环
    max_solve_iterations=15,    # Agent 解题最大步数
    enable_split_turns=False,   # LLM 分轮 vs 概率分轮
    enable_query_refinement=False,
    enable_user_interaction=False,
    enable_user_tool_use=False,
    enable_filteration=False,
    save_folder="data/",
    log_folder="log/",
)
```

---

## 🎨 设计亮点

### 1. 可解性是构造出来的，不是筛选出来的

整条流水线把"任务可解"拆成了四道互相衔接的保证，每一道都有明确的机制兜底：

```
图上建边        →  参数"存在来源"（user_provided 或某工具的输出）
TopologySampler →  采样时逐参数检查来源（validate_parameter）
Solvability 约束 →  query 把该表达的来源表达出来（显式/指代/推理）
真实执行        →  轨迹里的每次调用都真的发生过，不存在编造的 response
```

失败不出现在"筛选"阶段（造 10000 条丢 9000 条），而是让每一条产出的链天然就是可解的。

### 2. 执行反馈闭环贯穿始终

几乎所有 LLM 环节都配了"真实验证 + 带错重试"：SchemaGenerator 的输出要过真实 `load_scenario`，建图的 LLM 输出要过工具名校验，裁判的输出要过 XML 标签断言。错误信息直接拼回 prompt 重试（`context_manager.add_prompt`），和环境生成 Phase 4 的修订循环是同一个思想。

### 3. 状态快照串起三个阶段

同一对 `load_scenario`/`save_scenario`，在三个阶段各司其职：

| 阶段 | 用途 |
|---|---|
| 合成（本文档） | 轮间状态传递（t 轮终态 = t+1 轮初态）；并发 pass@k 的环境隔离（client_id 按 k 隔离） |
| SFT | 不直接用（轨迹已固化） |
| RL | `initial_config` 重放环境，`final_config` 做终态判分 |

### 4. 多 Agent 各司其职，prompt 一一对应

一条轨迹背后是 7 个角色分工明确的 Agent，且**每个 prompt 都在堵一个具体的失败模式**：

| Agent | 堵的失败模式 |
|---|---|
| ScenarioPlanner | 故事泛化、提到工具名 |
| SchemaGenerator | 跨 server 数据打架、schema 不合规 |
| QueryGenerator | 任务不可解、query 泄漏工具名 |
| QueryRefiner | 任务太简单（参数全给） |
| QuerySolver | 编造工具结果（不可能——真环境） |
| User 模拟器 | 用户开金手指（报 UUID、抢答） |
| SolutionSelector | 选了一条"能跑但绕路"的解 |

---

## 🐛 常见问题

### 1. 轨迹链只有一轮就断了

**现象**：落盘的 JSON 里只有第 1 个 node 有 `decision=true`。

**原因**：第 2 轮 `start()` 时用第 1 轮的 `final_scenario` 做初始状态（[:248-252](../src/gen/query_gen/query_gen_non_conv.py#L248-L252)），若第 1 轮 Agent 的执行把环境改出了 schema 不合法的状态（例如删掉了必填实体），第 2 轮 `load_scenario` 失败，链被截断。

**排查**：
```bash
# 看第 1 轮终态和第 2 轮初态是否一致、能否通过 Pydantic 校验
python3 -c "
import json; d = json.load(open('data/xxx.json'))
print(json.dumps(d['nodes'][0]['final_scenario'], indent=2)[:500])
"
```

### 2. 采样出的链全是同一个 server 的工具

`TopologySampler(max_servers=3)` 达到上限后只从已有服务器选；若起点工具所在 server 工具很多，链会被它垄断。可调小 `max_servers` 的触发概率或换 `RandomWalkSampler` 对比。

### 3. SchemaGenerator 反复重试仍灌不进场景

通常是该 server 的 `Scenario_Schema` 定义过绕（嵌套 `Dict[str, Model]` 太深）——这其实是环境质量问题，建议回到环境生成阶段修 Section 1，而不是在轨迹阶段硬扛。可以在 `envs/intermediate/` 的 checkpoint 里确认该环境的验证记录。

### 4. 想增量续跑而不是从头合成

`QueryGen.gen()` 支持传 checkpoint 路径：已 `decision=true` 的轮直接跳过，从第一个失败轮继续（[:627-630](../src/gen/query_gen/query_gen_non_conv.py#L627-L630)）。

### 5. 查看某次合成的完整 LLM 交互

所有 Agent 的输入输出都按 `conversation_id` 记进了 `log/` 目录（jsonl），配合 `terminate()` 里的 `dump_log` 可以回放每一步决策。

---

## 📚 相关文档

- [environment_generation.md](environment_generation.md) — 上游：MCP 环境怎么生成（`Scenario_Schema`、`load/save_scenario` 的出处）
- [examples/load_tool_graph.ipynb](../examples/load_tool_graph.ipynb) — 工具图构建可运行示例
- [examples/sythesize_query.py](../examples/sythesize_query.py) — 轨迹合成入口示例
- [examples/process_data.sh](../examples/process_data.sh) — 数据转换脚本
- [src/gen/query_gen/prompts.py](../src/gen/query_gen/prompts.py) — 全部轨迹合成 prompt
- [README.md](../README.md) — 项目总览与 SFT/RL 训练入口

---

## 📝 总结

轨迹合成流水线把"造训练数据"组织成了一条**每一步都有真实性保证的传递链**：工具图保证任务可解，采样保证依赖完整，场景规划保证动机自然，状态填充保证环境有货，反向任务生成保证 query 与工具调用互相锁定，真实执行保证轨迹可信，pass@k + 裁判保证质量，最后状态快照把整套数据无缝交给 RL。

和环境生成对照着看，两个流水线共享同一套哲学——**LLM 负责生成，机制负责验证**——只是验证器不同：环境生成的验证器是"代码能不能跑"，轨迹合成的验证器是"任务能不能被真地解出来"。

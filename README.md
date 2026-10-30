# 企业知识问答 Agent

题号 3。项目使用 Python CLI 和 Claude Agent SDK 调用 GLM，通过“检索候选 → 读取原文 → 选择继续检索、追问或结束”的实际工具循环，为合成制度和 FAQ 提供可追溯回答。最终答复由本地业务代码从本次已读取的原文构造；模型的自由文本不直接进入答案。

文档仅为合成样例，不代表真实企业制度。检索使用小规模别名、字符相似度和预设同义词；问题支持范围由文档事实属性和有限问句词表定义。这是演示而非通用企业搜索或自然语言蕴含判断。未知属性、限定条件或未识别的子问题会返回资料缺口。

为避免把“不是 30 天”误标成“30 天”，资料入口只接受本项目四类可解析的**正向事实格式**，例如“差旅费用须在费用发生后30天内提交报销。”、“员工应至少提前3个工作日提交年假申请。”、“新员工可在入职当天通过企业服务台开通协作平台账号。”和“Insight 产品支持将报表导出为 CSV 文件。” 允许替换对应的期限、日期、格式、版本等字段，但 `passage`、`value`、`aspects`、`scope` 必须与受支持句式一致。未知句式直接报资料格式错误并要求人工核实，不把它当成可回答证据。支持的完整语法见 `src/business.py` 的 `FACT_PATTERNS`；需要扩充资料时应同时扩展格式和测试。

## 安装与测试

需要 Python 3.12 和 [uv](https://docs.astral.sh/uv/)。依赖已锁定，Claude Agent SDK 版本为 0.2.175；SDK 负责 Agent 轮次与进程内 MCP 工具调用。

```bash
uv sync --locked --python 3.12
uv run --locked python -m pytest -q
```

## 离线运行

脚本式 Mock 不需要密钥，只验证业务流程，不代表真实模型决策：

```bash
uv run --locked python -m src.main --mode mock --input examples/normal.input.json
uv run --locked python -m src.main --mode mock --input examples/conflict.input.json
uv run --locked python -m src.main --mode mock --input examples/out-of-scope.input.json
uv run --locked python -m src.main --mode mock --input examples/followup.input.json
uv run --locked python -m src.main --mode mock --input examples/rewrite.input.json
uv run --locked python -m src.main --mode mock --input examples/duration.input.json
```

检验工具失败或替换文档：

```bash
uv run --locked python -m src.main --mode mock --input examples/failure.input.json --fail-once read_excerpt
uv run --locked python -m src.main --mode mock --input examples/conflict.input.json --data examples/updated-documents.json
```

工具失败和资料不足返回 `incomplete`，CLI 退出码为 1；能给出原文回答或发现冲突并请求确认时退出码为 0。输入或文档格式错误退出码为 2。

## 真实模型运行

```bash
cp .env.example .env
# 在本地编辑 .env，填写 ANTHROPIC_AUTH_TOKEN；不要把 .env 提交到仓库。
set -a
. ./.env
set +a
uv run --locked python -m src.main --mode glm --input examples/normal.input.json
uv run --locked python -m src.main --mode glm --input examples/conflict.input.json
uv run --locked python -m src.main --mode glm --input examples/conflict.input.json --data examples/updated-documents.json
uv run --locked python -m src.main --mode glm --input examples/rewrite.input.json
uv run --locked python -m src.main --mode glm --input examples/followup.input.json
uv run --locked python -m src.main --mode glm --input examples/duration.input.json
```

默认模型 `glm-5.3-flash`，请求地址 `https://open.bigmodel.cn/api/anthropic`。真实模式不会自动退回 Mock。缺密钥、API 或 SDK 错误、未恢复的工具错误以及执行上限均会保留实际 Trace 并返回未完成；`--max-turns` 默认 12，`--max-tool-calls` 默认 16，包括最终提交决策工具。

智谱支持 [Claude Code 接入 Coding Plan](https://docs.bigmodel.cn/cn/coding-plan/tool/claude)，其 [FAQ](https://docs.bigmodel.cn/cn/coding-plan/faq) 对自建应用要求使用标准 API。SDK 嵌入业务 Agent 的套餐适用范围需要以智谱规则为准，接口调用成功不代表套餐适用已获确认。

## 输入、输出和证据边界

输入是 JSON，如 `{"question":"差旅报销期限多久？","history":[{"role":"user","content":"之前的问题"},{"role":"assistant","content":"之前的回答"}]}`。`history` 可省略，只在本轮没有明确主题且属于代词追问时用于恢复最近一次用户问题，历史助手回答不作为引用。文档在每次运行开始时作为快照载入，替换材料后需重新运行。检索只返回 ID、标题和版本日期；`read_excerpt` 才返回原文。最终决策通过 `submit_decision` 提交，但业务层仍核查问题属性、已检索文档和已读取证据，输出由原文生成。

本项目中受限语法将每个 `aspect` 映射到一种合成主题；命中后会召回该属性的资料，即使它们的 `key` 和别名不同。相同主题属性的值不同时列出原文并返回 `needs_clarification`；同值但原文不同时也保守地请求核实，不会按版本日期自行选择。文档如标有明确适用对象，问题缺少该对象时也请求确认。识别不到的改写如果读到了候选原文，且模型提出回答或追问，返回 `unverified_rewrite`，在 `unverified_candidates` 中展示候选并请用户确认相关性；它们不属于已验证 `citations`，不会直接作为答案。受限语法和同义词表无法识别任意自然语言中的所有矛盾或优先级，也不能保证真实语料的召回率。

输出含 `answer`、文档 ID、版本与原文 `citations`、`missing_information`、任务 `status` 和 `trace`。已读相关资料不足时，可以展示带“不能回答完整问题”提示的原文，但整体状态仍是 `incomplete`。Trace 的 `next_action` 是业务工具的实际执行顺序；真实模式根据 SDK 工具请求 ID、返回 ID 和模型消息 ID 另外标记 `next_action_relation`。`same_model_message` 表示同一消息发出的并列调用，不能推断模型先观察了前一调用结果；`after_tool_result` 表示下一次模型消息出现在该工具结果之后。离线 Mock 标记为 `scripted_sequence`。

## 评审入口

- [需求](docs/01-requirements.md)、[实现与取舍](docs/02-solution.md)、[实际评估与限制](docs/03-evaluation.md)
- [AI 使用](docs/04-ai-usage.md)、[复盘](docs/05-retrospective.md)
- `examples/*.input.json`、`*.glm.output.json` 与 `*.mock.output.json` 分别是输入、真实调用和离线执行输出；每份输出包含 Trace。

工具接入参考 [Claude Agent SDK 自定义工具文档](https://code.claude.com/docs/en/agent-sdk/custom-tools)，交付目录参考 [agent-campus-submission-example](https://github.com/lwz1357/agent-campus-submission-example)；后者未实现真实模型驱动的决策。

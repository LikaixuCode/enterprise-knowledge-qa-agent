# 企业知识问答 Agent

题号 3。项目使用 Python CLI 和 Claude Agent SDK 调用 GLM，通过“检索候选 → 读取原文 → 选择继续检索、追问或结束”的实际工具循环，为合成制度和 FAQ 提供可追溯回答。最终答复由本地业务代码从本次已读取的原文构造；模型的自由文本不直接进入答案。

文档仅为合成样例，不代表真实企业制度。检索使用小规模别名和字符相似度规则，问题支持范围由文档标注的事实属性和有限问句词表定义；这是演示而非通用企业搜索或自然语言蕴含判断。未知属性、限定条件或未识别的子问题会返回资料缺口。修改材料时必须同时维护 `passage`、`value`、`aspects`、`scope` 和别名；系统核对值是否按词边界出现在原文，并拒绝同单位多值或特定否定标记，但无法自动验证人工标注的全部语义是否正确。

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
```

默认模型 `glm-5.3-flash`，请求地址 `https://open.bigmodel.cn/api/anthropic`。真实模式不会自动退回 Mock。缺密钥、API 或 SDK 错误、未恢复的工具错误以及执行上限均会保留实际 Trace 并返回未完成；`--max-turns` 默认 12，`--max-tool-calls` 默认 16，包括最终提交决策工具。

智谱支持 [Claude Code 接入 Coding Plan](https://docs.bigmodel.cn/cn/coding-plan/tool/claude)，其 [FAQ](https://docs.bigmodel.cn/cn/coding-plan/faq) 对自建应用要求使用标准 API。SDK 嵌入业务 Agent 的套餐适用范围需要以智谱规则为准，接口调用成功不代表套餐适用已获确认。

## 输入、输出和证据边界

输入是 JSON，如 `{"question":"差旅报销期限多久？","history":[{"role":"user","content":"之前的问题"},{"role":"assistant","content":"之前的回答"}]}`。`history` 可省略，只在本轮没有明确主题且属于代词追问时用于恢复最近一次用户问题，历史助手回答不作为引用。文档在每次运行开始时作为快照载入，替换材料后需重新运行。检索只返回 ID、标题和版本日期；`read_excerpt` 才返回原文。最终决策通过 `submit_decision` 提交，但业务层仍核查问题属性、已检索文档和已读取证据，输出由原文生成。

同主题被人工标注为同一个 `key` 且 `value` 不同时，列出所有冲突原文并返回 `needs_clarification`；同值但原文不同时也保守地请求核实，不会按版本日期自行选择。文档如标有明确适用对象，问题缺少该对象时也请求确认。这种冲突判断依赖资料维护者正确标注主题、事实值和属性，无法识别任意自然语言中的所有矛盾或优先级。语料、可识别的主题与问句、失败模拟均是有限样例，不等于真实企业文档上的召回率或答案正确率。

输出含 `answer`、文档 ID、版本与原文 `citations`、`missing_information`、任务 `status` 和 `trace`。Trace 记录工具参数、实际结果、简短决策依据与后续动作；真实模式还记录 SDK 工具请求 ID 和对应返回。已读相关资料不足时，可以展示带“不能回答完整问题”提示的原文，但整体状态仍是 `incomplete`。

## 评审入口

- [需求](docs/01-requirements.md)、[实现与取舍](docs/02-solution.md)、[实际评估与限制](docs/03-evaluation.md)
- [AI 使用](docs/04-ai-usage.md)、[复盘](docs/05-retrospective.md)
- `examples/*.input.json`、`*.glm.output.json` 与 `*.mock.output.json` 分别是输入、真实调用和离线执行输出；每份输出包含 Trace。

工具接入参考 [Claude Agent SDK 自定义工具文档](https://code.claude.com/docs/en/agent-sdk/custom-tools)，交付目录参考 [agent-campus-submission-example](https://github.com/lwz1357/agent-campus-submission-example)；后者未实现真实模型驱动的决策。

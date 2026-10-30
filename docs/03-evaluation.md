# 测试与实际运行记录

实际开发和测试时间：2026-10-30 23:27–23:43（北京时间），约 16 分钟，为本次 Agent 并行开发的墙钟时间，包含等待模型和本地测试；不代表个人独立投入工时。

## 自动化测试

在 Python 3.12.7、Claude Agent SDK 0.2.175 环境运行 `uv run --locked python -m pytest -q`，结果为 **55 passed**，原始输出见 [test-results.txt](../examples/test-results.txt)。测试覆盖正常答案、冲突、资料外、同义改写、上下文追问、检索与读取失败、重试、误引及未读取候选、模型和工具上限、资料替换、范围和未知属性等。五条保存的真实 Trace 还通过测试逐步重放业务工具并对比本次资料、返回、答案和状态；重放不是再次访问模型。

## 示例与实际结果

| 场景 | 输入 | 实际结果与运行记录 | 说明 |
| --- | --- | --- | --- |
| 明确答案 | [normal](../examples/normal.input.json) | [真实 GLM](../examples/normal.glm.output.json)、[Mock](../examples/normal.mock.output.json) | 检索 D03，读取后引用新员工账号原文，`completed`。 |
| 制度冲突 | [conflict](../examples/conflict.input.json) | [真实 GLM](../examples/conflict.glm.output.json)、[Mock](../examples/conflict.mock.output.json) | 读取 D01、D02，模型提交 `clarify`，列出 30 天和 15 天，`needs_clarification`。 |
| 资料外 | [out-of-scope](../examples/out-of-scope.input.json) | [真实 GLM](../examples/out-of-scope.glm.output.json)、[Mock](../examples/out-of-scope.mock.output.json) | 模型检索并改写后仍无支持资料，提交 `stop`，`incomplete/out_of_scope`。 |
| 工具失败 | [failure](../examples/failure.input.json) | [真实 GLM](../examples/failure.glm.output.json)、[Mock](../examples/failure.mock.output.json) | 读取被注入失败。真实模型重新读取成功后完成；Mock 直接停止并返回 `incomplete/tool_failure`。 |
| 多子问题 | [mixed](../examples/mixed.input.json) | [真实 GLM](../examples/mixed.glm.output.json)、[Mock](../examples/mixed.mock.output.json) | 真实模型提交 `answer`，业务校验发现资料未覆盖是否需要身份证，保留相关原文并返回 `incomplete`。 |
| 相似改写 | [paraphrase](../examples/paraphrase.input.json) | [Mock](../examples/paraphrase.mock.output.json) | “带薪年休假”匹配到 D04 并读取原文。 |
| 上下文追问 | [followup](../examples/followup.input.json) | [Mock](../examples/followup.mock.output.json) | 使用上一条用户问题找出年假主题，再读取 D04；历史助手文本不是证据。 |
| 缺问题 | [missing](../examples/missing.input.json) | [Mock](../examples/missing.mock.output.json) | 不启动模型，返回 `incomplete/missing_question`。 |
| 未覆盖属性 | [unsupported](../examples/unsupported.input.json) | [Mock](../examples/unsupported.mock.output.json) | 账号开通费用无资料；只展示参考原文，`incomplete`。 |
| 替换文档 | [conflict](../examples/conflict.input.json)、[修改后的文档](../examples/updated-documents.json) | [Mock](../examples/updated.mock.output.json) | 修改 D02 版本、原文和值使其与 D01 一致；重读后 `completed`。 |

所有 GLM 输出的 `terminal.sdk_events` 保留 SDK 工具请求 ID、参数和对应返回 ID；`trace` 保留真实业务工具的结果与后续动作。Mock 以 `scripted_mock` 标记，不证明模型决策。模型工具重试和改写会因生成结果而不同，不保证每次运行的路径相同。

## 限制

- 数据和 FAQ 均是合成样例；小型别名检索和有限问句词表无法衡量真实企业语料的召回率，也不保证处理所有自然语言。未知表达、复杂条件和多义问题可能需要用户改写或人工复核。
- 同主题、事实值、适用范围和问句属性依赖人工标注。代码检测了部分数值与否定格式、原文和标注不一致及同主题原文差异，但不能证明标注语义或覆盖任意冲突。用户替换文档时必须检查所有对应字段。
- 对话历史由用户在输入中提供，不持久化会话；无法验证历史本身的真实性。助手历史只用于给模型提供对话上下文，不能用作引用。
- 智谱 SDK 接口连通不证明本业务场景符合 Coding Plan 套餐适用规则；请以[智谱 FAQ](https://docs.bigmodel.cn/cn/coding-plan/faq)为准。

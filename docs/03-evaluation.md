# 测试与实际运行记录

首次开发与测试：2026-10-30 23:27–23:43（北京时间），约 16 分钟。复审后修复与本地验证：2026-10-30 23:56 至 2026-10-31 00:09，约 13 分钟，包含等待真实模型调用；不包含后续 GitHub 提交时间。这些是多 Agent 并行工作的墙钟时间，不代表个人独立投入工时。

## 自动化测试

在 Python 3.12.7、Claude Agent SDK 0.2.175 环境运行 `uv run --locked python -m pytest -q`，结果为 **79 passed**，原始输出见 [test-results.txt](../examples/test-results.txt)。除题目验收点外，新增测试覆盖已读的同义改写、无关改写不能作为答案、开通日期与耗时的区别、否定/未知原文拒绝、不同 key 与别名的冲突、遗漏适用范围，以及同一模型消息内并列工具请求的 Trace 标记。保存的真实 Trace 也通过测试逐步重放业务工具并对比对应资料、返回、答案和状态，另外核对新版记录的 SDK 请求、返回及消息 ID；重放不是再次访问模型。

## 示例与实际结果

| 场景 | 输入 | 实际结果与运行记录 | 说明 |
| --- | --- | --- | --- |
| 明确答案 | [normal](../examples/normal.input.json) | [早期真实 GLM](../examples/normal.glm.output.json)、[当前 Mock](../examples/normal.mock.output.json) | 检索 D03，读取后引用新员工账号原文，`completed`。 |
| 制度冲突 | [conflict](../examples/conflict.input.json) | [新版真实 GLM](../examples/conflict-current.glm.output.json)、[早期真实 GLM](../examples/conflict.glm.output.json)、[当前 Mock](../examples/conflict.mock.output.json) | 读取 D01、D02，模型提交 `clarify`，列出 30 天和 15 天，`needs_clarification`。 |
| 资料外 | [out-of-scope](../examples/out-of-scope.input.json) | [早期真实 GLM](../examples/out-of-scope.glm.output.json)、[当前 Mock](../examples/out-of-scope.mock.output.json) | 模型检索并改写后仍无支持资料，提交 `stop`，`incomplete/out_of_scope`。 |
| 工具失败 | [failure](../examples/failure.input.json) | [早期真实 GLM](../examples/failure.glm.output.json)、[当前 Mock](../examples/failure.mock.output.json) | 读取被注入失败。真实模型重新读取成功后完成；Mock 直接停止并返回 `incomplete/tool_failure`。 |
| 多子问题 | [mixed](../examples/mixed.input.json) | [早期真实 GLM](../examples/mixed.glm.output.json)、[当前 Mock](../examples/mixed.mock.output.json) | 真实模型提交 `answer`，业务校验发现资料未覆盖是否需要身份证，保留相关原文并返回 `incomplete`。 |
| 相似改写 | [paraphrase](../examples/paraphrase.input.json) | [Mock](../examples/paraphrase.mock.output.json) | “带薪年休假”匹配到 D04 并读取原文。 |
| 上下文追问 | [followup](../examples/followup.input.json) | [新版真实 GLM](../examples/followup-current.glm.output.json)、[Mock](../examples/followup.mock.output.json) | 真实模型使用上一条用户问题找出年假主题，再读取 D04；历史助手文本不是证据。 |
| 缺问题 | [missing](../examples/missing.input.json) | [Mock](../examples/missing.mock.output.json) | 不启动模型，返回 `incomplete/missing_question`。 |
| 未覆盖属性 | [unsupported](../examples/unsupported.input.json) | [Mock](../examples/unsupported.mock.output.json) | 账号开通费用无资料；只展示参考原文，`incomplete`。 |
| 同义改写 | [rewrite](../examples/rewrite.input.json) | [新版真实 GLM](../examples/rewrite-current.glm.output.json)、[Mock](../examples/rewrite.mock.output.json) | 真实模型先按“报账期限多久”检索到空结果，再改写为“费用报销 单据提交 时限 规定”，读取 D01、D02 后提交 `clarify`。Mock 也执行固定同义词流程，但不证明模型能力。 |
| 未知办理耗时 | [duration](../examples/duration.input.json) | [新版真实 GLM](../examples/duration-current.glm.output.json)、[Mock](../examples/duration.mock.output.json) | 真实模型提交 `answer`，但原文仅说明何时可开通并未说明办理耗时；业务层保留原文并返回 `incomplete`。 |
| 替换文档 | [conflict](../examples/conflict.input.json)、[修改后的文档](../examples/updated-documents.json) | [新版真实 GLM](../examples/updated-current.glm.output.json)、[Mock](../examples/updated.mock.output.json) | 相同问题修改 D02 版本、原文和值使其与 D01 一致；真实模型读取两份原文后提交 `answer`，返回 `completed`。与上一行的实际 `clarify` 形成资料变化对照。 |

标为“早期”的 GLM 记录实际生成于提交 `76605633bcbcadca8a9f0b8f0d4915728563be7e`，未因本次修复而伪造或重写。其 `terminal.sdk_events` 保留 SDK 请求和返回 ID，可用于核对原模型行为；旧版 `next_action` 只表示执行顺序，不应把同一消息中的并列请求理解为模型看过前一结果。当前输出增加 `next_action_relation`。Mock 以 `scripted_mock` 标记，不证明模型决策；模型动作具有随机性。

## 限制

- 数据和 FAQ 均是合成样例；小型别名检索和有限问句词表无法衡量真实企业语料的召回率，也不保证处理所有自然语言。未知表达、复杂条件和多义问题可能需要用户改写或人工复核。
- 为保障已支持数据的原文和事实标注一致，文档入口只接受四类有限的正向事实句式；复杂或不在语法内的正确文档也会被拒绝，需要先扩展语法和测试。不能证明对任意制度文档的语义理解；用户替换文档时必须检查所有对应字段。
- 对话历史由用户在输入中提供，不持久化会话；无法验证历史本身的真实性。助手历史只用于给模型提供对话上下文，不能用作引用。
- 智谱 SDK 接口连通不证明本业务场景符合 Coding Plan 套餐适用规则；请以[智谱 FAQ](https://docs.bigmodel.cn/cn/coding-plan/faq)为准。

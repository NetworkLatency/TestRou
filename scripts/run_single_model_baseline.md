# 单模型 Baseline 评测脚本说明

`run_single_model_baseline.py` 用于评估单个模型在指定数据集上的基础能力。它不使用 SARR-CoDE 的协同分步控制策略，不进入 `PDIController`，也不会在 SLM 和 LLM 之间切换；每道题只调用一次指定模型完成生成，然后用仓库已有的答案匹配逻辑计算正确率。

## 适用场景

- 只想测一个模型在 `math500`、`aime24`、`aime25`、`gpqa` 或 `gpqa_diamond` 上的 accuracy。
- 想得到一个纯 baseline，和 SARR-CoDE 的协同策略结果做对比。
- 想复用当前仓库的数据集加载、prompt 构造、answer matching 和 summary 输出格式。

## 基本命令

评估配置文件中的 LLM：

```powershell
python scripts\run_single_model_baseline.py `
  --config configs\sarr_code_aggressive.json `
  --dataset aime25 `
  --model-role llm `
  --max-problems 10 `
  --resume
```

评估配置文件中的 SLM：

```powershell
python scripts\run_single_model_baseline.py `
  --config configs\sarr_code_aggressive.json `
  --dataset aime25 `
  --model-role slm `
  --variant qwen3_1p7b_single_baseline `
  --resume
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `--config` | 必填。SARRConfig JSON 路径，脚本会复用其中的模型路径、backend、dataset_paths、generation 和 runtime 配置。 |
| `--dataset` | 要评测的数据集。支持 `math500`、`aime24`、`aime25`、`gpqa`、`gpqa_diamond`。 |
| `--model-role` | 选择评估哪个模型配置：`slm` 或 `llm`。默认是 `llm`。 |
| `--max-problems` | 只评测前 N 道题。调试时建议先设小一点。 |
| `--max-new-tokens` | 单题最大生成 token 数。不指定时默认使用 `think_token_budget + answer_token_budget`。 |
| `--output-root` | 输出根目录。不指定时使用 config 中的 `output_dir`。 |
| `--variant` | 输出目录中的实验名。不指定时默认为 `<model-role>_single_model_baseline_v0`。 |
| `--resume` | 跳过已经完成且有 summary 记录的题目。 |
| `--stop` | 可选 stop delimiter，可传多次。例如 `--stop "</think>" --stop "\n\n\n"`。 |

## 输出文件

默认输出目录：

```text
sarr_results/<dataset>/<variant>/
```

主要产物：

| 文件 | 说明 |
| --- | --- |
| `summary.csv` | 每道题一行，包含预测答案、是否正确、耗时、token 统计、模型 backend 等字段。 |
| `summary_metrics.json` | 汇总指标，包括 accuracy、题目数、平均耗时、总 token、失败数等。 |
| `<problem_id>/<problem_id>.problem.json` | 单题摘要，包含原始题目、gold answer、预测、正确性和统计字段。 |
| `<problem_id>/<problem_id>.generation.json` | 单题生成详情，包含完整输出文本、抽取答案、trace 等。 |

## 评测口径

- 数学数据集使用现有 `benchmark_eval_match` 中的 math matcher，会优先从 `\boxed{}` 中抽取答案。
- GPQA 数据集使用 choice letter matcher，要求最终答案能被解析为 `A`、`B`、`C` 或 `D`。
- 如果数据集中没有 gold answer，则该题的 `correct` 为 `None`，不会计入 accuracy。
- 如果触发 context budget 或生成异常，该题会记录到输出文件中，但不会作为已评估正确率样本。

## 和 SARR-CoDE 脚本的区别

`run_sarr_code.py` 会同时构建 SLM 和 LLM，并通过 SARR-CoDE 控制器执行分步推理、诊断、repair、handoff 和 final answer 生成。

`run_single_model_baseline.py` 只构建 `--model-role` 指定的一个模型，并执行：

```text
load dataset -> build prompt -> single generate_text -> extract answer -> match gold -> write summary
```

因此它适合作为纯单模型能力 baseline。

## 建议流程

先跑少量样本确认模型路径、数据路径和服务端可用：

```powershell
python scripts\run_single_model_baseline.py `
  --config configs\sarr_code_aggressive.json `
  --dataset aime25 `
  --model-role llm `
  --max-problems 3
```

确认输出正常后，再去掉 `--max-problems` 跑完整数据集，并加上 `--resume` 方便断点续跑。

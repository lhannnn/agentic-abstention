# Harbor Config Templates

These configs are templates for reproducing the terminal-environment experiments. They do not contain credentials.

Path assumptions:

- remote benchmark root: `/workspace/terminalbench`
- delayed dataset root: `/workspace/terminalbench/datasets/terminalbench_delayed_abstention_10`
- immediate dataset root: `/workspace/terminalbench/datasets/terminalbench_instruction_level_abstention_267`

If your root differs, update `jobs_dir` and `datasets[0].path` before launching.

Provider assumptions:

- direct OpenAI configs use an env file containing `OPENAI_API_KEY`
- OpenRouter configs use `OPENROUTER_API_KEY` and `api_base=https://openrouter.ai/api/v1`
- Gemini CLI configs use Vertex ADC, not Gemini API keys

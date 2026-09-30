import contextlib
import hashlib
import io
import json
import runpy

driver = ('/home/xanewok/.cache/claude-code-tmp/claude-1000/-home-xanewok-repos-qwen38-eval/'
          '93663eaa-5db0-437a-ac16-7d725d755e5b/scratchpad/gpu-sampling/items/codex_drive.py')
output = io.StringIO()
with contextlib.redirect_stdout(output):
    result = runpy.run_path(driver, run_name='__main__')
row = json.loads(output.getvalue().strip().splitlines()[-1])
message = result['r']['choices'][0]['message']
identity = {
    'content': message.get('content'),
    'reasoning_content': message.get('reasoning_content'),
    'tool_calls': [call['function'] for call in message.get('tool_calls') or []],
    'finish_reason': result['r']['choices'][0].get('finish_reason'),
}
row['completion_sha256'] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
print(json.dumps(row))

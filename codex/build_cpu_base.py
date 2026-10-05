from pathlib import Path
import shlex
import subprocess
import os
root = Path.cwd()
base = root / 'build-cpu/base'
base.mkdir(exist_ok=True)
source = base / 'qwen35.cpp'
source.write_bytes(subprocess.check_output(['git', 'show', '590912a1d:src/models/qwen35.cpp']))
unit = Path('build-cpu/src/CMakeFiles/llama.dir/Unity/unity_8_cxx.cxx')
replacement = base / 'unity.cpp'
replacement.write_text(unit.read_text().replace(str(root / 'src/models/qwen35.cpp'), str(source)))
flags = {}
for line in Path('build-cpu/src/CMakeFiles/llama.dir/flags.make').read_text().splitlines():
    if ' = ' in line:
        key, value = line.split(' = ', 1)
        flags[key] = shlex.split(value)
obj = base / 'unity.o'
subprocess.run(['c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'], '-c', str(replacement), '-o', str(obj)], check=True)
link = shlex.split(Path('build-cpu/src/CMakeFiles/llama.dir/link.txt').read_text())
link[link.index('-o') + 1] = str(base / 'libllama.so.0.5.0')
link[link.index('CMakeFiles/llama.dir/Unity/unity_8_cxx.cxx.o')] = str(obj)
subprocess.run(link, cwd='build-cpu/src', check=True)
for name in ('libllama.so', 'libllama.so.0'):
    link = base / name
    if not link.is_symlink():
        link.symlink_to('libllama.so.0.5.0')
    assert link.readlink() == Path('libllama.so.0.5.0')
print('Baseline CPU library: only qwen35 graph order restored to 590912a1d; all other loaded code unchanged.')

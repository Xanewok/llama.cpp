CREATE TABLE IF NOT EXISTS test_backend_ops (
  test_time TEXT,
  build_commit TEXT,
  backend_name TEXT,
  op_name TEXT,
  op_params TEXT,
  test_mode TEXT,
  supported INTEGER,
  passed INTEGER,
  error_message TEXT,
  time_us REAL,
  flops REAL,
  bandwidth_gb_s REAL,
  memory_kb INTEGER,
  n_runs INTEGER,
  device_description TEXT,
  backend_reg_name TEXT
);

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-05T07:48:17Z', 'b43e64807', 'CUDA0', 'FLASH_ATTN_EXT', 'name=draft-fa:f16:8192,type=f32,ne=[256,12,1,1],op_params=[0:1031798784,3:10],sources=f32[256,1,12,1],f16[256,8192,2,1]nb[2,512,8388608,16777216],f16[256,8192,2,1]nb[2,512,8388608,16777216],f16[8192,1,1,1]', 'test', '1', '1', '', '0.000000', '0.000000', '0.000000', '0', '0', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-05T07:48:17Z', 'b43e64807', 'CUDA0', 'FLASH_ATTN_EXT', 'name=draft-fa:q8_0:8192,type=f32,ne=[256,12,1,1],op_params=[0:1031798784,3:10],sources=f32[256,1,12,1],q8_0[256,8192,2,1]nb[34,272,4456448,8912896],q8_0[256,8192,2,1]nb[34,272,4456448,8912896],f16[8192,1,1,1]', 'test', '1', '1', '', '0.000000', '0.000000', '0.000000', '0', '0', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-05T07:48:17Z', 'b43e64807', 'CUDA0', 'FLASH_ATTN_EXT', 'name=draft-fa:f16:32768,type=f32,ne=[256,12,1,1],op_params=[0:1031798784,3:10],sources=f32[256,1,12,1],f16[256,32768,2,1]nb[2,512,33554432,67108864],f16[256,32768,2,1]nb[2,512,33554432,67108864],f16[32768,1,1,1]', 'test', '1', '1', '', '0.000000', '0.000000', '0.000000', '0', '0', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-05T07:48:17Z', 'b43e64807', 'CUDA0', 'FLASH_ATTN_EXT', 'name=draft-fa:q8_0:32768,type=f32,ne=[256,12,1,1],op_params=[0:1031798784,3:10],sources=f32[256,1,12,1],q8_0[256,32768,2,1]nb[34,272,17825792,35651584],q8_0[256,32768,2,1]nb[34,272,17825792,35651584],f16[32768,1,1,1]', 'test', '1', '1', '', '0.000000', '0.000000', '0.000000', '0', '0', '', '');
[FLASH_ATTN_EXT] ERR = 0.002410793 > 0.002252800 INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-05T07:48:18Z', 'b43e64807', 'CUDA0', 'FLASH_ATTN_EXT', 'name=draft-fa:f16:90112,type=f32,ne=[256,12,1,1],op_params=[0:1031798784,3:10],sources=f32[256,1,12,1],f16[256,90112,2,1]nb[2,512,92274688,184549376],f16[256,90112,2,1]nb[2,512,92274688,184549376],f16[90112,1,1,1]', 'test', '1', '0', 'test failed', '0.000000', '0.000000', '0.000000', '0', '0', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-05T07:48:19Z', 'b43e64807', 'CUDA0', 'FLASH_ATTN_EXT', 'name=draft-fa:q8_0:90112,type=f32,ne=[256,12,1,1],op_params=[0:1031798784,3:10],sources=f32[256,1,12,1],q8_0[256,90112,2,1]nb[34,272,49020928,98041856],q8_0[256,90112,2,1]nb[34,272,49020928,98041856],f16[90112,1,1,1]', 'test', '1', '1', '', '0.000000', '0.000000', '0.000000', '0', '0', '', '');

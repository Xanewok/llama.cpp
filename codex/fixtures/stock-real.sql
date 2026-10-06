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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-04T22:28:41Z', '94cceb8b8', 'CUDA0', 'MUL_MAT', 'type_a=q5_K,type_b=f32,m=4096,n=1,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 'perf', '1', '1', '', '99.736894', '1177503206488.119385', '0.000000', '39496', '10224', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-04T22:28:43Z', '94cceb8b8', 'CUDA0', 'MUL_MAT', 'type_a=q6_K,type_b=f32,m=4096,n=1,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 'perf', '1', '1', '', '118.269601', '992989839042.451904', '0.000000', '47112', '8520', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-04T22:28:45Z', '94cceb8b8', 'CUDA0', 'MUL_MAT', 'type_a=q5_K,type_b=f32,m=4096,n=5,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 'perf', '1', '1', '', '117.922456', '4979565209370.416016', '0.000000', '39784', '8550', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-04T22:28:46Z', '94cceb8b8', 'CUDA0', 'MUL_MAT', 'type_a=q6_K,type_b=f32,m=4096,n=5,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 'perf', '1', '1', '', '139.045424', '4223098787735.927734', '0.000000', '47400', '7353', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-04T22:28:48Z', '94cceb8b8', 'CUDA0', 'MUL_MAT', 'type_a=q5_K,type_b=f32,m=4096,n=8,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 'perf', '1', '1', '', '111.368936', '8436141411698.887695', '0.000000', '40000', '8988', '', '');
INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-04T22:28:49Z', '94cceb8b8', 'CUDA0', 'MUL_MAT', 'type_a=q6_K,type_b=f32,m=4096,n=8,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0', 'perf', '1', '1', '', '139.923141', '6714572637892.791016', '0.000000', '47616', '7169', '', '');

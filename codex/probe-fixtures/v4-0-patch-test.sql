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

INSERT INTO test_backend_ops (test_time, build_commit, backend_name, op_name, op_params, test_mode, supported, passed, error_message, time_us, flops, bandwidth_gb_s, memory_kb, n_runs, device_description, backend_reg_name) VALUES ('2026-10-05T21:11:23Z', 'd89b1b30c', 'CUDA0', 'FLASH_ATTN_EXT', 'hsk=256,hsv=256,nh=4,nr23=[6,1],kv=4096,nb=1,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,permute=[0,2,1,3],kv_view=0,v_is_view_of_k=0,n_kv_max=0,tail=4095', 'test', '1', '1', '', '0.000000', '0.000000', '0.000000', '0', '0', '', '');
[New LWP 893403]
[New LWP 893402]
[New LWP 893401]
[New LWP 893328]
[New LWP 893327]
[New LWP 893326]
[New LWP 893309]
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
0x000072a334d10913 in __GI___wait4 (pid=893476, stat_loc=0x0, options=0, usage=0x0) at ../sysdeps/unix/sysv/linux/wait4.c:30
#0  0x000072a334d10913 in __GI___wait4 (pid=893476, stat_loc=0x0, options=0, usage=0x0) at ../sysdeps/unix/sysv/linux/wait4.c:30
30	in ../sysdeps/unix/sysv/linux/wait4.c
#1  0x000072a3352db393 in ggml_print_backtrace () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-base.so.0
#2  0x000072a3352db53b in ggml_abort () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-base.so.0
#3  0x000072a330d02d57 in ggml_cuda_error(char const*, char const*, char const*, int, char const*) () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-cuda.so.0
#4  0x000072a331b17dd1 in ggml_cuda_kernel_can_use_pdl(void const*) () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-cuda.so.0
#5  0x000072a331b1ffba in void launch_fattn<256, 1, 1>(ggml_backend_cuda_context&, ggml_tensor*, void (*)(char const*, char const*, char const*, char const*, char const*, int const*, float*, float2*, float, float, float, float, unsigned int, float, int, uint3, int, int, int, int, int, int, int, int, int, int, int, long, int, int, long, int, int, int, int, int, long), int, unsigned long, int, bool, bool, bool, bool, int) () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-cuda.so.0
#6  0x000072a331b2f22f in void ggml_cuda_flash_attn_ext_vec_case<256, (ggml_type)8, (ggml_type)8>(ggml_backend_cuda_context&, ggml_tensor*) () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-cuda.so.0
#7  0x000072a330d1c73e in ggml_backend_cuda_graph_compute(ggml_backend*, ggml_cgraph*) () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-cuda.so.0
#8  0x000072a3352f4762 in ggml_backend_graph_compute () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-base.so.0
#9  0x000072a3352fab71 in ggml_backend_compare_graph_backend () from /home/xanewok/repos/qwen38-eval-worktrees/wt/fa-kvmax-decode/codex/frozen/fa-kvmax-decode-v4/build/bin/libggml-base.so.0
#10 0x000062258f337446 in test_case::eval(ggml_backend*, ggml_backend*, char const*, printer*) ()
#11 0x000062258f2b5ec7 in test_backend(ggml_backend*, ggml_backend_device*, test_mode, char const*, char const*, printer*, char const*, int)::{lambda(ggml_backend*, ggml_backend*)#1}::operator()(ggml_backend*, ggml_backend*) const ()
#12 0x000062258f266caa in main ()
[Inferior 1 (process 893279) detached]

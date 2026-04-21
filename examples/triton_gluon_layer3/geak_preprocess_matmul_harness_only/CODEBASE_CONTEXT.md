# Codebase Context

## Repository Layout

```
GEAK/
├── docs/
│   ├── developer/
│   │   ├── contribution_guidelines.md
│   │   ├── index.md
│   │   └── mcp-tools.md
│   ├── configuration.md
│   ├── docker_env.md
│   ├── env_install.md
│   ├── index.md
│   ├── model_config.md
│   ├── quick_start.md
│   ├── subagent_guide.md
│   ├── triton_gluon_api_quick_reference.md
│   ├── triton_gluon_layer1_scope.md
│   ├── triton_gluon_layer2_scope.md
│   ├── triton_gluon_layer3_scope.md
│   ├── triton_gluon_lift_rules.md
│   ├── triton_gluon_mi3xx_baseline.md
│   ├── triton_gluon_translation_rules.md
│   └── triton_gluon_writing_guide.md
├── examples/
│   ├── knn/
│   │   ├── scripts/
│   │   │   └── task_runner.py
│   │   ├── src/
│   │   │   ├── knn.cpp
│   │   │   ├── knn_cuda.cu
│   │   │   └── knn_cuda.hip
│   │   ├── __init__.py
│   │   ├── config.yaml
│   │   ├── kernel_loader.py
│   │   ├── knn_wrapper.py
│   │   ├── new_xyz.pt
│   │   ├── test_knn.py
│   │   └── xyz.pt
│   ├── mla_decode/
│   │   ├── .gitignore
│   │   ├── config.yaml
│   │   ├── kernel.py
│   │   ├── setup.sh
│   │   └── test_kernel_harness.py
│   ├── skills/
│   │   └── silu-optimization/
│   │       └── SKILL.md
│   ├── triton_gluon_layer2/
│   │   ├── geak_preprocess_cpasync_full/
│   │   │   ├── baseline_metrics.json
│   │   │   ├── benchmark_baseline.txt
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── COMMANDMENT.md
│   │   │   ├── discovery.json
│   │   │   ├── full_benchmark_baseline.txt
│   │   │   ├── harness_path.txt
│   │   │   ├── harness_results.json
│   │   │   ├── profile.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_cpasync_harness_only/
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── discovery.json
│   │   │   ├── harness_results.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_full/
│   │   │   ├── baseline_metrics.json
│   │   │   ├── benchmark_baseline.txt
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── COMMANDMENT.md
│   │   │   ├── discovery.json
│   │   │   ├── full_benchmark_baseline.txt
│   │   │   ├── harness_path.txt
│   │   │   ├── harness_results.json
│   │   │   ├── profile.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_harness_only/
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── discovery.json
│   │   │   ├── harness_results.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_phase0/
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── discovery.json
│   │   │   └── resolved.json
│   │   ├── geak_preprocess_small_mma_full/
│   │   │   ├── baseline_metrics.json
│   │   │   ├── benchmark_baseline.txt
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── COMMANDMENT.md
│   │   │   ├── discovery.json
│   │   │   ├── full_benchmark_baseline.txt
│   │   │   ├── harness_path.txt
│   │   │   ├── harness_results.json
│   │   │   ├── profile.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_small_mma_harness_only/
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── discovery.json
│   │   │   ├── harness_results.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── 03_elementwise_add_amd.py
│   │   ├── 03_elementwise_add_cpasync_amd.py
│   │   ├── 03_elementwise_add_cpasync_nv.py
│   │   ├── 03_elementwise_add_nv.py
│   │   ├── 05_small_mma_amd.py
│   │   ├── 05_small_mma_nv.py
│   │   ├── harness_shapes_source.txt
│   │   ├── layer2_summary.md
│   │   ├── layer2_summary_cpasync.md
│   │   ├── layer2_summary_gfx942.md
│   │   ├── layer2_summary_small_mma.md
│   │   ├── README.md
│   │   ├── run_manifest.md
│   │   ├── run_manifest_cpasync.md
│   │   ├── run_manifest_small_mma.md
│   │   ├── test_elementwise_add_cpasync_layer2.py
│   │   ├── test_elementwise_add_layer2.py
│   │   └── test_small_mma_layer2.py
│   ├── triton_gluon_layer3/
│   │   ├── geak_preprocess_matmul_harness_only/
│   │   │   └── resolved.json
│   │   ├── geak_preprocess_softmax_full/
│   │   │   ├── baseline_metrics.json
│   │   │   ├── benchmark_baseline.txt
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── COMMANDMENT.md
│   │   │   ├── discovery.json
│   │   │   ├── full_benchmark_baseline.txt
│   │   │   ├── harness_path.txt
│   │   │   ├── harness_results.json
│   │   │   ├── profile.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_softmax_harness_only/
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── discovery.json
│   │   │   ├── harness_results.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_vector_add_full/
│   │   │   ├── baseline_metrics.json
│   │   │   ├── benchmark_baseline.txt
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── COMMANDMENT.md
│   │   │   ├── discovery.json
│   │   │   ├── full_benchmark_baseline.txt
│   │   │   ├── harness_path.txt
│   │   │   ├── harness_results.json
│   │   │   ├── profile.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── geak_preprocess_vector_add_harness_only/
│   │   │   ├── CODEBASE_CONTEXT.md
│   │   │   ├── discovery.json
│   │   │   ├── harness_results.json
│   │   │   ├── resolved.json
│   │   │   └── testcase_selection.json
│   │   ├── 01_vector_add_amd.py
│   │   ├── 01_vector_add_triton.py
│   │   ├── 02_fused_softmax_amd.py
│   │   ├── 02_fused_softmax_triton.py
│   │   ├── 03_matmul_amd.py    ← TARGET KERNEL
│   │   ├── 03_matmul_triton.py
│   │   ├── layer3_summary.md
│   │   ├── README.md
│   │   ├── run_manifest.md
│   │   ├── test_fused_softmax_layer3.py
│   │   ├── test_matmul_layer3.py
│   │   └── test_vector_add_layer3.py
│   ├── triton_gluon_mi3xx/
│   │   ├── README.md
│   │   └── run_manifest_template.md
│   ├── matmul_optimized.hip
│   ├── test_subagent.py
│   └── verify_config.py
├── knowledge-base/
│   ├── amd-knowledge-base/
│   │   ├── best-practices/
│   │   │   ├── ci-cd/
│   │   │   │   ... (1 items)
│   │   │   ├── debugging/
│   │   │   │   ... (1 items)
│   │   │   ├── performance/
│   │   │   │   ... (3 items)
│   │   │   └── testing/
│   │   │       ... (1 items)
│   │   ├── layer-1-hardware/
│   │   │   └── amd-gpu-arch/
│   │   │       ... (3 items)
│   │   ├── layer-2-compute-stack/
│   │   │   ├── hip/
│   │   │   │   ... (7 items)
│   │   │   ├── rocm/
│   │   │   │   ... (5 items)
│   │   │   ├── rocm-systems/
│   │   │   │   ... (2 items)
│   │   │   └── therock/
│   │   │       ... (1 items)
│   │   ├── layer-3-libraries/
│   │   │   ├── algorithms/
│   │   │   │   ... (2 items)
│   │   │   ├── blas/
│   │   │   │   ... (2 items)
│   │   │   ├── communications/
│   │   │   │   ... (1 items)
│   │   │   ├── compilers/
│   │   │   │   ... (1 items)
│   │   │   ├── fft/
│   │   │   │   ... (1 items)
│   │   │   ├── ml-primitives/
│   │   │   │   ... (1 items)
│   │   │   ├── random/
│   │   │   │   ... (1 items)
│   │   │   ├── rocm-libraries/
│   │   │   │   ... (1 items)
│   │   │   ├── solver/
│   │   │   │   ... (1 items)
│   │   │   └── sparse/
│   │   │       ... (1 items)
│   │   ├── layer-4-frameworks/
│   │   │   ├── jax/
│   │   │   │   ... (1 items)
│   │   │   ├── onnx/
│   │   │   │   ... (1 items)
│   │   │   ├── pytorch/
│   │   │   │   ... (1 items)
│   │   │   └── tensorflow/
│   │   │       ... (1 items)
│   │   ├── layer-5-llm/
│   │   │   ├── 00-quickstart/
│   │   │   │   ... (2 items)
│   │   │   ├── 01-foundations/
│   │   │   │   ... (2 items)
│   │   │   ├── 02-inference/
│   │   │   │   ... (3 items)
│   │   │   ├── 03-training/
│   │   │   │   ... (4 items)
│   │   │   ├── 04-models/
│   │   │   │   ... (4 items)
│   │   │   └── 05-advanced/
│   │   │       ... (2 items)
│   │   ├── layer-6-extended/
│   │   │   └── optimize-guides/
│   │   │       ... (7 items)
│   │   └── README.md
│   ├── comparisons/
│   │   ├── hip-cuda-programming-comparison.md
│   │   └── rocm-vs-cuda.md
│   ├── nvidia-knowledge-base/
│   │   ├── best-practices/
│   │   │   └── performance/
│   │   │       ... (3 items)
│   │   ├── layer-1-hardware/
│   │   │   └── nvidia-gpu-arch/
│   │   │       ... (3 items)
│   │   ├── layer-2-compute-stack/
│   │   │   └── cuda/
│   │   │       ... (8 items)
│   │   ├── layer-3-libraries/
│   │   │   ├── blas/
│   │   │   │   ... (1 items)
│   │   │   ├── communications/
│   │   │   │   ... (1 items)
│   │   │   └── dnn/
│   │   │       ... (1 items)
│   │   ├── layer-4-frameworks/
│   │   │   ├── jax/
│   │   │   │   ... (1 items)
│   │   │   ├── pytorch/
│   │   │   │   ... (1 items)
│   │   │   └── tensorflow/
│   │   │       ... (1 items)
│   │   ├── layer-5-llm/
│   │   │   ├── 00-quickstart/
│   │   │   │   ... (2 items)
│   │   │   ├── 01-foundations/
│   │   │   │   ... (2 items)
│   │   │   ├── 02-inference/
│   │   │   │   ... (1 items)
│   │   │   └── 03-training/
│   │   │       ... (1 items)
│   │   ├── CHANGELOG.md
│   │   ├── INDEX.md
│   │   └── README.md
│   ├── INDEX.md
│   └── README.md
├── mcp_tools/
│   ├── automated-test-discovery/
│   │   ├── src/
│   │   │   └── automated_test_discovery/
│   │   │       ... (3 items)
│   │   └── pyproject.toml
│   ├── metrix-mcp/
│   │   ├── src/
│   │   │   └── metrix_mcp/
│   │   │       ... (4 items)
│   │   └── ... (2 more)
│   └── ... (2 more)
└── ... (21 more)
```

## Kernel Dependency Tree

Target kernel: `examples/triton_gluon_layer3/03_matmul_amd.py`

No in-repo dependencies found.

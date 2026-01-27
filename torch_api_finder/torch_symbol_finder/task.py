# find_torch_symbols.py
import importlib
import pkgutil
import sys
from types import ModuleType

SYMS = [
"torch._C._cuda_attach_out_of_memory_observer",
"torch._C._cuda_beginAllocateCurrentThreadToPool",
"torch._C._cuda_clearCublasWorkspaces",
"torch._C._cuda_endAllocateToPool",
"torch._C._cuda_getDeviceCount",
"torch._C._cuda_setRNGState",
"torch._C._dynamo.eval_frame._set_lru_cache",
"torch._dynamo.functional_export.dynamo_graph_capture_for_export",
"torch._dynamo.is_exporting",
"torch._grouped_mm",
"torch._inductor.utils.is_cudagraph_unsafe_op",
"torch.accelerator.default_stream",
"torch.accelerator.Event",
"torch.accelerator.manual_seed",
"torch.accelerator.manual_seed_all",
"torch.accelerator.stream",
"torch.cuda.amp.grad_scaler._MultiDeviceReplicator",
"torch.cuda.default_generators.seed",
"torch.distributed._composable.fsdp.CPUOffloadPolicy",
"torch.distributed._composable.fsdp.fully_shard",
"torch.distributed._composable.fsdp.MixedPrecisionPolicy",
"torch.distributed._composable.fsdp.OffloadPolicy",
"torch.distributed._composable.replicate.replicate",
"torch.distributed._mesh_layout._MeshLayout",
"torch.distributed._symmetric_memory.set_backend",
"torch.distributed._tensor._utils.compute_local_shape_and_global_offset",
"torch.distributed._tools.fsdp2_mem_tracker.FSDPMemTracker",
"torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook.PostLocalSGDState",
"torch.distributed.checkpoint._consolidate_hf_safetensors.consolidate_safetensors_files_on_every_rank",
"torch.distributed.checkpoint.format_utils._EmptyStateDictLoadPlanner",
"torch.distributed.checkpoint.format_utils._load_state_dict",
"torch.distributed.checkpoint.format_utils.dcp_to_torch_save",
"torch.distributed.checkpoint.HuggingFaceStorageReader",
"torch.distributed.checkpoint.HuggingFaceStorageWriter",
"torch.distributed.checkpoint.quantized_hf_storage.QuantizedHuggingFaceStorageReader",
"torch.distributed.checkpoint.staging.DefaultStager",
"torch.distributed.checkpoint.staging.StagingOptions",
"torch.distributed.checkpoint.state_dict.get_model_state_dict",
"torch.distributed.checkpoint.state_dict.get_optimizer_state_dict",
"torch.distributed.checkpoint.state_dict.get_state_dict",
"torch.distributed.checkpoint.state_dict.set_model_state_dict",
"torch.distributed.checkpoint.state_dict.set_optimizer_state_dict",
"torch.distributed.checkpoint.state_dict.set_state_dict",
"torch.distributed.checkpoint.state_dict.StateDictOptions",
"torch.distributed.device_mesh.DeviceMesh._concatenate",
"torch.distributed.distributed_c10d._all_gather_base_coalesced",
"torch.distributed.distributed_c10d._shutdown_backend",
"torch.distributed.distributed_c10d.ProcessGroupXCCL",
"torch.distributed.fetch",
"torch.distributed.pipelining.PipelineStage",
"torch.distributed.pipelining.ScheduleGPipe",
"torch.distributed.pipelining.schedules.get_schedule_class",
"torch.distributed.ProcessGroup.Options",
"torch.distributed.ProcessGroupNCCL.Options",
"torch.distributed.rendezvous.rendezvous",
"torch.distributed.ring_exchange",
"torch.distributed.tensor.parallel.PrepareModuleInputOutput",
"torch.get_current_dtype",
"torch.multiprocessing.get_start_methods",
"torch.multiprocessing.reductions._rebuild_cuda_tensor_original",
"torch.multiprocessing.reductions._reduce_tensor_original",
"torch.nested._internal.ops.extract_kwargs",
"torch.nn.attention.flex_attention.and_masks",
"torch.nn.attention.flex_attention.BlockMask.from_kv_blocks",
"torch.nn.attention.flex_attention.create_block_mask",
"torch.nn.attention.flex_attention.flex_attention",
"torch.nn.nn.LayerNorm",
"torch.nn.utils.spectral_norm.SpectralNorm.apply",
"torch.optim.adam.Adam",
"torch.optim.adamw.AdamW",
"torch.optim.optimizer._default_to_fused_or_foreach",
"torch.optim.optimizer._disable_dynamo_if_unsupported",
"torch.optim.optimizer._get_capturable_supported_devices",
"torch.optim.optimizer._get_value",
"torch.optim.optimizer._stack_if_compiling",
"torch.optim.optimizer._view_as_real",
"torch.optim.optimizer.Optimizer",
"torch.optim.optimizer.Optimizer._group_tensors_by_device_and_dtype",
"torch.random.randint",
"torch.version.cuda.split",
"torch.zeroes",
]

def try_resolve(dotted: str):
    parts = dotted.split(".")
    # 先尝试逐段 import module
    for i in range(len(parts), 0, -1):
        modname = ".".join(parts[:i])
        attrpath = parts[i:]
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        obj = mod
        ok = True
        for a in attrpath:
            if not hasattr(obj, a):
                ok = False
                break
            obj = getattr(obj, a)
        if ok:
            return modname, attrpath
    return None

def scan_torch_packages():
    # 扫描 torch.* 子模块，建立 “名字 -> 可能模块” 的反向索引（粗略）
    import torch
    hits = {}
    for m in pkgutil.walk_packages(torch.__path__, torch.__name__ + "."):
        name = m.name
        # 只扫描常见相关包，避免太慢
        if not any(name.startswith(p) for p in [
            "torch._dynamo", "torch._inductor", "torch.distributed",
            "torch.cuda", "torch.accelerator", "torch.nn", "torch.optim",
            "torch.multiprocessing", "torch.nested"
        ]):
            continue
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        for k in getattr(mod, "__all__", []) or []:
            hits.setdefault(k, set()).add(name)
    return hits

def main():
    import torch
    print("torch.__version__ =", torch.__version__)
    print("torch.version.cuda =", torch.version.cuda)
    print()

    # 1) 直接解析
    for s in SYMS:
        r = try_resolve(s)
        if r:
            modname, attrpath = r
            if attrpath:
                print(f"[OK] {s}\n     => from {modname} import {attrpath[0]}  (then .{'.'.join(attrpath[1:])} if needed)")
            else:
                print(f"[OK] {s}\n     => import {modname}")
        else:
            print(f"[MISS] {s}")

    print("\n---- optional: __all__ reverse index scan ----")
    hits = scan_torch_packages()
    for s in SYMS:
        leaf = s.split(".")[-1]
        if leaf in hits:
            mods = sorted(hits[leaf])[:8]
            print(f"[HINT] {leaf} might be exported by: {mods}")

if __name__ == "__main__":
    main()

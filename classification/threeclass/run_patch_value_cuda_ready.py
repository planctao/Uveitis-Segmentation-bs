"""Original value study with versioned CUDA-ready checkpoint restoration."""
import register_patch_value_experiments as engine
from patch_adapter_cuda_ready import cuda_ready_restore

engine.restore = cuda_ready_restore(engine.restore)

if __name__ == '__main__': engine.main()

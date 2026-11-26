"""Original joint confirmation with CUDA conversion before inference mode."""
import register_patch_joint_confirmation as engine
from patch_adapter_cuda_ready import cuda_ready_restore

engine.restore_query = cuda_ready_restore(engine.restore_query)

if __name__ == '__main__': engine.main()

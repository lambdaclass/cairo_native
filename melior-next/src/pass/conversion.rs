//! Dialect conversion passes.

use super::Pass;
use crate::mlir_sys::{
    mlirCreateConversionArithToLLVMConversionPass, mlirCreateConversionConvertAMDGPUToROCDL,
    mlirCreateConversionConvertAffineForToGPU, mlirCreateConversionConvertAffineToStandard,
    mlirCreateConversionConvertArithToSPIRV, mlirCreateConversionConvertArmNeon2dToIntr,
    mlirCreateConversionConvertAsyncToLLVM, mlirCreateConversionConvertBufferizationToMemRef,
    mlirCreateConversionConvertComplexToLLVM, mlirCreateConversionConvertComplexToLibm,
    mlirCreateConversionConvertComplexToStandard, mlirCreateConversionConvertControlFlowToLLVM,
    mlirCreateConversionConvertControlFlowToSPIRV, mlirCreateConversionConvertFuncToLLVM,
    mlirCreateConversionConvertFuncToSPIRV, mlirCreateConversionConvertGPUToSPIRV,
    mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc,
    mlirCreateConversionConvertGpuOpsToNVVMOps, mlirCreateConversionConvertGpuOpsToROCDLOps,
    mlirCreateConversionConvertIndexToLLVMPass, mlirCreateConversionConvertLinalgToLLVM,
    mlirCreateConversionConvertLinalgToStandard, mlirCreateConversionConvertMathToFuncs,
    mlirCreateConversionConvertMathToLLVM, mlirCreateConversionConvertMathToLibm,
    mlirCreateConversionConvertMathToSPIRV, mlirCreateConversionConvertMemRefToSPIRV,
    mlirCreateConversionConvertNVGPUToNVVM, mlirCreateConversionConvertOpenACCToLLVM,
    mlirCreateConversionConvertOpenACCToSCF, mlirCreateConversionConvertOpenMPToLLVM,
    mlirCreateConversionConvertPDLToPDLInterp, mlirCreateConversionConvertParallelLoopToGpu,
    mlirCreateConversionConvertSCFToOpenMP, mlirCreateConversionConvertSPIRVToLLVM,
    mlirCreateConversionConvertShapeConstraints, mlirCreateConversionConvertShapeToStandard,
    mlirCreateConversionConvertTensorToLinalg, mlirCreateConversionConvertTensorToSPIRV,
    mlirCreateConversionConvertVectorToGPU, mlirCreateConversionConvertVectorToLLVM,
    mlirCreateConversionConvertVectorToSCF, mlirCreateConversionConvertVectorToSPIRV,
    mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls,
    mlirCreateConversionGpuToLLVMConversionPass, mlirCreateConversionLowerHostCodeToLLVM,
    mlirCreateConversionMapMemRefStorageClass, mlirCreateConversionMemRefToLLVMConversionPass,
    mlirCreateConversionReconcileUnrealizedCasts, mlirCreateConversionSCFToControlFlow,
    mlirCreateConversionSCFToSPIRV, mlirCreateConversionTosaToArith,
    mlirCreateConversionTosaToLinalg, mlirCreateConversionTosaToLinalgNamed,
    mlirCreateConversionTosaToSCF, mlirCreateConversionTosaToTensor,
};

/// Creates a pass to convert the `arith` dialect to the `llvm` dialect.
pub fn convert_arithmetic_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionArithToLLVMConversionPass)
}

/// Creates a pass to convert the `cf` dialect to the `llvm` dialect.
pub fn convert_cf_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToLLVM)
}

/// Creates a pass to convert the `scf` dialect to the `cf` dialect.
pub fn convert_scf_to_cf() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionSCFToControlFlow)
}

/// Creates a pass to convert the `func` dialect to the `llvm` dialect.
pub fn convert_func_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertFuncToLLVM)
}

/// Creates a pass to convert the `math` dialect to the `llvm` dialect.
pub fn convert_math_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToLLVM)
}

/// Creates a pass to convert the builtin index to the `llvm` dialect.
pub fn convert_index_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertIndexToLLVMPass)
}

/// Creates a pass to convert the `cf` dialect to the `spirv` dialect.
pub fn convert_cf_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToSPIRV)
}

/// Creates a pass to convert the `math` dialect to the `spirv` dialect.
pub fn convert_math_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToSPIRV)
}

/// Creates a pass to convert the `math` dialect to the `libm` dialect.
pub fn convert_math_to_libm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToLibm)
}

/// Creates a pass to convert the `affine for` dialect to the `gpu` dialect.
pub fn convert_affine_for_to_gpu() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAffineForToGPU)
}

/// Creates a pass to convert the `affine` dialect to the `standard` dialect.
pub fn convert_affine_to_standard() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAffineToStandard)
}

/// Creates a pass to convert the `async` dialect to the `llvm` dialect.
pub fn convert_async_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAsyncToLLVM)
}

/// Creates a pass to convert the `gpu` dialect to the `llvm` dialect.
pub fn convert_gpu_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionGpuToLLVMConversionPass)
}

/// Creates a pass to convert the `affiner for` dialect to the `gpu` dialect.
pub fn convert_affiner_for_to_gpu() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAffineForToGPU)
}

/// Creates a pass to reconcile builtin.unrealized_conversion_cast
pub fn convert_reconcile_casts() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionReconcileUnrealizedCasts)
}

pub fn convert_arith_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionArithToLLVMConversionPass)
}
pub fn convert_amdgpu_to_rocdl() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertAMDGPUToROCDL)
}
pub fn convert_arith_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertArithToSPIRV)
}
pub fn convert_arm_neon2d_to_intr() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertArmNeon2dToIntr)
}
pub fn convert_bufferization_to_mem_ref() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertBufferizationToMemRef)
}
pub fn convert_complex_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertComplexToLLVM)
}
pub fn convert_complex_to_libm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertComplexToLibm)
}
pub fn convert_complex_to_standard() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertComplexToStandard)
}
pub fn convert_control_flow_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToLLVM)
}
pub fn convert_control_flow_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertControlFlowToSPIRV)
}
pub fn convert_func_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertFuncToSPIRV)
}
pub fn convert_gpu_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertGPUToSPIRV)
}
pub fn convert_gpu_launch_func_to_vulkan_launch_func() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertGpuLaunchFuncToVulkanLaunchFunc)
}
pub fn convert_gpu_ops_to_nvvmops() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertGpuOpsToNVVMOps)
}
pub fn convert_gpu_ops_to_rocdlops() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertGpuOpsToROCDLOps)
}
pub fn convert_index_to_llvmpass() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertIndexToLLVMPass)
}
pub fn convert_linalg_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertLinalgToLLVM)
}
pub fn convert_linalg_to_standard() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertLinalgToStandard)
}
pub fn convert_math_to_funcs() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMathToFuncs)
}
pub fn convert_mem_ref_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertMemRefToSPIRV)
}
pub fn convert_nvgpu_to_nvvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertNVGPUToNVVM)
}
pub fn convert_open_acc_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertOpenACCToLLVM)
}
pub fn convert_open_acc_to_scf() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertOpenACCToSCF)
}
pub fn convert_openmp_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertOpenMPToLLVM)
}
pub fn convert_pdl_to_pdlinterp() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertPDLToPDLInterp)
}
pub fn convert_parallel_loop_to_gpu() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertParallelLoopToGpu)
}
pub fn convert_scf_to_openmp() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertSCFToOpenMP)
}
pub fn convert_spirv_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertSPIRVToLLVM)
}
pub fn convert_shape_constraints() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertShapeConstraints)
}
pub fn convert_shape_to_standard() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertShapeToStandard)
}
pub fn convert_tensor_to_linalg() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertTensorToLinalg)
}
pub fn convert_tensor_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertTensorToSPIRV)
}
pub fn convert_vector_to_gpu() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertVectorToGPU)
}
pub fn convert_vector_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertVectorToLLVM)
}
pub fn convert_vector_to_scf() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertVectorToSCF)
}
pub fn convert_vector_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertVectorToSPIRV)
}
pub fn convert_vulkan_launch_func_to_vulkan_calls() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionConvertVulkanLaunchFuncToVulkanCalls)
}
pub fn convert_gpu_to_llvmconversion_pass() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionGpuToLLVMConversionPass)
}
pub fn convert_lower_host_code_to_llvm() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionLowerHostCodeToLLVM)
}
pub fn convert_map_memref_storage_class() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionMapMemRefStorageClass)
}
pub fn convert_memref_to_llvmconversion_pass() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionMemRefToLLVMConversionPass)
}
pub fn convert_reconcile_unrealized_casts() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionReconcileUnrealizedCasts)
}
pub fn convert_scf_to_spirv() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionSCFToSPIRV)
}
pub fn convert_tosa_to_arith() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionTosaToArith)
}
pub fn convert_tosa_to_linalg() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionTosaToLinalg)
}
pub fn convert_tosa_to_linalg_named() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionTosaToLinalgNamed)
}
pub fn convert_tosa_to_scf() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionTosaToSCF)
}
pub fn convert_tosa_to_tensor() -> Pass {
    Pass::from_raw_fn(mlirCreateConversionTosaToTensor)
}

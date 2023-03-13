module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @kernel1(%arg0: i256, %arg1: i256) kernel {
      %0 = arith.addi %arg0, %arg1 : i256
      %1 = arith.trunci %0 : i256 to i32
      gpu.printf "suma: %d " %1 : i32
      gpu.return
    }
  }
  func.func @main() -> i32 attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c4_i256 = arith.constant 4 : i256
    %c2_i256 = arith.constant 2 : i256
    gpu.launch_func  @kernels::@kernel1 blocks in (%c1, %c1, %c1) threads in (%c1, %c1, %c1) dynamic_shared_memory_size %c0_i32 args(%c4_i256 : i256, %c2_i256 : i256)
    %c0_i32_0 = arith.constant 0 : i32
    return %c0_i32_0 : i32
  }
}

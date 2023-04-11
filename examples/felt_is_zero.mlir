module attributes {llvm.data_layout = ""} {
  llvm.func @malloc(i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @felt252_mul(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i256 to i512
    %1 = llvm.zext %arg1 : i256 to i512
    %2 = llvm.mul %0, %1  : i512
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %4 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %5 = llvm.srem %2, %4  : i512
    %6 = llvm.trunc %5 : i512 to i256
    llvm.return %6 : i256
  }
  llvm.func @"felt_is_zero::felt_is_zero::mul_if_not_zero"(%arg0: i256) -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1(%arg0 : i256)
  ^bb1(%0: i256):  // pred: ^bb0
    %1 = llvm.mlir.constant(0 : i256) : i256
    %2 = llvm.icmp "eq" %0, %1 : i256
    llvm.cond_br %2, ^bb2, ^bb3(%0 : i256)
  ^bb2:  // pred: ^bb1
    %3 = llvm.mlir.constant(0 : i256) : i256
    llvm.br ^bb4(%3 : i256)
  ^bb3(%4: i256):  // pred: ^bb1
    %5 = llvm.mlir.constant(2 : i256) : i256
    %6 = llvm.call @felt252_mul(%4, %5) : (i256, i256) -> i256
    llvm.br ^bb4(%6 : i256)
  ^bb4(%7: i256):  // 2 preds: ^bb2, ^bb3
    llvm.return %7 : i256
  }
  llvm.func @"_mlir_ciface_felt_is_zero::felt_is_zero::mul_if_not_zero"(%arg0: i256) -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"felt_is_zero::felt_is_zero::mul_if_not_zero"(%arg0) : (i256) -> i256
    llvm.return %0 : i256
  }
}

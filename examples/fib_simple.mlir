module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @felt252_add(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.add %arg0, %arg1  : i256
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.icmp "uge" %0, %1 : i256
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %0 : i256
  ^bb2:  // pred: ^bb0
    %3 = llvm.sub %0, %1  : i256
    llvm.return %3 : i256
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.sub %arg0, %arg1  : i256
    %1 = llvm.mlir.constant(0 : i256) : i256
    %2 = llvm.icmp "slt" %0, %1 : i256
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %0 : i256
  ^bb2:  // pred: ^bb0
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %4 = llvm.add %0, %3  : i256
    llvm.return %4 : i256
  }
  llvm.func @"fib_simple::fib_simple::fib"(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1(%arg0, %arg1, %arg2 : i256, i256, i256)
  ^bb1(%0: i256, %1: i256, %2: i256):  // pred: ^bb0
    %3 = llvm.mlir.constant(0 : i256) : i256
    %4 = llvm.icmp "eq" %2, %3 : i256
    llvm.cond_br %4, ^bb2(%0 : i256), ^bb3(%0, %1, %2 : i256, i256, i256)
  ^bb2(%5: i256):  // pred: ^bb1
    llvm.br ^bb4(%5 : i256)
  ^bb3(%6: i256, %7: i256, %8: i256):  // pred: ^bb1
    %9 = llvm.call @felt252_add(%6, %7) : (i256, i256) -> i256
    %10 = llvm.mlir.constant(1 : i256) : i256
    %11 = llvm.call @felt252_sub(%8, %10) : (i256, i256) -> i256
    %12 = llvm.call @"fib_simple::fib_simple::fib"(%7, %9, %11) : (i256, i256, i256) -> i256
    llvm.br ^bb4(%12 : i256)
  ^bb4(%13: i256):  // 2 preds: ^bb2, ^bb3
    llvm.return %13 : i256
  }
  llvm.func @"_mlir_ciface_fib_simple::fib_simple::fib"(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"fib_simple::fib_simple::fib"(%arg0, %arg1, %arg2) : (i256, i256, i256) -> i256
    llvm.return %0 : i256
  }
}

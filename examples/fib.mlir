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
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func @"fib::fib::fib"(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
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
    %12 = llvm.call @"fib::fib::fib"(%7, %9, %11) : (i256, i256, i256) -> i256
    llvm.br ^bb4(%12 : i256)
  ^bb4(%13: i256):  // 2 preds: ^bb2, ^bb3
    llvm.return %13 : i256
  }
  llvm.func @"_mlir_ciface_fib::fib::fib"(%arg0: i256, %arg1: i256, %arg2: i256) -> i256 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"fib::fib::fib"(%arg0, %arg1, %arg2) : (i256, i256, i256) -> i256
    llvm.return %0 : i256
  }
  llvm.func @"fib::fib::fib_mid"(%arg0: i256) -> !llvm.struct<()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1(%arg0 : i256)
  ^bb1(%0: i256):  // pred: ^bb0
    %1 = llvm.mlir.constant(0 : i256) : i256
    %2 = llvm.icmp "eq" %0, %1 : i256
    llvm.cond_br %2, ^bb2, ^bb3(%0 : i256)
  ^bb2:  // pred: ^bb1
    llvm.br ^bb4
  ^bb3(%3: i256):  // pred: ^bb1
    %4 = llvm.mlir.constant(0 : i256) : i256
    %5 = llvm.mlir.constant(1 : i256) : i256
    %6 = llvm.mlir.constant(500 : i256) : i256
    %7 = llvm.call @"fib::fib::fib"(%4, %5, %6) : (i256, i256, i256) -> i256
    %8 = llvm.mlir.constant(1 : i256) : i256
    %9 = llvm.call @felt252_sub(%3, %8) : (i256, i256) -> i256
    %10 = llvm.call @"fib::fib::fib_mid"(%9) : (i256) -> !llvm.struct<()>
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %11 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %11 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_fib::fib::fib_mid"(%arg0: !llvm.ptr<struct<()>>, %arg1: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"fib::fib::fib_mid"(%arg1) : (i256) -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
  llvm.func @"fib::fib::main"(%arg0: i256) -> !llvm.struct<()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(100 : i256) : i256
    %1 = llvm.call @"fib::fib::fib_mid"(%0) : (i256) -> !llvm.struct<()>
    %2 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %2 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_fib::fib::main"(%arg0: !llvm.ptr<struct<()>>, %arg1: i256) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"fib::fib::main"(%arg1) : (i256) -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}

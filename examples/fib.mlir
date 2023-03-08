module attributes {llvm.data_layout = ""} {
  llvm.func @fib(%arg0: i256, %arg1: i256, %arg2: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.icmp "eq" %arg2, %0 : i256
    %2 = llvm.mlir.constant(1 : i256) : i256
    llvm.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb3(%arg0, %0 : i256, i256)
  ^bb2:  // pred: ^bb0
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %4 = llvm.add %arg0, %arg1  : i256
    %5 = llvm.srem %4, %3  : i256
    %6 = llvm.sub %arg2, %2  : i256
    %7 = llvm.srem %6, %3  : i256
    %8 = llvm.call @fib(%arg1, %5, %7) : (i256, i256, i256) -> !llvm.struct<(i256, i256)>
    %9 = llvm.extractvalue %8[0] : !llvm.struct<(i256, i256)> 
    %10 = llvm.extractvalue %8[1] : !llvm.struct<(i256, i256)> 
    %11 = llvm.add %10, %2  : i256
    %12 = llvm.srem %11, %3  : i256
    llvm.br ^bb3(%9, %12 : i256, i256)
  ^bb3(%13: i256, %14: i256):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %15 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %16 = llvm.insertvalue %13, %15[0] : !llvm.struct<(i256, i256)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(i256, i256)> 
    llvm.return %17 : !llvm.struct<(i256, i256)>
  }
  llvm.func @_mlir_ciface_fib(%arg0: !llvm.ptr<struct<(i256, i256)>>, %arg1: i256, %arg2: i256, %arg3: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @fib(%arg1, %arg2, %arg3) : (i256, i256, i256) -> !llvm.struct<(i256, i256)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i256, i256)>>
    llvm.return
  }
  llvm.func @fib_mid(%arg0: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.icmp "eq" %arg0, %0 : i256
    %2 = llvm.mlir.constant(1 : i256) : i256
    %3 = llvm.mlir.constant(500 : i256) : i256
    llvm.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    %4 = llvm.call @fib(%0, %2, %2) : (i256, i256, i256) -> !llvm.struct<(i256, i256)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i256, i256)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i256, i256)> 
    %7 = llvm.sub %arg0, %2  : i256
    %8 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %9 = llvm.srem %7, %8  : i256
    llvm.call @fib_mid(%9) : (i256) -> ()
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    llvm.return
  }
  llvm.func @_mlir_ciface_fib_mid(%arg0: i256) attributes {llvm.emit_c_interface} {
    llvm.call @fib_mid(%arg0) : (i256) -> ()
    llvm.return
  }
  llvm.func @main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(100 : i256) : i256
    llvm.call @fib_mid(%0) : (i256) -> ()
    %1 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %1 : i32
  }
  llvm.func @_mlir_ciface_main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @main() : () -> i32
    llvm.return %0 : i32
  }
}

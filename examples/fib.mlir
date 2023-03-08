module attributes {llvm.data_layout = ""} {
  llvm.func @fib(%arg0: i256, %arg1: i256, %arg2: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.icmp "eq" %arg2, %0 : i256
    %2 = llvm.mlir.constant(1 : i256) : i256
    llvm.cond_br %1, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb3(%arg0, %0 : i256, i256)
  ^bb2:  // pred: ^bb0
    %3 = llvm.add %arg0, %arg1  : i256
    %4 = llvm.sub %arg2, %2  : i256
    %5 = llvm.call @fib(%arg1, %3, %4) : (i256, i256, i256) -> !llvm.struct<(i256, i256)>
    %6 = llvm.extractvalue %5[0] : !llvm.struct<(i256, i256)>
    %7 = llvm.extractvalue %5[1] : !llvm.struct<(i256, i256)>
    %8 = llvm.add %7, %2  : i256
    llvm.br ^bb3(%6, %8 : i256, i256)
  ^bb3(%9: i256, %10: i256):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb4
  ^bb4:  // pred: ^bb3
    %11 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %12 = llvm.insertvalue %9, %11[0] : !llvm.struct<(i256, i256)>
    %13 = llvm.insertvalue %10, %12[1] : !llvm.struct<(i256, i256)>
    llvm.return %13 : !llvm.struct<(i256, i256)>
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
    llvm.call @fib_mid(%7) : (i256) -> ()
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    llvm.return
  }
  llvm.func @_mlir_ciface_fib_mid(%arg0: i256) attributes {llvm.emit_c_interface} {
    llvm.call @fib_mid(%arg0) : (i256) -> ()
    llvm.return
  }
  llvm.func @main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(500 : i256) : i256
    llvm.call @fib_mid(%0) : (i256) -> ()
    %1 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %1 : i32
  }
  llvm.func @_mlir_ciface_main() -> i32 attributes {llvm.emit_c_interface} {
    %0 = llvm.call @main() : () -> i32
    llvm.return %0 : i32
  }
}

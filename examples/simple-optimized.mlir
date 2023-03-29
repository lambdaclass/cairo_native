module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"dup<felt252>"(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @felt252_add(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %1 = llvm.add %arg0, %arg1  : i256
    %2 = llvm.icmp "uge" %1, %0 : i256
    llvm.cond_br %2, ^bb2, ^bb1(%1 : i256)
  ^bb1(%3: i256):  // 2 preds: ^bb0, ^bb2
    llvm.return %3 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.sub %1, %0  : i256
    llvm.br ^bb1(%4 : i256)
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %1 = llvm.sub %arg0, %arg1  : i256
    %2 = llvm.icmp "ult" %arg0, %arg1 : i256
    llvm.cond_br %2, ^bb2, ^bb1(%1 : i256)
  ^bb1(%3: i256):  // 2 preds: ^bb0, ^bb2
    llvm.return %3 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.sub %1, %0  : i256
    llvm.br ^bb1(%4 : i256)
  }
  llvm.func internal @"struct_construct<Tuple<felt252, felt252>>"(%arg0: i256, %arg1: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @"store_temp<Tuple<felt252, felt252>>"(%arg0: !llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)> {
    llvm.return %arg0 : !llvm.struct<(i256, i256)>
  }
  llvm.func @simple_simple_something(%arg0: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %1 = llvm.mlir.constant(2 : i256) : i256
    %2 = llvm.mlir.constant(1 : i256) : i256
    %3 = llvm.mlir.constant(3 : i256) : i256
    %4 = llvm.add %arg0, %1  : i256
    %5 = llvm.icmp "uge" %4, %0 : i256
    llvm.cond_br %5, ^bb1, ^bb2(%4 : i256)
  ^bb1:  // pred: ^bb0
    %6 = llvm.add %arg0, %2  : i256
    llvm.br ^bb2(%6 : i256)
  ^bb2(%7: i256):  // 2 preds: ^bb0, ^bb1
    %8 = llvm.sub %arg0, %1  : i256
    %9 = llvm.icmp "ult" %arg0, %1 : i256
    llvm.cond_br %9, ^bb3, ^bb4(%8 : i256)
  ^bb3:  // pred: ^bb2
    %10 = llvm.sub %arg0, %3  : i256
    llvm.br ^bb4(%10 : i256)
  ^bb4(%11: i256):  // 2 preds: ^bb2, ^bb3
    %12 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %13 = llvm.insertvalue %7, %12[0] : !llvm.struct<(i256, i256)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(i256, i256)> 
    llvm.return %14 : !llvm.struct<(i256, i256)>
  }
  llvm.func @_mlir_ciface_simple_simple_something(%arg0: !llvm.ptr<struct<(i256, i256)>>, %arg1: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_simple_something(%arg1) : (i256) -> !llvm.struct<(i256, i256)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i256, i256)>>
    llvm.return
  }
}

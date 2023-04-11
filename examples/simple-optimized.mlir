module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @felt252_add(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %1 = llvm.add %arg0, %arg1  : i256
    %2 = llvm.icmp "uge" %1, %0 : i256
    llvm.cond_br %2, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %1 : i256
  ^bb2:  // pred: ^bb0
    %3 = llvm.sub %1, %0  : i256
    llvm.return %3 : i256
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %1 = llvm.mlir.constant(0 : i256) : i256
    %2 = llvm.sub %arg0, %arg1  : i256
    %3 = llvm.icmp "slt" %2, %1 : i256
    llvm.cond_br %3, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.return %2 : i256
  ^bb2:  // pred: ^bb0
    %4 = llvm.add %2, %0  : i256
    llvm.return %4 : i256
  }
  llvm.func internal @"struct_construct<Tuple<felt252, felt252>>"(%arg0: i256, %arg1: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func @"simple::simple::something"(%arg0: i256) -> !llvm.struct<(i256, i256)> attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.mlir.constant(2 : i256) : i256
    llvm.br ^bb1(%arg0 : i256)
  ^bb1(%3: i256):  // pred: ^bb0
    %4 = llvm.add %3, %2  : i256
    %5 = llvm.icmp "uge" %4, %1 : i256
    llvm.cond_br %5, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb4(%4 : i256)
  ^bb3:  // pred: ^bb1
    %6 = llvm.sub %4, %1  : i256
    llvm.br ^bb4(%6 : i256)
  ^bb4(%7: i256):  // 2 preds: ^bb2, ^bb3
    %8 = llvm.sub %3, %2  : i256
    %9 = llvm.icmp "slt" %8, %0 : i256
    llvm.cond_br %9, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    llvm.br ^bb7(%8 : i256)
  ^bb6:  // pred: ^bb4
    %10 = llvm.add %8, %1  : i256
    llvm.br ^bb7(%10 : i256)
  ^bb7(%11: i256):  // 2 preds: ^bb5, ^bb6
    %12 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %13 = llvm.insertvalue %7, %12[0] : !llvm.struct<(i256, i256)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(i256, i256)> 
    llvm.return %14 : !llvm.struct<(i256, i256)>
  }
  llvm.func @"_mlir_ciface_simple::simple::something"(%arg0: !llvm.ptr<struct<(i256, i256)>>, %arg1: i256) attributes {llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256) : i256
    %2 = llvm.mlir.constant(2 : i256) : i256
    llvm.br ^bb1(%arg1 : i256)
  ^bb1(%3: i256):  // pred: ^bb0
    %4 = llvm.add %3, %2  : i256
    %5 = llvm.icmp "uge" %4, %1 : i256
    llvm.cond_br %5, ^bb3, ^bb2
  ^bb2:  // pred: ^bb1
    llvm.br ^bb4(%4 : i256)
  ^bb3:  // pred: ^bb1
    %6 = llvm.sub %4, %1  : i256
    llvm.br ^bb4(%6 : i256)
  ^bb4(%7: i256):  // 2 preds: ^bb2, ^bb3
    %8 = llvm.sub %3, %2  : i256
    %9 = llvm.icmp "slt" %8, %0 : i256
    llvm.cond_br %9, ^bb6, ^bb5
  ^bb5:  // pred: ^bb4
    llvm.br ^bb7(%8 : i256)
  ^bb6:  // pred: ^bb4
    %10 = llvm.add %8, %1  : i256
    llvm.br ^bb7(%10 : i256)
  ^bb7(%11: i256):  // 2 preds: ^bb5, ^bb6
    %12 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %13 = llvm.insertvalue %7, %12[0] : !llvm.struct<(i256, i256)> 
    %14 = llvm.insertvalue %11, %13[1] : !llvm.struct<(i256, i256)> 
    llvm.br ^bb8(%14 : !llvm.struct<(i256, i256)>)
  ^bb8(%15: !llvm.struct<(i256, i256)>):  // pred: ^bb7
    llvm.store %15, %arg0 : !llvm.ptr<struct<(i256, i256)>>
    llvm.return
  }
}

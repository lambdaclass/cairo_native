module attributes {llvm.data_layout = ""} {
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
  llvm.func internal @simple_simple_something(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.constant(2 : i256) : i256
    %1 = llvm.call @"dup<felt252>"(%arg0) : (i256) -> !llvm.struct<(i256, i256)>
    %2 = llvm.extractvalue %1[0] : !llvm.struct<(i256, i256)> 
    %3 = llvm.extractvalue %1[1] : !llvm.struct<(i256, i256)> 
    %4 = llvm.call @felt252_add(%3, %0) : (i256, i256) -> i256
    %5 = llvm.call @felt252_sub(%2, %0) : (i256, i256) -> i256
    %6 = llvm.call @"struct_construct<Tuple<felt252, felt252>>"(%4, %5) : (i256, i256) -> !llvm.struct<(i256, i256)>
    %7 = llvm.call @"store_temp<Tuple<felt252, felt252>>"(%6) : (!llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)>
    llvm.return %7 : !llvm.struct<(i256, i256)>
  }
}

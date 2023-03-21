module attributes {llvm.data_layout = ""} {
  llvm.func internal @"dup<felt252>"(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg0, %1[1] : !llvm.struct<(i256, i256)> 
    llvm.return %2 : !llvm.struct<(i256, i256)>
  }
  llvm.func internal @felt252_add(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.add %1, %1  : i512
    %3 = llvm.srem %2, %0  : i512
    %4 = llvm.trunc %3 : i512 to i256
    llvm.return %4 : i256
  }
  llvm.func internal @felt252_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.sub %1, %1  : i512
    %3 = llvm.srem %2, %0  : i512
    %4 = llvm.trunc %3 : i512 to i256
    llvm.return %4 : i256
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
  llvm.func @simple_simple_something(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.add %1, %1  : i512
    %3 = llvm.srem %2, %0  : i512
    %4 = llvm.trunc %3 : i512 to i256
    %5 = llvm.sub %1, %1  : i512
    %6 = llvm.srem %5, %0  : i512
    %7 = llvm.trunc %6 : i512 to i256
    %8 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %9 = llvm.insertvalue %4, %8[0] : !llvm.struct<(i256, i256)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(i256, i256)> 
    llvm.return %10 : !llvm.struct<(i256, i256)>
  }
}

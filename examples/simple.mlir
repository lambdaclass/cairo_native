module {
  func.func @"dup<felt>"(%arg0: i256) -> (i256, i256) {
    return %arg0, %arg0 : i256, i256
  }
  func.func @felt_add(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.sext %arg0 : i256 to i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.add %0, %1  : i512
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %4 = llvm.srem %2, %3  : i512
    %5 = llvm.trunc %4 : i512 to i256
    return %5 : i256
  }
  func.func @felt_sub(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.sext %arg0 : i256 to i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.sub %0, %1  : i512
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %4 = llvm.srem %2, %3  : i512
    %5 = llvm.trunc %4 : i512 to i256
    return %5 : i256
  }
  func.func @"struct_construct<Tuple<felt, felt>>"(%arg0: i256, %arg1: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i256, i256)> 
    return %2 : !llvm.struct<(i256, i256)>
  }
  func.func @"store_temp<Tuple<felt, felt>>"(%arg0: !llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)> {
    return %arg0 : !llvm.struct<(i256, i256)>
  }
  func.func @simple_simple_something(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.constant(2 : i256) : i256
    %1:2 = call @"dup<felt>"(%arg0) : (i256) -> (i256, i256)
    %2 = call @felt_add(%1#1, %0) : (i256, i256) -> i256
    %3 = llvm.mlir.constant(2 : i256) : i256
    %4 = call @felt_sub(%1#0, %3) : (i256, i256) -> i256
    %5 = call @"struct_construct<Tuple<felt, felt>>"(%2, %4) : (i256, i256) -> !llvm.struct<(i256, i256)>
    %6 = call @"store_temp<Tuple<felt, felt>>"(%5) : (!llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)>
    return %6 : !llvm.struct<(i256, i256)>
  }
}


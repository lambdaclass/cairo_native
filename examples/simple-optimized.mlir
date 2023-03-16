module {
  func.func @sierra_func_8337028565305778931(%arg0: i256) -> (i256, i256) {
    return %arg0, %arg0 : i256, i256
  }
  func.func @sierra_func_12222469136193516584(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.add %1, %1  : i512
    %3 = llvm.srem %2, %0  : i512
    %4 = llvm.trunc %3 : i512 to i256
    return %4 : i256
  }
  func.func @sierra_func_3615142468138874161(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.sub %1, %1  : i512
    %3 = llvm.srem %2, %0  : i512
    %4 = llvm.trunc %3 : i512 to i256
    return %4 : i256
  }
  func.func @sierra_func_12060906448728155324(%arg0: i256, %arg1: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i256, i256)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i256, i256)> 
    return %2 : !llvm.struct<(i256, i256)>
  }
  func.func @sierra_func_5156205023848900733(%arg0: !llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)> {
    return %arg0 : !llvm.struct<(i256, i256)>
  }
  func.func @sierra_func_13732356220495838267(%arg0: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.add %1, %1  : i512
    %3 = llvm.srem %2, %0  : i512
    %4 = llvm.trunc %3 : i512 to i256
    %5 = llvm.sext %arg0 : i256 to i512
    %6 = llvm.sub %5, %5  : i512
    %7 = llvm.srem %6, %0  : i512
    %8 = llvm.trunc %7 : i512 to i256
    %9 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %10 = llvm.insertvalue %4, %9[0] : !llvm.struct<(i256, i256)> 
    %11 = llvm.insertvalue %8, %10[1] : !llvm.struct<(i256, i256)> 
    return %11 : !llvm.struct<(i256, i256)>
  }
}

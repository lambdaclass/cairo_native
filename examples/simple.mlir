module {
  func.func @sierra_func_8337028565305778931(%arg0: i256) -> (i256, i256) {
    return %arg0, %arg0 : i256, i256
  }
  func.func @sierra_func_12222469136193516584(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.sext %arg0 : i256 to i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.add %0, %1  : i512
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %4 = llvm.srem %2, %3  : i512
    %5 = llvm.trunc %4 : i512 to i256
    return %5 : i256
  }
  func.func @sierra_func_3615142468138874161(%arg0: i256, %arg1: i256) -> i256 {
    %0 = llvm.sext %arg0 : i256 to i512
    %1 = llvm.sext %arg0 : i256 to i512
    %2 = llvm.sub %0, %1  : i512
    %3 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i512) : i512
    %4 = llvm.srem %2, %3  : i512
    %5 = llvm.trunc %4 : i512 to i256
    return %5 : i256
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
    %0 = llvm.mlir.constant(2 : i256) : i256
    %1:2 = call @sierra_func_8337028565305778931(%arg0) : (i256) -> (i256, i256)
    %2 = call @sierra_func_12222469136193516584(%1#1, %0) : (i256, i256) -> i256
    %3 = llvm.mlir.constant(2 : i256) : i256
    %4 = call @sierra_func_3615142468138874161(%1#0, %3) : (i256, i256) -> i256
    %5 = call @sierra_func_12060906448728155324(%2, %4) : (i256, i256) -> !llvm.struct<(i256, i256)>
    %6 = call @sierra_func_5156205023848900733(%5) : (!llvm.struct<(i256, i256)>) -> !llvm.struct<(i256, i256)>
    return %6 : !llvm.struct<(i256, i256)>
  }
}

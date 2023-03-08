module attributes {llvm.data_layout = ""} {
  llvm.func @fib(%arg0: i256, %arg1: i256, %arg2: i256) -> !llvm.struct<(i256, i256)> {
    %0 = llvm.mlir.constant(0 : i256) : i256
    %1 = llvm.icmp "eq" %arg2, %0 : i256
    %2 = llvm.mlir.constant(1 : i256) : i256
    %3:2 = scf.if %1 -> (i256, i256) {
      scf.yield %arg0, %0 : i256, i256
    } else {
      %7 = llvm.add %arg0, %arg1  : i256
      %8 = llvm.sub %arg0, %2  : i256
      %9 = llvm.call @fib(%arg1, %7, %8) : (i256, i256, i256) -> !llvm.struct<(i256, i256)>
      %10 = llvm.extractvalue %9[0] : !llvm.struct<(i256, i256)>
      %11 = llvm.extractvalue %9[1] : !llvm.struct<(i256, i256)>
      %12 = llvm.add %11, %2  : i256
      scf.yield %10, %12 : i256, i256
    }
    %4 = llvm.mlir.undef : !llvm.struct<(i256, i256)>
    %5 = llvm.insertvalue %3#0, %4[0] : !llvm.struct<(i256, i256)>
    %6 = llvm.insertvalue %3#1, %5[1] : !llvm.struct<(i256, i256)>
    llvm.return %6 : !llvm.struct<(i256, i256)>
  }
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(1 : i256) : i256
    %1 = llvm.mlir.constant(1 : i256) : i256
    %2 = llvm.mlir.constant(20 : i256) : i256
    %3 = llvm.call @fib(%0, %1, %2) : (i256, i256, i256) -> !llvm.struct<(i256, i256)>
    %4 = llvm.extractvalue %3[0] : !llvm.struct<(i256, i256)>
    %5 = llvm.extractvalue %3[1] : !llvm.struct<(i256, i256)>
    %6 = llvm.mlir.constant(0 : i32) : i32
    llvm.return %6 : i32
  }
}

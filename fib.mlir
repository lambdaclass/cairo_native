module {
  func.func @fib(%arg0: i256, %arg1: i256, %arg2: i256) -> (i256, i256) {
    %c0_i256 = arith.constant 0 : i256
    %0 = arith.cmpi eq, %arg2, %c0_i256 : i256
    %c1_i256 = arith.constant 1 : i256
    %1:2 = scf.if %0 -> (i256, i256) {
      scf.yield %arg0, %c0_i256 : i256, i256
    } else {
      %2 = arith.addi %arg0, %arg1 : i256
      %3 = arith.subi %arg0, %c1_i256 : i256
      %4:2 = func.call @fib(%arg1, %2, %3) : (i256, i256, i256) -> (i256, i256)
      %5 = arith.addi %4#1, %c1_i256 : i256
      scf.yield %4#0, %5 : i256, i256
    }
    return %1#0, %1#1 : i256, i256
  }
}

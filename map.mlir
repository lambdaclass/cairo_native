module {
  func.func public @"map::map::iterate_map"(%arg0: i252, %arg1: i252) -> i252 {
    cf.br ^bb1(%arg1 : i252)
  ^bb1(%0: i252):  // pred: ^bb0
    cf.br ^bb2(%arg0, %0 : i252, i252)
  ^bb2(%1: i252, %2: i252):  // pred: ^bb1
    %3 = arith.extui %1 : i252 to i504
    %4 = arith.extui %2 : i252 to i504
    %5 = arith.muli %3, %4 : i504
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i504
    %6 = arith.remui %5, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504 : i504
    %7 = arith.trunci %6 : i504 to i252
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    %c-1_i252 = arith.constant -1 : i252
    cf.br ^bb4(%0, %c-1_i252 : i252, i252)
  ^bb4(%8: i252, %9: i252):  // pred: ^bb3
    %10 = arith.extui %8 : i252 to i504
    %11 = arith.extui %9 : i252 to i504
    %12 = arith.muli %10, %11 : i504
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504_0 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i504
    %13 = arith.remui %12, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504_0 : i504
    %14 = arith.trunci %13 : i504 to i252
    cf.br ^bb5(%7 : i252)
  ^bb5(%15: i252):  // pred: ^bb4
    cf.br ^bb6(%14 : i252)
  ^bb6(%16: i252):  // pred: ^bb5
    cf.br ^bb7(%15, %16 : i252, i252)
  ^bb7(%17: i252, %18: i252):  // pred: ^bb6
    %19 = arith.extui %17 : i252 to i504
    %20 = arith.extui %18 : i252 to i504
    %21 = arith.muli %19, %20 : i504
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504_1 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i504
    %22 = arith.remui %21, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504_1 : i504
    %23 = arith.trunci %22 : i504 to i252
    cf.br ^bb8(%23 : i252)
  ^bb8(%24: i252):  // pred: ^bb7
    cf.br ^bb9(%24 : i252)
  ^bb9(%25: i252):  // pred: ^bb8
    return %24 : i252
  }
  func.func public @"map::map::main"(%arg0: !llvm.array<0 x i8>, %arg1: !llvm.array<0 x i8>) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<35 x i8>)>>) {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c1234567890123456789012345678901234567890_i252 = arith.constant 1234567890123456789012345678901234567890 : i252
    cf.br ^bb3
  ^bb3:  // pred: ^bb2
    %c1000_i252 = arith.constant 1000 : i252
    cf.br ^bb4(%arg0 : !llvm.array<0 x i8>)
  ^bb4(%0: !llvm.array<0 x i8>):  // pred: ^bb3
    cf.br ^bb5(%arg1 : !llvm.array<0 x i8>)
  ^bb5(%1: !llvm.array<0 x i8>):  // pred: ^bb4
    cf.br ^bb6(%c1234567890123456789012345678901234567890_i252 : i252)
  ^bb6(%2: i252):  // pred: ^bb5
    cf.br ^bb7(%c1000_i252 : i252)
  ^bb7(%3: i252):  // pred: ^bb6
    cf.br ^bb8(%0, %1, %2, %3 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i252, i252)
  ^bb8(%4: !llvm.array<0 x i8>, %5: !llvm.array<0 x i8>, %6: i252, %7: i252):  // pred: ^bb7
    %8:3 = call @"main[expr20]"(%4, %5, %6, %7) : (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i252, i252) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
    cf.br ^bb9(%8#2 : !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb9(%9: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb8
    %10 = llvm.load %9 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<99 x i8>)>>
    %11 = llvm.extractvalue %10[0] : !llvm.struct<(i1, array<99 x i8>)> 
    cf.switch %11 : i1, [
      default: ^bb10,
      0: ^bb11,
      1: ^bb12
    ]
  ^bb10:  // pred: ^bb9
    %false = arith.constant false
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb11:  // pred: ^bb9
    %12 = llvm.bitcast %9 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %13 = llvm.load %12 : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %14 = llvm.extractvalue %13[1] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    cf.br ^bb13
  ^bb12:  // pred: ^bb9
    %15 = llvm.bitcast %9 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %16 = llvm.load %15 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %17 = llvm.extractvalue %16[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb23
  ^bb13:  // pred: ^bb11
    cf.br ^bb14(%14 : !llvm.struct<(i252, i252, i252)>)
  ^bb14(%18: !llvm.struct<(i252, i252, i252)>):  // pred: ^bb13
    %19 = llvm.extractvalue %18[0] : !llvm.struct<(i252, i252, i252)> 
    %20 = llvm.extractvalue %18[1] : !llvm.struct<(i252, i252, i252)> 
    %21 = llvm.extractvalue %18[2] : !llvm.struct<(i252, i252, i252)> 
    cf.br ^bb15(%19 : i252)
  ^bb15(%22: i252):  // pred: ^bb14
    cf.br ^bb16(%20 : i252)
  ^bb16(%23: i252):  // pred: ^bb15
    cf.br ^bb17(%21 : i252)
  ^bb17(%24: i252):  // pred: ^bb16
    %25 = llvm.mlir.undef : !llvm.struct<(i252)>
    %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<(i252)> 
    cf.br ^bb18(%26 : !llvm.struct<(i252)>)
  ^bb18(%27: !llvm.struct<(i252)>):  // pred: ^bb17
    %false_0 = arith.constant false
    %28 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i252)>)>
    %29 = llvm.insertvalue %false_0, %28[0] : !llvm.struct<(i1, struct<(i252)>)> 
    %30 = llvm.insertvalue %27, %29[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %c1_i64 = arith.constant 1 : i64
    %31 = llvm.alloca %c1_i64 x !llvm.struct<(i1, struct<(i252)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(i252)>)>>
    llvm.store %30, %31 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(i252)>)>>
    %32 = llvm.bitcast %31 : !llvm.ptr<struct<(i1, struct<(i252)>)>> to !llvm.ptr<struct<(i1, array<35 x i8>)>>
    cf.br ^bb19(%8#0 : !llvm.array<0 x i8>)
  ^bb19(%33: !llvm.array<0 x i8>):  // pred: ^bb18
    cf.br ^bb20(%8#1 : !llvm.array<0 x i8>)
  ^bb20(%34: !llvm.array<0 x i8>):  // pred: ^bb19
    cf.br ^bb21(%32 : !llvm.ptr<struct<(i1, array<35 x i8>)>>)
  ^bb21(%35: !llvm.ptr<struct<(i1, array<35 x i8>)>>):  // pred: ^bb20
    cf.br ^bb22(%33, %34, %35 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<35 x i8>)>>)
  ^bb22(%36: !llvm.array<0 x i8>, %37: !llvm.array<0 x i8>, %38: !llvm.ptr<struct<(i1, array<35 x i8>)>>):  // pred: ^bb21
    return %33, %34, %35 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<35 x i8>)>>
  ^bb23:  // pred: ^bb12
    cf.br ^bb24(%17 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb24(%39: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb23
    %true = arith.constant true
    %40 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %41 = llvm.insertvalue %true, %40[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %42 = llvm.insertvalue %39, %41[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %c1_i64_1 = arith.constant 1 : i64
    %43 = llvm.alloca %c1_i64_1 x !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %42, %43 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %44 = llvm.bitcast %43 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>> to !llvm.ptr<struct<(i1, array<35 x i8>)>>
    cf.br ^bb25(%8#0 : !llvm.array<0 x i8>)
  ^bb25(%45: !llvm.array<0 x i8>):  // pred: ^bb24
    cf.br ^bb26(%8#1 : !llvm.array<0 x i8>)
  ^bb26(%46: !llvm.array<0 x i8>):  // pred: ^bb25
    cf.br ^bb27(%44 : !llvm.ptr<struct<(i1, array<35 x i8>)>>)
  ^bb27(%47: !llvm.ptr<struct<(i1, array<35 x i8>)>>):  // pred: ^bb26
    cf.br ^bb28(%45, %46, %47 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<35 x i8>)>>)
  ^bb28(%48: !llvm.array<0 x i8>, %49: !llvm.array<0 x i8>, %50: !llvm.ptr<struct<(i1, array<35 x i8>)>>):  // pred: ^bb27
    return %45, %46, %47 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<35 x i8>)>>
  }
  func.func private @realloc(!llvm.ptr, i64) -> !llvm.ptr
  func.func public @"main[expr20]"(%arg0: !llvm.array<0 x i8>, %arg1: !llvm.array<0 x i8>, %arg2: i252, %arg3: i252) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>) {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %0 = llvm.mlir.undef : !llvm.array<0 x i8>
    cf.br ^bb3(%0 : !llvm.array<0 x i8>)
  ^bb3(%1: !llvm.array<0 x i8>):  // pred: ^bb2
    cf.br ^bb4(%arg0, %arg1, %1 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.array<0 x i8>)
  ^bb4(%2: !llvm.array<0 x i8>, %3: !llvm.array<0 x i8>, %4: !llvm.array<0 x i8>):  // pred: ^bb3
    %false = arith.constant false
    cf.cond_br %false, ^bb5, ^bb48
  ^bb5:  // pred: ^bb4
    cf.br ^bb6
  ^bb6:  // pred: ^bb5
    %c4_i252 = arith.constant 4 : i252
    cf.br ^bb7(%c4_i252 : i252)
  ^bb7(%5: i252):  // pred: ^bb6
    cf.br ^bb8(%arg2 : i252)
  ^bb8(%6: i252):  // pred: ^bb7
    cf.br ^bb9(%5, %6 : i252, i252)
  ^bb9(%7: i252, %8: i252):  // pred: ^bb8
    %9 = call @"map::map::iterate_map"(%7, %8) : (i252, i252) -> i252
    cf.br ^bb10(%arg3 : i252)
  ^bb10(%10: i252):  // pred: ^bb9
    cf.br ^bb11(%2 : !llvm.array<0 x i8>)
  ^bb11(%11: !llvm.array<0 x i8>):  // pred: ^bb10
    cf.br ^bb12(%10 : i252)
  ^bb12(%12: i252):  // pred: ^bb11
    %c0_i252 = arith.constant 0 : i252
    %13 = arith.cmpi eq, %12, %c0_i252 : i252
    cf.cond_br %13, ^bb13, ^bb21
  ^bb13:  // pred: ^bb12
    cf.br ^bb14(%9 : i252)
  ^bb14(%14: i252):  // pred: ^bb13
    cf.br ^bb15(%14, %10, %14 : i252, i252, i252)
  ^bb15(%15: i252, %16: i252, %17: i252):  // pred: ^bb14
    %18 = llvm.mlir.undef : !llvm.struct<(i252, i252, i252)>
    %19 = llvm.insertvalue %15, %18[0] : !llvm.struct<(i252, i252, i252)> 
    %20 = llvm.insertvalue %16, %19[1] : !llvm.struct<(i252, i252, i252)> 
    %21 = llvm.insertvalue %17, %20[2] : !llvm.struct<(i252, i252, i252)> 
    cf.br ^bb16(%21 : !llvm.struct<(i252, i252, i252)>)
  ^bb16(%22: !llvm.struct<(i252, i252, i252)>):  // pred: ^bb15
    %false_0 = arith.constant false
    %23 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i252, i252, i252)>)>
    %24 = llvm.insertvalue %false_0, %23[0] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    %25 = llvm.insertvalue %22, %24[1] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    %c1_i64 = arith.constant 1 : i64
    %26 = llvm.alloca %c1_i64 x !llvm.struct<(i1, struct<(i252, i252, i252)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    llvm.store %25, %26 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %27 = llvm.bitcast %26 : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>> to !llvm.ptr<struct<(i1, array<99 x i8>)>>
    cf.br ^bb17(%11 : !llvm.array<0 x i8>)
  ^bb17(%28: !llvm.array<0 x i8>):  // pred: ^bb16
    cf.br ^bb18(%3 : !llvm.array<0 x i8>)
  ^bb18(%29: !llvm.array<0 x i8>):  // pred: ^bb17
    cf.br ^bb19(%27 : !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb19(%30: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb18
    cf.br ^bb20(%28, %29, %30 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb20(%31: !llvm.array<0 x i8>, %32: !llvm.array<0 x i8>, %33: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb19
    return %28, %29, %30 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>
  ^bb21:  // pred: ^bb12
    cf.br ^bb22(%12 : i252)
  ^bb22(%34: i252):  // pred: ^bb21
    cf.br ^bb23
  ^bb23:  // pred: ^bb22
    %c1_i252 = arith.constant 1 : i252
    cf.br ^bb24(%10, %c1_i252 : i252, i252)
  ^bb24(%35: i252, %36: i252):  // pred: ^bb23
    %37 = arith.subi %35, %36 : i252
    %38 = arith.cmpi ult, %35, %36 : i252
    %39 = scf.if %38 -> (i252) {
      %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 = arith.constant -3618502788666131000275863779947924135206266826270938552493006944358698582015 : i252
      %129 = arith.addi %37, %c-3618502788666131000275863779947924135206266826270938552493006944358698582015_i252 : i252
      scf.yield %129 : i252
    } else {
      scf.yield %37 : i252
    }
    cf.br ^bb25(%11 : !llvm.array<0 x i8>)
  ^bb25(%40: !llvm.array<0 x i8>):  // pred: ^bb24
    cf.br ^bb26(%3 : !llvm.array<0 x i8>)
  ^bb26(%41: !llvm.array<0 x i8>):  // pred: ^bb25
    cf.br ^bb27(%9 : i252)
  ^bb27(%42: i252):  // pred: ^bb26
    cf.br ^bb28(%39 : i252)
  ^bb28(%43: i252):  // pred: ^bb27
    cf.br ^bb29(%40, %41, %42, %43 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i252, i252)
  ^bb29(%44: !llvm.array<0 x i8>, %45: !llvm.array<0 x i8>, %46: i252, %47: i252):  // pred: ^bb28
    %48:3 = call @"main[expr20]"(%44, %45, %46, %47) : (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i252, i252) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
    cf.br ^bb30(%48#2 : !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb30(%49: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb29
    %50 = llvm.load %49 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<99 x i8>)>>
    %51 = llvm.extractvalue %50[0] : !llvm.struct<(i1, array<99 x i8>)> 
    cf.switch %51 : i1, [
      default: ^bb31,
      0: ^bb32,
      1: ^bb33
    ]
  ^bb31:  // pred: ^bb30
    %false_1 = arith.constant false
    cf.assert %false_1, "Invalid enum tag."
    llvm.unreachable
  ^bb32:  // pred: ^bb30
    %52 = llvm.bitcast %49 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %53 = llvm.load %52 : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %54 = llvm.extractvalue %53[1] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    cf.br ^bb34
  ^bb33:  // pred: ^bb30
    %55 = llvm.bitcast %49 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %56 = llvm.load %55 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %57 = llvm.extractvalue %56[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    cf.br ^bb42
  ^bb34:  // pred: ^bb32
    cf.br ^bb35(%54 : !llvm.struct<(i252, i252, i252)>)
  ^bb35(%58: !llvm.struct<(i252, i252, i252)>):  // pred: ^bb34
    %59 = llvm.extractvalue %58[0] : !llvm.struct<(i252, i252, i252)> 
    %60 = llvm.extractvalue %58[1] : !llvm.struct<(i252, i252, i252)> 
    %61 = llvm.extractvalue %58[2] : !llvm.struct<(i252, i252, i252)> 
    cf.br ^bb36(%59, %60, %61 : i252, i252, i252)
  ^bb36(%62: i252, %63: i252, %64: i252):  // pred: ^bb35
    %65 = llvm.mlir.undef : !llvm.struct<(i252, i252, i252)>
    %66 = llvm.insertvalue %62, %65[0] : !llvm.struct<(i252, i252, i252)> 
    %67 = llvm.insertvalue %63, %66[1] : !llvm.struct<(i252, i252, i252)> 
    %68 = llvm.insertvalue %64, %67[2] : !llvm.struct<(i252, i252, i252)> 
    cf.br ^bb37(%68 : !llvm.struct<(i252, i252, i252)>)
  ^bb37(%69: !llvm.struct<(i252, i252, i252)>):  // pred: ^bb36
    %false_2 = arith.constant false
    %70 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i252, i252, i252)>)>
    %71 = llvm.insertvalue %false_2, %70[0] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    %72 = llvm.insertvalue %69, %71[1] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    %c1_i64_3 = arith.constant 1 : i64
    %73 = llvm.alloca %c1_i64_3 x !llvm.struct<(i1, struct<(i252, i252, i252)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    llvm.store %72, %73 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %74 = llvm.bitcast %73 : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>> to !llvm.ptr<struct<(i1, array<99 x i8>)>>
    cf.br ^bb38(%48#0 : !llvm.array<0 x i8>)
  ^bb38(%75: !llvm.array<0 x i8>):  // pred: ^bb37
    cf.br ^bb39(%48#1 : !llvm.array<0 x i8>)
  ^bb39(%76: !llvm.array<0 x i8>):  // pred: ^bb38
    cf.br ^bb40(%74 : !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb40(%77: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb39
    cf.br ^bb41(%75, %76, %77 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb41(%78: !llvm.array<0 x i8>, %79: !llvm.array<0 x i8>, %80: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb40
    return %75, %76, %77 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>
  ^bb42:  // pred: ^bb33
    cf.br ^bb43(%57 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb43(%81: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb42
    %true = arith.constant true
    %82 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %83 = llvm.insertvalue %true, %82[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %84 = llvm.insertvalue %81, %83[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %c1_i64_4 = arith.constant 1 : i64
    %85 = llvm.alloca %c1_i64_4 x !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %84, %85 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %86 = llvm.bitcast %85 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>> to !llvm.ptr<struct<(i1, array<99 x i8>)>>
    cf.br ^bb44(%48#0 : !llvm.array<0 x i8>)
  ^bb44(%87: !llvm.array<0 x i8>):  // pred: ^bb43
    cf.br ^bb45(%48#1 : !llvm.array<0 x i8>)
  ^bb45(%88: !llvm.array<0 x i8>):  // pred: ^bb44
    cf.br ^bb46(%86 : !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb46(%89: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb45
    cf.br ^bb47(%87, %88, %89 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb47(%90: !llvm.array<0 x i8>, %91: !llvm.array<0 x i8>, %92: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb46
    return %87, %88, %89 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>
  ^bb48:  // pred: ^bb4
    cf.br ^bb49(%arg2 : i252)
  ^bb49(%93: i252):  // pred: ^bb48
    cf.br ^bb50(%arg3 : i252)
  ^bb50(%94: i252):  // pred: ^bb49
    cf.br ^bb51
  ^bb51:  // pred: ^bb50
    %95 = llvm.mlir.null : !llvm.ptr<i252>
    %c0_i32 = arith.constant 0 : i32
    %96 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %97 = llvm.insertvalue %95, %96[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %98 = llvm.insertvalue %c0_i32, %97[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %99 = llvm.insertvalue %c0_i32, %98[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb52
  ^bb52:  // pred: ^bb51
    %c375233589013918064796019_i252 = arith.constant 375233589013918064796019 : i252
    cf.br ^bb53(%c375233589013918064796019_i252 : i252)
  ^bb53(%100: i252):  // pred: ^bb52
    cf.br ^bb54(%99, %100 : !llvm.struct<(ptr<i252>, i32, i32)>, i252)
  ^bb54(%101: !llvm.struct<(ptr<i252>, i32, i32)>, %102: i252):  // pred: ^bb53
    %103 = llvm.extractvalue %101[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %104 = llvm.extractvalue %101[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %105 = llvm.extractvalue %101[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %106 = arith.cmpi uge, %104, %105 : i32
    %107:2 = scf.if %106 -> (!llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>) {
      %c8_i32 = arith.constant 8 : i32
      %129 = arith.addi %105, %105 : i32
      %130 = arith.maxui %c8_i32, %129 : i32
      %131 = arith.extui %130 : i32 to i64
      %c32_i64 = arith.constant 32 : i64
      %132 = arith.muli %131, %c32_i64 : i64
      %133 = llvm.bitcast %103 : !llvm.ptr<i252> to !llvm.ptr
      %134 = func.call @realloc(%133, %132) : (!llvm.ptr, i64) -> !llvm.ptr
      %135 = llvm.bitcast %134 : !llvm.ptr to !llvm.ptr<i252>
      %136 = llvm.insertvalue %135, %101[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
      %137 = llvm.insertvalue %130, %136[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
      scf.yield %137, %135 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    } else {
      scf.yield %101, %103 : !llvm.struct<(ptr<i252>, i32, i32)>, !llvm.ptr<i252>
    }
    %108 = llvm.getelementptr %107#1[%104] : (!llvm.ptr<i252>, i32) -> !llvm.ptr<i252>
    llvm.store %102, %108 : !llvm.ptr<i252>
    %c1_i32 = arith.constant 1 : i32
    %109 = arith.addi %104, %c1_i32 : i32
    %110 = llvm.insertvalue %109, %107#0[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    cf.br ^bb55
  ^bb55:  // pred: ^bb54
    %111 = llvm.mlir.undef : !llvm.struct<()>
    cf.br ^bb56(%111, %110 : !llvm.struct<()>, !llvm.struct<(ptr<i252>, i32, i32)>)
  ^bb56(%112: !llvm.struct<()>, %113: !llvm.struct<(ptr<i252>, i32, i32)>):  // pred: ^bb55
    %114 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %115 = llvm.insertvalue %112, %114[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %116 = llvm.insertvalue %113, %115[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    cf.br ^bb57(%116 : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)
  ^bb57(%117: !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>):  // pred: ^bb56
    %true_5 = arith.constant true
    %118 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %119 = llvm.insertvalue %true_5, %118[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %120 = llvm.insertvalue %117, %119[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %c1_i64_6 = arith.constant 1 : i64
    %121 = llvm.alloca %c1_i64_6 x !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %120, %121 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %122 = llvm.bitcast %121 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>> to !llvm.ptr<struct<(i1, array<99 x i8>)>>
    cf.br ^bb58(%2 : !llvm.array<0 x i8>)
  ^bb58(%123: !llvm.array<0 x i8>):  // pred: ^bb57
    cf.br ^bb59(%3 : !llvm.array<0 x i8>)
  ^bb59(%124: !llvm.array<0 x i8>):  // pred: ^bb58
    cf.br ^bb60(%122 : !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb60(%125: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb59
    cf.br ^bb61(%123, %124, %125 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
  ^bb61(%126: !llvm.array<0 x i8>, %127: !llvm.array<0 x i8>, %128: !llvm.ptr<struct<(i1, array<99 x i8>)>>):  // pred: ^bb60
    return %123, %124, %125 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, !llvm.ptr<struct<(i1, array<99 x i8>)>>
  }
}

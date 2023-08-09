#loc = loc(unknown)
module {
  func.func public @"program::program::felt_to_bool"(%arg0: i252 loc(unknown)) -> !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> attributes {llvm.emit_c_interface} {
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %0 = llvm.alloca %c1_i64 x !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>> loc(#loc)
    %c1_i64_0 = arith.constant 1 : i64 loc(#loc1)
    %1 = llvm.alloca %c1_i64_0 x !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> {alignment = 1 : i64} : (i64) -> !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>> loc(#loc1)
    cf.br ^bb1(%arg0 : i252) loc(#loc)
  ^bb1(%2: i252 loc(unknown)):  // pred: ^bb0
    cf.br ^bb2 loc(#loc)
  ^bb2:  // pred: ^bb1
    %c4_i252 = arith.constant 4 : i252 loc(#loc)
    cf.br ^bb3(%2, %c4_i252 : i252, i252) loc(#loc)
  ^bb3(%3: i252 loc(unknown), %4: i252 loc(unknown)):  // pred: ^bb2
    %5 = arith.extui %3 : i252 to i256 loc(#loc)
    %6 = arith.extui %4 : i252 to i256 loc(#loc)
    %7 = arith.subi %5, %6 : i256 loc(#loc)
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i256 loc(#loc)
    %8 = arith.addi %7, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i256 : i256 loc(#loc)
    %9 = arith.cmpi ult, %5, %6 : i256 loc(#loc)
    %10 = arith.select %9, %8, %7 : i256 loc(#loc)
    %11 = arith.trunci %10 : i256 to i252 loc(#loc)
    cf.br ^bb4(%11 : i252) loc(#loc)
  ^bb4(%12: i252 loc(unknown)):  // pred: ^bb3
    cf.br ^bb5(%12 : i252) loc(#loc2)
  ^bb5(%13: i252 loc(unknown)):  // pred: ^bb4
    %c0_i252 = arith.constant 0 : i252 loc(#loc)
    %14 = arith.cmpi eq, %13, %c0_i252 : i252 loc(#loc)
    cf.cond_br %14, ^bb6, ^bb11 loc(#loc)
  ^bb6:  // pred: ^bb5
    cf.br ^bb7 loc(#loc3)
  ^bb7:  // pred: ^bb6
    %15 = llvm.mlir.undef : !llvm.struct<()> loc(#loc3)
    cf.br ^bb8(%15 : !llvm.struct<()>) loc(#loc3)
  ^bb8(%16: !llvm.struct<()> loc(unknown)):  // pred: ^bb7
    %true = arith.constant true loc(#loc1)
    %17 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<()>)> loc(#loc1)
    %18 = llvm.insertvalue %true, %17[0] : !llvm.struct<(i1, array<0 x i8>, struct<()>)>  loc(#loc1)
    %19 = llvm.insertvalue %16, %18[2] : !llvm.struct<(i1, array<0 x i8>, struct<()>)>  loc(#loc1)
    %20 = llvm.bitcast %1 : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>> loc(#loc1)
    llvm.store %19, %20 {alignment = 1 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>> loc(#loc1)
    %21 = llvm.load %1 {alignment = 1 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>> loc(#loc1)
    cf.br ^bb9(%21 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>) loc(#loc1)
  ^bb9(%22: !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(unknown)):  // pred: ^bb8
    cf.br ^bb10 loc(#loc1)
  ^bb10:  // pred: ^bb9
    cf.br ^bb16(%22 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>) loc(#loc4)
  ^bb11:  // pred: ^bb5
    cf.br ^bb12(%13 : i252) loc(#loc)
  ^bb12(%23: i252 loc(unknown)):  // pred: ^bb11
    cf.br ^bb13 loc(#loc)
  ^bb13:  // pred: ^bb12
    %24 = llvm.mlir.undef : !llvm.struct<()> loc(#loc)
    cf.br ^bb14(%24 : !llvm.struct<()>) loc(#loc)
  ^bb14(%25: !llvm.struct<()> loc(unknown)):  // pred: ^bb13
    %false = arith.constant false loc(#loc)
    %26 = llvm.mlir.undef : !llvm.struct<(i1, array<0 x i8>, struct<()>)> loc(#loc)
    %27 = llvm.insertvalue %false, %26[0] : !llvm.struct<(i1, array<0 x i8>, struct<()>)>  loc(#loc)
    %28 = llvm.insertvalue %25, %27[2] : !llvm.struct<(i1, array<0 x i8>, struct<()>)>  loc(#loc)
    %29 = llvm.bitcast %0 : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>> to !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>> loc(#loc)
    llvm.store %28, %29 {alignment = 1 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>)>> loc(#loc)
    %30 = llvm.load %0 {alignment = 1 : i64} : !llvm.ptr<struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>> loc(#loc)
    cf.br ^bb15(%30 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>) loc(#loc)
  ^bb15(%31: !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(unknown)):  // pred: ^bb14
    cf.br ^bb16(%31 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>) loc(#loc)
  ^bb16(%32: !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(unknown)):  // 2 preds: ^bb10, ^bb15
    cf.br ^bb17(%32 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>) loc(#loc)
  ^bb17(%33: !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(unknown)):  // pred: ^bb16
    cf.br ^bb18(%33 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>) loc(#loc)
  ^bb18(%34: !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(unknown)):  // pred: ^bb17
    return %33 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(#loc)
  } loc(#loc)
  func.func public @"program::program::main"() -> !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> attributes {llvm.emit_c_interface} {
    cf.br ^bb1 loc(#loc)
  ^bb1:  // pred: ^bb0
    cf.br ^bb2 loc(#loc)
  ^bb2:  // pred: ^bb1
    %c0_i252 = arith.constant 0 : i252 loc(#loc)
    cf.br ^bb3(%c0_i252 : i252) loc(#loc)
  ^bb3(%0: i252 loc(unknown)):  // pred: ^bb2
    cf.br ^bb4(%0 : i252) loc(#loc5)
  ^bb4(%1: i252 loc(unknown)):  // pred: ^bb3
    %2 = call @"program::program::felt_to_bool"(%1) : (i252) -> !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(#loc6)
    cf.br ^bb5 loc(#loc6)
  ^bb5:  // pred: ^bb4
    %c4_i252 = arith.constant 4 : i252 loc(#loc)
    cf.br ^bb6(%c4_i252 : i252) loc(#loc)
  ^bb6(%3: i252 loc(unknown)):  // pred: ^bb5
    cf.br ^bb7(%3 : i252) loc(#loc7)
  ^bb7(%4: i252 loc(unknown)):  // pred: ^bb6
    %5 = call @"program::program::felt_to_bool"(%4) : (i252) -> !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(#loc8)
    cf.br ^bb8(%2, %5 : !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>) loc(#loc8)
  ^bb8(%6: !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(unknown), %7: !llvm.struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)> loc(unknown)):  // pred: ^bb7
    %8 = llvm.mlir.undef : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> loc(#loc9)
    %9 = llvm.insertvalue %6, %8[0] : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>  loc(#loc9)
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>  loc(#loc9)
    cf.br ^bb9(%10 : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>) loc(#loc9)
  ^bb9(%11: !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> loc(unknown)):  // pred: ^bb8
    cf.br ^bb10(%11 : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)>) loc(#loc)
  ^bb10(%12: !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> loc(unknown)):  // pred: ^bb9
    return %11 : !llvm.struct<(struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>, struct<(i1, array<0 x i8>, struct<()>, array<0 x i8>)>)> loc(#loc)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("program.cairo":8:16)
#loc2 = loc("program.cairo":5:20)
#loc3 = loc("program.cairo":6:16)
#loc4 = loc("/data1/edgar/work/cairo_sierra_2_MLIR/corelib/src/lib.cairo":135:8)
#loc5 = loc("program.cairo":13:26)
#loc6 = loc("program.cairo":13:13)
#loc7 = loc("program.cairo":13:43)
#loc8 = loc("program.cairo":13:30)
#loc9 = loc("program.cairo":13:12)


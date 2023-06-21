module {
  func.func public @"map::map::iterate_map"(%arg0: i252, %arg1: i252) -> i252 {
    %c7237005577332262213973186563042994240829374041602535252466099000494570602495_i504 = arith.constant 7237005577332262213973186563042994240829374041602535252466099000494570602495 : i504
    %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504 = arith.constant 3618502788666131213697322783095070105623107215331596699973092056135872020481 : i504
    %0 = arith.extui %arg0 : i252 to i504
    %1 = arith.extui %arg1 : i252 to i504
    %2 = arith.muli %0, %1 : i504
    %3 = arith.remui %2, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504 : i504
    %4 = arith.trunci %3 : i504 to i252
    %5 = arith.extui %arg1 : i252 to i504
    %6 = arith.muli %5, %c7237005577332262213973186563042994240829374041602535252466099000494570602495_i504 : i504
    %7 = arith.remui %6, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504 : i504
    %8 = arith.trunci %7 : i504 to i252
    %9 = arith.extui %4 : i252 to i504
    %10 = arith.extui %8 : i252 to i504
    %11 = arith.muli %9, %10 : i504
    %12 = arith.remui %11, %c3618502788666131213697322783095070105623107215331596699973092056135872020481_i504 : i504
    %13 = arith.trunci %12 : i504 to i252
    return %13 : i252
  }
  func.func public @"map::map::main"(%arg0: !llvm.array<0 x none>, %arg1: !llvm.array<0 x none>) -> (!llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<35 x i8>)>>) {
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %false = arith.constant false
    %c1000_i252 = arith.constant 1000 : i252
    %c1234567890123456789012345678901234567890_i252 = arith.constant 1234567890123456789012345678901234567890 : i252
    %0:3 = call @"main[expr20]"(%arg0, %arg1, %c1234567890123456789012345678901234567890_i252, %c1000_i252) : (!llvm.array<0 x none>, !llvm.array<0 x none>, i252, i252) -> (!llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
    %1 = llvm.load %0#2 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<99 x i8>)>>
    %2 = llvm.extractvalue %1[0] : !llvm.struct<(i1, array<99 x i8>)> 
    cf.switch %2 : i1, [
      default: ^bb1,
      0: ^bb2,
      1: ^bb3
    ]
  ^bb1:  // pred: ^bb0
    cf.assert %false, "Invalid enum tag."
    llvm.unreachable
  ^bb2:  // pred: ^bb0
    %3 = llvm.bitcast %0#2 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %4 = llvm.load %3 : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %5 = llvm.extractvalue %4[1] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    %6 = llvm.extractvalue %5[2] : !llvm.struct<(i252, i252, i252)> 
    %7 = llvm.mlir.undef : !llvm.struct<(i252)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(i252)> 
    %9 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i252)>)>
    %10 = llvm.insertvalue %false, %9[0] : !llvm.struct<(i1, struct<(i252)>)> 
    %11 = llvm.insertvalue %8, %10[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %12 = llvm.alloca %c1_i64 x !llvm.struct<(i1, struct<(i252)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(i252)>)>>
    llvm.store %11, %12 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(i252)>)>>
    %13 = llvm.bitcast %12 : !llvm.ptr<struct<(i1, struct<(i252)>)>> to !llvm.ptr<struct<(i1, array<35 x i8>)>>
    return %0#0, %0#1, %13 : !llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<35 x i8>)>>
  ^bb3:  // pred: ^bb0
    %14 = llvm.bitcast %0#2 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %15 = llvm.load %14 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %16 = llvm.extractvalue %15[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %17 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %18 = llvm.insertvalue %true, %17[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %19 = llvm.insertvalue %16, %18[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %20 = llvm.alloca %c1_i64 x !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %19, %20 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %21 = llvm.bitcast %20 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>> to !llvm.ptr<struct<(i1, array<35 x i8>)>>
    return %0#0, %0#1, %21 : !llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<35 x i8>)>>
  }
  func.func private @realloc(!llvm.ptr, i64) -> !llvm.ptr
  func.func public @"main[expr20]"(%arg0: !llvm.array<0 x none>, %arg1: !llvm.array<0 x none>, %arg2: i252, %arg3: i252) -> (!llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<99 x i8>)>>) {
    %c1_i32 = arith.constant 1 : i32
    %c256_i64 = arith.constant 256 : i64
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c375233589013918064796019_i252 = arith.constant 375233589013918064796019 : i252
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = llvm.mlir.null : !llvm.ptr<i252>
    %1 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %2 = llvm.insertvalue %0, %1[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %3 = llvm.insertvalue %c0_i32, %2[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %4 = llvm.insertvalue %c0_i32, %3[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %5 = llvm.bitcast %0 : !llvm.ptr<i252> to !llvm.ptr
    %6 = call @realloc(%5, %c256_i64) : (!llvm.ptr, i64) -> !llvm.ptr
    %7 = llvm.bitcast %6 : !llvm.ptr to !llvm.ptr<i252>
    %8 = llvm.insertvalue %7, %4[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %9 = llvm.insertvalue %c8_i32, %8[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    llvm.store %c375233589013918064796019_i252, %7 : !llvm.ptr<i252>
    %10 = llvm.insertvalue %c1_i32, %9[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %11 = llvm.mlir.undef : !llvm.struct<()>
    %12 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %14 = llvm.insertvalue %10, %13[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %15 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %16 = llvm.insertvalue %true, %15[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %17 = llvm.insertvalue %14, %16[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %18 = llvm.alloca %c1_i64 x !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %17, %18 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %19 = llvm.bitcast %18 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>> to !llvm.ptr<struct<(i1, array<99 x i8>)>>
    return %arg0, %arg1, %19 : !llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<99 x i8>)>>
  }
}


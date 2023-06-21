module attributes {llvm.data_layout = ""} {
  llvm.func @abort()
  llvm.func @puts(!llvm.ptr<i8>)
  llvm.mlir.global private constant @assert_msg_0(dense<[73, 110, 118, 97, 108, 105, 100, 32, 101, 110, 117, 109, 32, 116, 97, 103, 46, 0]> : tensor<18xi8>) {addr_space = 0 : i32} : !llvm.array<18 x i8>
  llvm.func @"map::map::iterate_map"(%arg0: i252, %arg1: i252) -> i252 attributes {sym_visibility = "public"} {
    %0 = llvm.mlir.constant(7237005577332262213973186563042994240829374041602535252466099000494570602495 : i504) : i504
    %1 = llvm.mlir.constant(3618502788666131213697322783095070105623107215331596699973092056135872020481 : i504) : i504
    %2 = llvm.zext %arg0 : i252 to i504
    %3 = llvm.zext %arg1 : i252 to i504
    %4 = llvm.mul %2, %3  : i504
    %5 = llvm.urem %4, %1  : i504
    %6 = llvm.trunc %5 : i504 to i252
    %7 = llvm.mul %3, %0  : i504
    %8 = llvm.urem %7, %1  : i504
    %9 = llvm.trunc %8 : i504 to i252
    %10 = llvm.zext %6 : i252 to i504
    %11 = llvm.zext %9 : i252 to i504
    %12 = llvm.mul %10, %11  : i504
    %13 = llvm.urem %12, %1  : i504
    %14 = llvm.trunc %13 : i504 to i252
    llvm.return %14 : i252
  }
  func.func public @"map::map::main"(%arg0: !llvm.array<0 x none>, %arg1: !llvm.array<0 x none>) -> (!llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<35 x i8>)>>) {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(false) : i1
    %3 = llvm.mlir.constant(1000 : i252) : i252
    %4 = llvm.mlir.constant(1234567890123456789012345678901234567890 : i252) : i252
    %5:3 = call @"main[expr20]"(%arg0, %arg1, %4, %3) : (!llvm.array<0 x none>, !llvm.array<0 x none>, i252, i252) -> (!llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<99 x i8>)>>)
    %6 = llvm.load %5#2 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, array<99 x i8>)>>
    %7 = llvm.extractvalue %6[0] : !llvm.struct<(i1, array<99 x i8>)> 
    llvm.switch %7 : i1, ^bb1 [
      0: ^bb3,
      1: ^bb4
    ]
  ^bb1:  // pred: ^bb0
    llvm.cond_br %2, ^bb2, ^bb6
  ^bb2:  // pred: ^bb1
    llvm.unreachable
  ^bb3:  // pred: ^bb0
    %8 = llvm.bitcast %5#2 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %9 = llvm.load %8 : !llvm.ptr<struct<(i1, struct<(i252, i252, i252)>)>>
    %10 = llvm.extractvalue %9[1] : !llvm.struct<(i1, struct<(i252, i252, i252)>)> 
    %11 = llvm.extractvalue %10[2] : !llvm.struct<(i252, i252, i252)> 
    %12 = llvm.mlir.undef : !llvm.struct<(i252)>
    %13 = llvm.insertvalue %11, %12[0] : !llvm.struct<(i252)> 
    %14 = llvm.mlir.undef : !llvm.struct<(i1, struct<(i252)>)>
    %15 = llvm.insertvalue %2, %14[0] : !llvm.struct<(i1, struct<(i252)>)> 
    %16 = llvm.insertvalue %13, %15[1] : !llvm.struct<(i1, struct<(i252)>)> 
    %17 = llvm.alloca %1 x !llvm.struct<(i1, struct<(i252)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(i252)>)>>
    llvm.store %16, %17 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(i252)>)>>
    %18 = llvm.bitcast %17 : !llvm.ptr<struct<(i1, struct<(i252)>)>> to !llvm.ptr<struct<(i1, array<35 x i8>)>>
    cf.br ^bb5(%5#0, %5#1, %18 : !llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<35 x i8>)>>)
  ^bb4:  // pred: ^bb0
    %19 = llvm.bitcast %5#2 : !llvm.ptr<struct<(i1, array<99 x i8>)>> to !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %20 = llvm.load %19 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %21 = llvm.extractvalue %20[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %22 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %23 = llvm.insertvalue %0, %22[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %24 = llvm.insertvalue %21, %23[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %25 = llvm.alloca %1 x !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %24, %25 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %26 = llvm.bitcast %25 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>> to !llvm.ptr<struct<(i1, array<35 x i8>)>>
    cf.br ^bb5(%5#0, %5#1, %26 : !llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<35 x i8>)>>)
  ^bb5(%27: !llvm.array<0 x none>, %28: !llvm.array<0 x none>, %29: !llvm.ptr<struct<(i1, array<35 x i8>)>>):  // 2 preds: ^bb3, ^bb4
    return %27, %28, %29 : !llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<35 x i8>)>>
  ^bb6:  // pred: ^bb1
    %30 = llvm.mlir.addressof @assert_msg_0 : !llvm.ptr<array<18 x i8>>
    %31 = llvm.getelementptr %30[0] : (!llvm.ptr<array<18 x i8>>) -> !llvm.ptr<i8>
    llvm.call @puts(%31) : (!llvm.ptr<i8>) -> ()
    llvm.call @abort() : () -> ()
    llvm.unreachable
  }
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
  func.func public @"main[expr20]"(%arg0: !llvm.array<0 x none>, %arg1: !llvm.array<0 x none>, %arg2: i252, %arg3: i252) -> (!llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<99 x i8>)>>) {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(8 : i32) : i32
    %2 = llvm.mlir.constant(true) : i1
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.mlir.constant(375233589013918064796019 : i252) : i252
    %5 = llvm.mlir.constant(0 : i32) : i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.mlir.null : !llvm.ptr<i252>
    %8 = llvm.mlir.undef : !llvm.struct<(ptr<i252>, i32, i32)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %10 = llvm.insertvalue %5, %9[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %11 = llvm.insertvalue %5, %10[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %12 = llvm.bitcast %7 : !llvm.ptr<i252> to !llvm.ptr
    %13 = llvm.call @realloc(%12, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %14 = llvm.bitcast %13 : !llvm.ptr to !llvm.ptr<i252>
    %15 = llvm.insertvalue %14, %11[0] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %16 = llvm.insertvalue %1, %15[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    llvm.store %4, %14 : !llvm.ptr<i252>
    %17 = llvm.insertvalue %3, %16[1] : !llvm.struct<(ptr<i252>, i32, i32)> 
    %18 = llvm.mlir.undef : !llvm.struct<()>
    %19 = llvm.mlir.undef : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>
    %20 = llvm.insertvalue %18, %19[0] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %21 = llvm.insertvalue %17, %20[1] : !llvm.struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)> 
    %22 = llvm.mlir.undef : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>
    %23 = llvm.insertvalue %2, %22[0] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %24 = llvm.insertvalue %21, %23[1] : !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> 
    %25 = llvm.alloca %6 x !llvm.struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)> {alignment = 8 : i64} : (i64) -> !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    llvm.store %24, %25 {alignment = 8 : i64} : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>>
    %26 = llvm.bitcast %25 : !llvm.ptr<struct<(i1, struct<(struct<()>, struct<(ptr<i252>, i32, i32)>)>)>> to !llvm.ptr<struct<(i1, array<99 x i8>)>>
    return %arg0, %arg1, %26 : !llvm.array<0 x none>, !llvm.array<0 x none>, !llvm.ptr<struct<(i1, array<99 x i8>)>>
  }
}


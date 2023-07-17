module {
  func.func private @cairo_native__alloc_dict() -> !llvm.ptr
  func.func private @realloc(!llvm.ptr, i64) -> !llvm.ptr
  func.func private @cairo_native__dict_get(!llvm.ptr, !llvm.ptr<i252>) -> !llvm.ptr
  func.func private @cairo_native__dict_insert(!llvm.ptr, !llvm.ptr<i252>, !llvm.ptr) -> !llvm.ptr
  func.func public @"ex_dict::ex_dict::main"(%arg0: !llvm.array<0 x i8>, %arg1: !llvm.array<0 x i8>, %arg2: i64) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, i32) attributes {llvm.emit_c_interface} {
    %c1_i32 = arith.constant 1 : i32
    %c4_i64 = arith.constant 4 : i64
    %c2_i252 = arith.constant 2 : i252
    %0 = call @cairo_native__alloc_dict() : () -> !llvm.ptr
    %1 = llvm.alloca %c2_i252 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252>
    llvm.store %c2_i252, %1 {alignment = 8 : i64} : !llvm.ptr<i252>
    %2 = call @cairo_native__dict_get(%0, %1) : (!llvm.ptr, !llvm.ptr<i252>) -> !llvm.ptr
    %3 = llvm.mlir.null : !llvm.ptr
    %4 = llvm.icmp "eq" %2, %3 : !llvm.ptr
    cf.cond_br %4, ^bb1, ^bb2(%2 : !llvm.ptr)
  ^bb1:  // pred: ^bb0
    %5 = call @realloc(%2, %c4_i64) : (!llvm.ptr, i64) -> !llvm.ptr
    cf.br ^bb2(%5 : !llvm.ptr)
  ^bb2(%6: !llvm.ptr):  // 2 preds: ^bb0, ^bb1
    llvm.store %c1_i32, %6 {alignment = 4 : i64} : i32, !llvm.ptr
    %7 = llvm.alloca %c2_i252 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252>
    llvm.store %c2_i252, %7 {alignment = 8 : i64} : !llvm.ptr<i252>
    %8 = call @cairo_native__dict_insert(%0, %7, %6) : (!llvm.ptr, !llvm.ptr<i252>, !llvm.ptr) -> !llvm.ptr
    %9 = llvm.alloca %c2_i252 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252>
    llvm.store %c2_i252, %9 {alignment = 8 : i64} : !llvm.ptr<i252>
    %10 = call @cairo_native__dict_get(%0, %9) : (!llvm.ptr, !llvm.ptr<i252>) -> !llvm.ptr
    %11 = llvm.mlir.null : !llvm.ptr
    %12 = llvm.icmp "eq" %10, %11 : !llvm.ptr
    cf.cond_br %12, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %13 = call @realloc(%10, %c4_i64) : (!llvm.ptr, i64) -> !llvm.ptr
    %14 = llvm.mlir.undef : i32
    cf.br ^bb5(%13, %14 : !llvm.ptr, i32)
  ^bb4:  // pred: ^bb2
    %15 = llvm.load %10 {alignment = 4 : i64} : !llvm.ptr -> i32
    cf.br ^bb5(%10, %15 : !llvm.ptr, i32)
  ^bb5(%16: !llvm.ptr, %17: i32):  // 2 preds: ^bb3, ^bb4
    llvm.store %17, %16 {alignment = 4 : i64} : i32, !llvm.ptr
    %18 = llvm.alloca %c2_i252 x i252 {alignment = 8 : i64} : (i252) -> !llvm.ptr<i252>
    llvm.store %c2_i252, %18 {alignment = 8 : i64} : !llvm.ptr<i252>
    %19 = call @cairo_native__dict_insert(%0, %18, %16) : (!llvm.ptr, !llvm.ptr<i252>, !llvm.ptr) -> !llvm.ptr
    %20:4 = call @"core::dict::Felt252DictDestruct::<core::integer::u32, core::integer::u32Drop, core::integer::U32Felt252DictValue>::destruct"(%arg0, %arg1, %arg2, %0) : (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.struct<()>)
    return %20#0, %20#1, %20#2, %17 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, i32
  }
  func.func public @"core::dict::Felt252DictDestruct::<core::integer::u32, core::integer::u32Drop, core::integer::U32Felt252DictValue>::destruct"(%arg0: !llvm.array<0 x i8>, %arg1: !llvm.array<0 x i8>, %arg2: i64, %arg3: !llvm.ptr) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.struct<()>) attributes {llvm.emit_c_interface} {
    %0:4 = call @"core::dict::Felt252DictImpl::<core::integer::u32, core::integer::U32Felt252DictValue>::squash"(%arg0, %arg1, %arg2, %arg3) : (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr)
    %1 = llvm.mlir.undef : !llvm.struct<()>
    return %0#0, %0#1, %0#2, %1 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.struct<()>
  }
  func.func public @"core::dict::Felt252DictImpl::<core::integer::u32, core::integer::U32Felt252DictValue>::squash"(%arg0: !llvm.array<0 x i8>, %arg1: !llvm.array<0 x i8>, %arg2: i64, %arg3: !llvm.ptr) -> (!llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr) attributes {llvm.emit_c_interface} {
    return %arg0, %arg1, %arg2, %arg3 : !llvm.array<0 x i8>, !llvm.array<0 x i8>, i64, !llvm.ptr
  }
}


module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @u8_to_felt252(%arg0: i8) -> i256 {
    %0 = llvm.zext %arg0 : i8 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u16_to_felt252(%arg0: i16) -> i256 {
    %0 = llvm.zext %arg0 : i16 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u32_to_felt252(%arg0: i32) -> i256 {
    %0 = llvm.zext %arg0 : i32 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u64_to_felt252(%arg0: i64) -> i256 {
    %0 = llvm.zext %arg0 : i64 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u128_to_felt252(%arg0: i128) -> i256 {
    %0 = llvm.zext %arg0 : i128 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u8_wide_mul(%arg0: i8, %arg1: i8) -> i16 {
    %0 = llvm.zext %arg0 : i8 to i16
    %1 = llvm.zext %arg1 : i8 to i16
    %2 = llvm.mul %0, %1  : i16
    llvm.return %2 : i16
  }
  llvm.func internal @u16_wide_mul(%arg0: i16, %arg1: i16) -> i32 {
    %0 = llvm.zext %arg0 : i16 to i32
    %1 = llvm.zext %arg1 : i16 to i32
    %2 = llvm.mul %0, %1  : i32
    llvm.return %2 : i32
  }
  llvm.func internal @u32_wide_mul(%arg0: i32, %arg1: i32) -> i64 {
    %0 = llvm.zext %arg0 : i32 to i64
    %1 = llvm.zext %arg1 : i32 to i64
    %2 = llvm.mul %0, %1  : i64
    llvm.return %2 : i64
  }
  llvm.func internal @u64_wide_mul(%arg0: i64, %arg1: i64) -> i128 {
    %0 = llvm.zext %arg0 : i64 to i128
    %1 = llvm.zext %arg1 : i64 to i128
    %2 = llvm.mul %0, %1  : i128
    llvm.return %2 : i128
  }
  func.func @u128_wide_mul(%arg0: none, %arg1: i128, %arg2: i128) -> (none, i128, i128) attributes {llvm.linkage = #llvm.linkage<internal>} {
    %0 = llvm.zext %arg1 : i128 to i256
    %1 = llvm.zext %arg2 : i128 to i256
    %2 = llvm.mul %0, %1  : i256
    %3 = llvm.trunc %2 : i256 to i128
    %4 = llvm.mlir.constant(128 : i256) : i256
    %5 = llvm.lshr %2, %4  : i256
    %6 = llvm.trunc %5 : i256 to i128
    return %arg0, %3, %6 : none, i128, i128
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  func.func @"uint::uint::main"(%arg0: none) -> (none, !llvm.struct<()>) attributes {llvm.emit_c_interface} {
    cf.br ^bb1(%arg0 : none)
  ^bb1(%0: none):  // pred: ^bb0
    %1 = llvm.mlir.constant(0 : i8) : i8
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.mlir.constant(0 : i64) : i64
    %5 = llvm.mlir.constant(0 : i128) : i128
    %6 = llvm.call @u8_to_felt252(%1) : (i8) -> i256
    %7 = llvm.call @u16_to_felt252(%2) : (i16) -> i256
    %8 = llvm.call @u32_to_felt252(%3) : (i32) -> i256
    %9 = llvm.call @u64_to_felt252(%4) : (i64) -> i256
    %10 = llvm.call @u128_to_felt252(%5) : (i128) -> i256
    %11 = llvm.call @u8_wide_mul(%1, %1) : (i8, i8) -> i16
    %12 = llvm.call @u16_wide_mul(%2, %2) : (i16, i16) -> i32
    %13 = llvm.call @u32_wide_mul(%3, %3) : (i32, i32) -> i64
    %14 = llvm.call @u64_wide_mul(%4, %4) : (i64, i64) -> i128
    %15:3 = call @u128_wide_mul(%0, %5, %5) : (none, i128, i128) -> (none, i128, i128)
    %16 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    return %15#0, %16 : none, !llvm.struct<()>
  }
}

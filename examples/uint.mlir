module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @u8_to_felt252(%arg0: i8) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u16_to_felt252(%arg0: i16) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u32_to_felt252(%arg0: i32) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i32 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u64_to_felt252(%arg0: i64) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i64 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u128_to_felt252(%arg0: i128) -> i256 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i128 to i256
    llvm.return %0 : i256
  }
  llvm.func internal @u8_wide_mul(%arg0: i8, %arg1: i8) -> i16 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i16
    %1 = llvm.zext %arg1 : i8 to i16
    %2 = llvm.mul %0, %1  : i16
    llvm.return %2 : i16
  }
  llvm.func internal @u16_wide_mul(%arg0: i16, %arg1: i16) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i32
    %1 = llvm.zext %arg1 : i16 to i32
    %2 = llvm.mul %0, %1  : i32
    llvm.return %2 : i32
  }
  llvm.func internal @u32_wide_mul(%arg0: i32, %arg1: i32) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i32 to i64
    %1 = llvm.zext %arg1 : i32 to i64
    %2 = llvm.mul %0, %1  : i64
    llvm.return %2 : i64
  }
  llvm.func internal @u64_wide_mul(%arg0: i64, %arg1: i64) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i64 to i128
    %1 = llvm.zext %arg1 : i64 to i128
    %2 = llvm.mul %0, %1  : i128
    llvm.return %2 : i128
  }
  llvm.func internal @u128_wide_mul(%arg0: i128, %arg1: i128) -> !llvm.struct<(i128, i128)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i128 to i256
    %1 = llvm.zext %arg1 : i128 to i256
    %2 = llvm.mul %0, %1  : i256
    %3 = llvm.trunc %2 : i256 to i128
    %4 = llvm.mlir.constant(128 : i256) : i256
    %5 = llvm.lshr %2, %4  : i256
    %6 = llvm.trunc %5 : i256 to i128
    %7 = llvm.mlir.undef : !llvm.struct<(i128, i128)>
    %8 = llvm.insertvalue %6, %7[0] : !llvm.struct<(i128, i128)> 
    %9 = llvm.insertvalue %3, %8[1] : !llvm.struct<(i128, i128)> 
    llvm.return %9 : !llvm.struct<(i128, i128)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed ()>
    llvm.return %0 : !llvm.struct<packed ()>
  }
  llvm.func @"uint::uint::main"() -> !llvm.struct<packed ()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.call @u8_to_felt252(%0) : (i8) -> i256
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.call @u16_to_felt252(%2) : (i16) -> i256
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.call @u32_to_felt252(%4) : (i32) -> i256
    %6 = llvm.mlir.constant(0 : i64) : i64
    %7 = llvm.call @u64_to_felt252(%6) : (i64) -> i256
    %8 = llvm.mlir.constant(0 : i128) : i128
    %9 = llvm.call @u128_to_felt252(%8) : (i128) -> i256
    %10 = llvm.call @u8_wide_mul(%0, %0) : (i8, i8) -> i16
    %11 = llvm.call @u16_wide_mul(%2, %2) : (i16, i16) -> i32
    %12 = llvm.call @u32_wide_mul(%4, %4) : (i32, i32) -> i64
    %13 = llvm.call @u64_wide_mul(%6, %6) : (i64, i64) -> i128
    %14 = llvm.call @u128_wide_mul(%8, %8) : (i128, i128) -> !llvm.struct<(i128, i128)>
    %15 = llvm.extractvalue %14[0] : !llvm.struct<(i128, i128)> 
    %16 = llvm.extractvalue %14[1] : !llvm.struct<(i128, i128)> 
    %17 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<packed ()>
    llvm.return %17 : !llvm.struct<packed ()>
  }
  llvm.func @"_mlir_ciface_uint::uint::main"(%arg0: !llvm.ptr<struct<packed ()>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"uint::uint::main"() : () -> !llvm.struct<packed ()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed ()>>
    llvm.return
  }
}

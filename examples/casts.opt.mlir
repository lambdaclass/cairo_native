module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"upcast<u8, u16>"(%arg0: i8) -> i16 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i16
    llvm.return %0 : i16
  }
  llvm.func internal @"upcast<u8, u32>"(%arg0: i8) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i32
    llvm.return %0 : i32
  }
  llvm.func internal @"upcast<u8, u64>"(%arg0: i8) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u8, u128>"(%arg0: i8) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u16, u32>"(%arg0: i16) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i32
    llvm.return %0 : i32
  }
  llvm.func internal @"upcast<u16, u64>"(%arg0: i16) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u16, u128>"(%arg0: i16) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u32, u64>"(%arg0: i32) -> i64 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i32 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u32, u128>"(%arg0: i32) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i32 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u64, u128>"(%arg0: i64) -> i128 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.zext %arg0 : i64 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func @"casts::casts::main"() -> !llvm.struct<()> attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.mlir.constant(0 : i8) : i8
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %4 = llvm.call @"upcast<u8, u16>"(%3) : (i8) -> i16
    %5 = llvm.call @"upcast<u8, u32>"(%3) : (i8) -> i32
    %6 = llvm.call @"upcast<u8, u64>"(%3) : (i8) -> i64
    %7 = llvm.call @"upcast<u8, u128>"(%3) : (i8) -> i128
    %8 = llvm.call @"upcast<u16, u32>"(%2) : (i16) -> i32
    %9 = llvm.call @"upcast<u16, u64>"(%2) : (i16) -> i64
    %10 = llvm.call @"upcast<u16, u128>"(%2) : (i16) -> i128
    %11 = llvm.call @"upcast<u32, u64>"(%1) : (i32) -> i64
    %12 = llvm.call @"upcast<u32, u128>"(%1) : (i32) -> i128
    %13 = llvm.call @"upcast<u64, u128>"(%0) : (i64) -> i128
    %14 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %14 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_casts::casts::main"(%arg0: !llvm.ptr<struct<()>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.mlir.constant(0 : i8) : i8
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %4 = llvm.call @"upcast<u8, u16>"(%3) : (i8) -> i16
    %5 = llvm.call @"upcast<u8, u32>"(%3) : (i8) -> i32
    %6 = llvm.call @"upcast<u8, u64>"(%3) : (i8) -> i64
    %7 = llvm.call @"upcast<u8, u128>"(%3) : (i8) -> i128
    %8 = llvm.call @"upcast<u16, u32>"(%2) : (i16) -> i32
    %9 = llvm.call @"upcast<u16, u64>"(%2) : (i16) -> i64
    %10 = llvm.call @"upcast<u16, u128>"(%2) : (i16) -> i128
    %11 = llvm.call @"upcast<u32, u64>"(%1) : (i32) -> i64
    %12 = llvm.call @"upcast<u32, u128>"(%1) : (i32) -> i128
    %13 = llvm.call @"upcast<u64, u128>"(%0) : (i64) -> i128
    %14 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.br ^bb2(%14 : !llvm.struct<()>)
  ^bb2(%15: !llvm.struct<()>):  // pred: ^bb1
    llvm.store %15, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}

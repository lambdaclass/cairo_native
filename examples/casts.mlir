module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"upcast<u8, u16>"(%arg0: i8) -> i16 {
    %0 = llvm.zext %arg0 : i8 to i16
    llvm.return %0 : i16
  }
  llvm.func internal @"upcast<u8, u32>"(%arg0: i8) -> i32 {
    %0 = llvm.zext %arg0 : i8 to i32
    llvm.return %0 : i32
  }
  llvm.func internal @"upcast<u8, u64>"(%arg0: i8) -> i64 {
    %0 = llvm.zext %arg0 : i8 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u8, u128>"(%arg0: i8) -> i128 {
    %0 = llvm.zext %arg0 : i8 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u16, u32>"(%arg0: i16) -> i32 {
    %0 = llvm.zext %arg0 : i16 to i32
    llvm.return %0 : i32
  }
  llvm.func internal @"upcast<u16, u64>"(%arg0: i16) -> i64 {
    %0 = llvm.zext %arg0 : i16 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u16, u128>"(%arg0: i16) -> i128 {
    %0 = llvm.zext %arg0 : i16 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u32, u64>"(%arg0: i32) -> i64 {
    %0 = llvm.zext %arg0 : i32 to i64
    llvm.return %0 : i64
  }
  llvm.func internal @"upcast<u32, u128>"(%arg0: i32) -> i128 {
    %0 = llvm.zext %arg0 : i32 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"upcast<u64, u128>"(%arg0: i64) -> i128 {
    %0 = llvm.zext %arg0 : i64 to i128
    llvm.return %0 : i128
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @print_Unit(%arg0: !llvm.struct<()>) {
    llvm.return
  }
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"casts::casts::main"() : () -> !llvm.struct<()>
    llvm.call @print_Unit(%0) : (!llvm.struct<()>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"casts::casts::main"() -> !llvm.struct<()> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(0 : i8) : i8
    %1 = llvm.mlir.constant(0 : i16) : i16
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(0 : i64) : i64
    %4 = llvm.mlir.constant(0 : i128) : i128
    %5 = llvm.call @"upcast<u8, u16>"(%0) : (i8) -> i16
    %6 = llvm.call @"upcast<u8, u32>"(%0) : (i8) -> i32
    %7 = llvm.call @"upcast<u8, u64>"(%0) : (i8) -> i64
    %8 = llvm.call @"upcast<u8, u128>"(%0) : (i8) -> i128
    %9 = llvm.call @"upcast<u16, u32>"(%1) : (i16) -> i32
    %10 = llvm.call @"upcast<u16, u64>"(%1) : (i16) -> i64
    %11 = llvm.call @"upcast<u16, u128>"(%1) : (i16) -> i128
    %12 = llvm.call @"upcast<u32, u64>"(%2) : (i32) -> i64
    %13 = llvm.call @"upcast<u32, u128>"(%2) : (i32) -> i128
    %14 = llvm.call @"upcast<u64, u128>"(%3) : (i64) -> i128
    %15 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    llvm.return %15 : !llvm.struct<()>
  }
  llvm.func @"_mlir_ciface_casts::casts::main"(%arg0: !llvm.ptr<struct<()>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"casts::casts::main"() : () -> !llvm.struct<()>
    llvm.store %0, %arg0 : !llvm.ptr<struct<()>>
    llvm.return
  }
}

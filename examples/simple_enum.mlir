module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"enum_init<simple_enum::simple_enum::MyEnum, 0>"(%arg0: i8) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i8)>
    llvm.store %arg0, %4 : i8, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<simple_enum::simple_enum::MyEnum, 1>"(%arg0: i16) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i16)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<simple_enum::simple_enum::MyEnum, 2>"(%arg0: i32) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(2 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i32)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<simple_enum::simple_enum::MyEnum, 3>"(%arg0: i64) -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(3 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i64)>
    llvm.store %arg0, %4 : i64, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func internal @print_u8(%arg0: i8) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.zext %arg0 : i8 to i32
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
    %3 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %3, %2 : !llvm.array<4 x i8>, !llvm.ptr
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.call @dprintf(%4, %2, %0) : (i32, !llvm.ptr, i32) -> i32
    llvm.return
  }
  llvm.func internal @print_u16(%arg0: i16) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.zext %arg0 : i16 to i32
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
    %3 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %3, %2 : !llvm.array<4 x i8>, !llvm.ptr
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.call @dprintf(%4, %2, %0) : (i32, !llvm.ptr, i32) -> i32
    llvm.return
  }
  llvm.func internal @print_u32(%arg0: i32) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %2, %1 : !llvm.array<4 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1, %arg0) : (i32, !llvm.ptr, i32) -> i32
    llvm.return
  }
  llvm.func internal @print_u64(%arg0: i64) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.mlir.constant(5 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[37, 108, 88, 10, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %2, %1 : !llvm.array<5 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1, %arg0) : (i32, !llvm.ptr, i64) -> i32
    llvm.return
  }
  llvm.func internal @"print_simple_enum::simple_enum::MyEnum"(%arg0: !llvm.struct<packed (i16, array<8 x i8>)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %arg0, %1 : !llvm.struct<packed (i16, array<8 x i8>)>, !llvm.ptr
    %2 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<8 x i8>)>
    %3 = llvm.load %2 : !llvm.ptr -> i16
    %4 = llvm.zext %3 : i16 to i32
    %5 = llvm.mlir.constant(4 : i64) : i64
    %6 = llvm.alloca %5 x i8 : (i64) -> !llvm.ptr
    %7 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %7, %6 : !llvm.array<4 x i8>, !llvm.ptr
    %8 = llvm.mlir.constant(1 : i32) : i32
    %9 = llvm.call @dprintf(%8, %6, %4) : (i32, !llvm.ptr, i32) -> i32
    llvm.switch %3 : i16, ^bb5 [
      0: ^bb1,
      1: ^bb2,
      2: ^bb3,
      3: ^bb4
    ]
  ^bb1:  // pred: ^bb0
    %10 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i8)>
    %11 = llvm.load %10 : !llvm.ptr -> i8
    llvm.call @print_u8(%11) : (i8) -> ()
    llvm.return
  ^bb2:  // pred: ^bb0
    %12 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i16)>
    %13 = llvm.load %12 : !llvm.ptr -> i16
    llvm.call @print_u16(%13) : (i16) -> ()
    llvm.return
  ^bb3:  // pred: ^bb0
    %14 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i32)>
    %15 = llvm.load %14 : !llvm.ptr -> i32
    llvm.call @print_u32(%15) : (i32) -> ()
    llvm.return
  ^bb4:  // pred: ^bb0
    %16 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, i64)>
    %17 = llvm.load %16 : !llvm.ptr -> i64
    llvm.call @print_u64(%17) : (i64) -> ()
    llvm.return
  ^bb5:  // pred: ^bb0
    llvm.return
  }
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"simple_enum::simple_enum::main"() : () -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.call @"print_simple_enum::simple_enum::MyEnum"(%0) : (!llvm.struct<packed (i16, array<8 x i8>)>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"simple_enum::simple_enum::my_enum_a"() -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(4 : i8) : i8
    %1 = llvm.call @"enum_init<simple_enum::simple_enum::MyEnum, 0>"(%0) : (i8) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %1 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func @"_mlir_ciface_simple_enum::simple_enum::my_enum_a"(%arg0: !llvm.ptr<struct<packed (i16, array<8 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"simple_enum::simple_enum::my_enum_a"() : () -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @"simple_enum::simple_enum::my_enum_b"() -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(8 : i16) : i16
    %1 = llvm.call @"enum_init<simple_enum::simple_enum::MyEnum, 1>"(%0) : (i16) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %1 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func @"_mlir_ciface_simple_enum::simple_enum::my_enum_b"(%arg0: !llvm.ptr<struct<packed (i16, array<8 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"simple_enum::simple_enum::my_enum_b"() : () -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @"simple_enum::simple_enum::my_enum_c"() -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(16 : i32) : i32
    %1 = llvm.call @"enum_init<simple_enum::simple_enum::MyEnum, 2>"(%0) : (i32) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %1 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func @"_mlir_ciface_simple_enum::simple_enum::my_enum_c"(%arg0: !llvm.ptr<struct<packed (i16, array<8 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"simple_enum::simple_enum::my_enum_c"() : () -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @"simple_enum::simple_enum::my_enum_d"() -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.call @"enum_init<simple_enum::simple_enum::MyEnum, 3>"(%0) : (i64) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %1 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func @"_mlir_ciface_simple_enum::simple_enum::my_enum_d"(%arg0: !llvm.ptr<struct<packed (i16, array<8 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"simple_enum::simple_enum::my_enum_d"() : () -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @"simple_enum::simple_enum::main"() -> !llvm.struct<packed (i16, array<8 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.call @"enum_init<simple_enum::simple_enum::MyEnum, 3>"(%0) : (i64) -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.return %1 : !llvm.struct<packed (i16, array<8 x i8>)>
  }
  llvm.func @"_mlir_ciface_simple_enum::simple_enum::main"(%arg0: !llvm.ptr<struct<packed (i16, array<8 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"simple_enum::simple_enum::main"() : () -> !llvm.struct<packed (i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<8 x i8>)>>
    llvm.return
  }
}

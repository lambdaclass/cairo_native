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

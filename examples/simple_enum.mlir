module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(%arg0: i8) -> !llvm.struct<(i16, array<2 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %arg0, %4 : i8, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<2 x i8>)>
  }
  llvm.func internal @"store_temp<simple_enum_simple_enum_MyEnum>"(%arg0: !llvm.struct<(i16, array<2 x i8>)>) -> !llvm.struct<(i16, array<2 x i8>)> {
    llvm.return %arg0 : !llvm.struct<(i16, array<2 x i8>)>
  }
  llvm.func internal @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(%arg0: i16) -> !llvm.struct<(i16, array<2 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<2 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<2 x i8>)>
  }
  llvm.func @simple_enum_simple_enum_my_enum() -> !llvm.struct<(i16, array<2 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(4 : i8) : i8
    %1 = llvm.call @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(%0) : (i8) -> !llvm.struct<(i16, array<2 x i8>)>
    %2 = llvm.call @"store_temp<simple_enum_simple_enum_MyEnum>"(%1) : (!llvm.struct<(i16, array<2 x i8>)>) -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.return %2 : !llvm.struct<(i16, array<2 x i8>)>
  }
  llvm.func @_mlir_ciface_simple_enum_simple_enum_my_enum(%arg0: !llvm.ptr<struct<(i16, array<2 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_my_enum() : () -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<2 x i8>)>>
    llvm.return
  }
  llvm.func @simple_enum_simple_enum_my_enum2() -> !llvm.struct<(i16, array<2 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(8 : i16) : i16
    %1 = llvm.call @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(%0) : (i16) -> !llvm.struct<(i16, array<2 x i8>)>
    %2 = llvm.call @"store_temp<simple_enum_simple_enum_MyEnum>"(%1) : (!llvm.struct<(i16, array<2 x i8>)>) -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.return %2 : !llvm.struct<(i16, array<2 x i8>)>
  }
  llvm.func @_mlir_ciface_simple_enum_simple_enum_my_enum2(%arg0: !llvm.ptr<struct<(i16, array<2 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_my_enum2() : () -> !llvm.struct<(i16, array<2 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<2 x i8>)>>
    llvm.return
  }
}

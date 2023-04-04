module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @"enum_init<core::bool, 1>"(%arg0: !llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<0 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<0 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @"enum_init<core::bool, 0>"(%arg0: !llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<0 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<()>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<0 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @"store_temp<core::bool>"(%arg0: !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)> {
    llvm.return %arg0 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @bool_or_impl(%arg0: !llvm.struct<(i16, array<0 x i8>)>, %arg1: !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)> {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %1 = llvm.extractvalue %arg1[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %2 = llvm.or %0, %1  : i16
    %3 = llvm.mlir.undef : !llvm.struct<(i16, array<0 x i8>)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i16, array<0 x i8>)> 
    llvm.return %4 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func internal @print_Unit(%arg0: !llvm.struct<()>) {
    llvm.return
  }
  llvm.func internal @"print_core::bool"(%arg0: !llvm.struct<(i16, array<0 x i8>)>) {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<0 x i8>)> 
    %1 = llvm.zext %0 : i16 to i32
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.alloca %2 x i8 : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %4, %3 : !llvm.array<4 x i8>, !llvm.ptr
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.call @dprintf(%5, %3, %1) : (i32, !llvm.ptr, i32) -> i32
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.array<0 x i8> : (i64) -> !llvm.ptr
    %9 = llvm.extractvalue %arg0[1] : !llvm.struct<(i16, array<0 x i8>)> 
    llvm.store %9, %8 : !llvm.array<0 x i8>, !llvm.ptr
    llvm.switch %0 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    %10 = llvm.load %8 : !llvm.ptr -> !llvm.struct<()>
    llvm.call @print_Unit(%10) : (!llvm.struct<()>) -> ()
    llvm.return
  ^bb2:  // pred: ^bb0
    %11 = llvm.load %8 : !llvm.ptr -> !llvm.struct<()>
    llvm.call @print_Unit(%11) : (!llvm.struct<()>) -> ()
    llvm.return
  ^bb3:  // pred: ^bb0
    llvm.return
  }
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"boolean::boolean::main"() : () -> !llvm.struct<(i16, array<0 x i8>)>
    llvm.call @"print_core::bool"(%0) : (!llvm.struct<(i16, array<0 x i8>)>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"boolean::boolean::main"() -> !llvm.struct<(i16, array<0 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %1 = llvm.call @"enum_init<core::bool, 1>"(%0) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)>
    %2 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %3 = llvm.call @"enum_init<core::bool, 0>"(%2) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<0 x i8>)>
    %4 = llvm.call @"store_temp<core::bool>"(%1) : (!llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %5 = llvm.call @"store_temp<core::bool>"(%3) : (!llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %6 = llvm.call @bool_or_impl(%4, %5) : (!llvm.struct<(i16, array<0 x i8>)>, !llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    %7 = llvm.call @"store_temp<core::bool>"(%6) : (!llvm.struct<(i16, array<0 x i8>)>) -> !llvm.struct<(i16, array<0 x i8>)>
    llvm.return %7 : !llvm.struct<(i16, array<0 x i8>)>
  }
  llvm.func @"_mlir_ciface_boolean::boolean::main"(%arg0: !llvm.ptr<struct<(i16, array<0 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @"boolean::boolean::main"() : () -> !llvm.struct<(i16, array<0 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<0 x i8>)>>
    llvm.return
  }
}

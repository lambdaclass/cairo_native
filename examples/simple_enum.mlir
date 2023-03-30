module attributes {llvm.data_layout = ""} {
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(%arg0: i8) -> !llvm.struct<(i16, array<8 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %arg0, %4 : i8, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func internal @"store_temp<simple_enum_simple_enum_MyEnum>"(%arg0: !llvm.struct<(i16, array<8 x i8>)>) -> !llvm.struct<(i16, array<8 x i8>)> {
    llvm.return %arg0 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(%arg0: i16) -> !llvm.struct<(i16, array<8 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<simple_enum_simple_enum_MyEnum, 2>"(%arg0: i32) -> !llvm.struct<(i16, array<8 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(2 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func internal @"enum_init<simple_enum_simple_enum_MyEnum, 3>"(%arg0: i64) -> !llvm.struct<(i16, array<8 x i8>)> {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(3 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %arg0, %4 : i64, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func internal @print_u8(%arg0: i8) {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %2, %1 : !llvm.array<4 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1, %arg0) : (i32, !llvm.ptr, i8) -> i32
    llvm.return
  }
  llvm.func internal @print_u16(%arg0: i16) {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %2, %1 : !llvm.array<4 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1, %arg0) : (i32, !llvm.ptr, i16) -> i32
    llvm.return
  }
  llvm.func internal @print_u32(%arg0: i32) {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %2, %1 : !llvm.array<4 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1, %arg0) : (i32, !llvm.ptr, i32) -> i32
    llvm.return
  }
  llvm.func internal @print_u64(%arg0: i64) {
    %0 = llvm.mlir.constant(4 : i64) : i64
    %1 = llvm.alloca %0 x i8 : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %2, %1 : !llvm.array<4 x i8>, !llvm.ptr
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.call @dprintf(%3, %1, %arg0) : (i32, !llvm.ptr, i64) -> i32
    llvm.return
  }
  llvm.func internal @"print_simple_enum::simple_enum::MyEnum"(%arg0: !llvm.struct<(i16, array<8 x i8>)>) {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<8 x i8>)> 
    %1 = llvm.mlir.constant(4 : i64) : i64
    %2 = llvm.alloca %1 x i8 : (i64) -> !llvm.ptr
    %3 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %3, %2 : !llvm.array<4 x i8>, !llvm.ptr
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.call @dprintf(%4, %2, %0) : (i32, !llvm.ptr, i16) -> i32
    %6 = llvm.mlir.constant(1 : i64) : i64
    %7 = llvm.alloca %6 x !llvm.struct<(i16, array<8 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %arg0, %7 : !llvm.struct<(i16, array<8 x i8>)>, !llvm.ptr
    %8 = llvm.getelementptr inbounds %7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<8 x i8>)>
    llvm.switch %0 : i16, ^bb5 [
      0: ^bb1,
      1: ^bb2,
      2: ^bb3,
      3: ^bb4
    ]
  ^bb1:  // pred: ^bb0
    %9 = llvm.load %8 : !llvm.ptr -> i8
    llvm.call @print_u8(%9) : (i8) -> ()
    llvm.return
  ^bb2:  // pred: ^bb0
    %10 = llvm.load %8 : !llvm.ptr -> i16
    llvm.call @print_u16(%10) : (i16) -> ()
    llvm.return
  ^bb3:  // pred: ^bb0
    %11 = llvm.load %8 : !llvm.ptr -> i32
    llvm.call @print_u32(%11) : (i32) -> ()
    llvm.return
  ^bb4:  // pred: ^bb0
    %12 = llvm.load %8 : !llvm.ptr -> i64
    llvm.call @print_u64(%12) : (i64) -> ()
    llvm.return
  ^bb5:  // pred: ^bb0
    llvm.return
  }
  llvm.func @main() attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_main() : () -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.call @"print_simple_enum::simple_enum::MyEnum"(%0) : (!llvm.struct<(i16, array<8 x i8>)>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @simple_enum_simple_enum_my_enum_a() -> !llvm.struct<(i16, array<8 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(4 : i8) : i8
    %1 = llvm.call @"enum_init<simple_enum_simple_enum_MyEnum, 0>"(%0) : (i8) -> !llvm.struct<(i16, array<8 x i8>)>
    %2 = llvm.call @"store_temp<simple_enum_simple_enum_MyEnum>"(%1) : (!llvm.struct<(i16, array<8 x i8>)>) -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %2 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func @_mlir_ciface_simple_enum_simple_enum_my_enum_a(%arg0: !llvm.ptr<struct<(i16, array<8 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_my_enum_a() : () -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @simple_enum_simple_enum_my_enum_b() -> !llvm.struct<(i16, array<8 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(8 : i16) : i16
    %1 = llvm.call @"enum_init<simple_enum_simple_enum_MyEnum, 1>"(%0) : (i16) -> !llvm.struct<(i16, array<8 x i8>)>
    %2 = llvm.call @"store_temp<simple_enum_simple_enum_MyEnum>"(%1) : (!llvm.struct<(i16, array<8 x i8>)>) -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %2 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func @_mlir_ciface_simple_enum_simple_enum_my_enum_b(%arg0: !llvm.ptr<struct<(i16, array<8 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_my_enum_b() : () -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @simple_enum_simple_enum_my_enum_c() -> !llvm.struct<(i16, array<8 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(16 : i32) : i32
    %1 = llvm.call @"enum_init<simple_enum_simple_enum_MyEnum, 2>"(%0) : (i32) -> !llvm.struct<(i16, array<8 x i8>)>
    %2 = llvm.call @"store_temp<simple_enum_simple_enum_MyEnum>"(%1) : (!llvm.struct<(i16, array<8 x i8>)>) -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %2 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func @_mlir_ciface_simple_enum_simple_enum_my_enum_c(%arg0: !llvm.ptr<struct<(i16, array<8 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_my_enum_c() : () -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @simple_enum_simple_enum_my_enum_d() -> !llvm.struct<(i16, array<8 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.call @"enum_init<simple_enum_simple_enum_MyEnum, 3>"(%0) : (i64) -> !llvm.struct<(i16, array<8 x i8>)>
    %2 = llvm.call @"store_temp<simple_enum_simple_enum_MyEnum>"(%1) : (!llvm.struct<(i16, array<8 x i8>)>) -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %2 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func @_mlir_ciface_simple_enum_simple_enum_my_enum_d(%arg0: !llvm.ptr<struct<(i16, array<8 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_my_enum_d() : () -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<8 x i8>)>>
    llvm.return
  }
  llvm.func @simple_enum_simple_enum_main() -> !llvm.struct<(i16, array<8 x i8>)> attributes {llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(16 : i64) : i64
    %1 = llvm.call @"enum_init<simple_enum_simple_enum_MyEnum, 3>"(%0) : (i64) -> !llvm.struct<(i16, array<8 x i8>)>
    %2 = llvm.call @"store_temp<simple_enum_simple_enum_MyEnum>"(%1) : (!llvm.struct<(i16, array<8 x i8>)>) -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.return %2 : !llvm.struct<(i16, array<8 x i8>)>
  }
  llvm.func @_mlir_ciface_simple_enum_simple_enum_main(%arg0: !llvm.ptr<struct<(i16, array<8 x i8>)>>) attributes {llvm.emit_c_interface} {
    %0 = llvm.call @simple_enum_simple_enum_main() : () -> !llvm.struct<(i16, array<8 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<8 x i8>)>>
    llvm.return
  }
}

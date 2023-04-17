module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"enum_init<enum_match::enum_match::MyEnum, 1>"(%arg0: i16) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
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
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"enum_match::enum_match::main"() : () -> i16
    llvm.call @print_u16(%0) : (i16) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"enum_match::enum_match::get_my_enum_b"(%arg0: !llvm.struct<(i16, array<4 x i8>)>) -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb5(%arg0 : !llvm.struct<(i16, array<4 x i8>)>)
  ^bb1:  // pred: ^bb5
    llvm.br ^bb6
  ^bb2:  // pred: ^bb5
    %0 = llvm.load %5 : !llvm.ptr -> i16
    llvm.br ^bb7(%0 : i16)
  ^bb3:  // pred: ^bb5
    llvm.br ^bb8
  ^bb4:  // pred: ^bb5
    llvm.unreachable
  ^bb5(%1: !llvm.struct<(i16, array<4 x i8>)>):  // pred: ^bb0
    %2 = llvm.extractvalue %1[0] : !llvm.struct<(i16, array<4 x i8>)> 
    %3 = llvm.extractvalue %1[1] : !llvm.struct<(i16, array<4 x i8>)> 
    %4 = llvm.mlir.constant(1 : i64) : i64
    %5 = llvm.alloca %4 x !llvm.array<4 x i8> : (i64) -> !llvm.ptr
    llvm.store %3, %5 : !llvm.array<4 x i8>, !llvm.ptr
    llvm.switch %2 : i16, ^bb4 [
      0: ^bb1,
      1: ^bb2,
      2: ^bb3
    ]
  ^bb6:  // pred: ^bb1
    %6 = llvm.mlir.constant(1 : i16) : i16
    llvm.br ^bb9(%6 : i16)
  ^bb7(%7: i16):  // pred: ^bb2
    llvm.br ^bb9(%7 : i16)
  ^bb8:  // pred: ^bb3
    %8 = llvm.mlir.constant(0 : i16) : i16
    llvm.br ^bb9(%8 : i16)
  ^bb9(%9: i16):  // 3 preds: ^bb6, ^bb7, ^bb8
    llvm.return %9 : i16
  }
  llvm.func @"_mlir_ciface_enum_match::enum_match::get_my_enum_b"(%arg0: !llvm.struct<(i16, array<4 x i8>)>) -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"enum_match::enum_match::get_my_enum_b"(%arg0) : (!llvm.struct<(i16, array<4 x i8>)>) -> i16
    llvm.return %0 : i16
  }
  llvm.func @"enum_match::enum_match::main"() -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %0 = llvm.mlir.constant(16 : i16) : i16
    %1 = llvm.call @"enum_init<enum_match::enum_match::MyEnum, 1>"(%0) : (i16) -> !llvm.struct<(i16, array<4 x i8>)>
    %2 = llvm.call @"enum_match::enum_match::get_my_enum_b"(%1) : (!llvm.struct<(i16, array<4 x i8>)>) -> i16
    llvm.return %2 : i16
  }
  llvm.func @"_mlir_ciface_enum_match::enum_match::main"() -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"enum_match::enum_match::main"() : () -> i16
    llvm.return %0 : i16
  }
}

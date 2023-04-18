module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"enum_init<enum_match::enum_match::MyEnum, 1>"(%arg0: i16) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i16) : i16
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %0, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : i16, !llvm.ptr
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
  }
  llvm.func @"enum_match::enum_match::get_my_enum_b"(%arg0: !llvm.struct<(i16, array<4 x i8>)>) -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(1 : i16) : i16
    %2 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb4(%arg0 : !llvm.struct<(i16, array<4 x i8>)>)
  ^bb1(%3: i16):  // 2 preds: ^bb4, ^bb4
    llvm.br ^bb5(%3 : i16)
  ^bb2:  // pred: ^bb4
    %4 = llvm.load %8 : !llvm.ptr -> i16
    llvm.br ^bb5(%4 : i16)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%5: !llvm.struct<(i16, array<4 x i8>)>):  // pred: ^bb0
    %6 = llvm.extractvalue %5[0] : !llvm.struct<(i16, array<4 x i8>)> 
    %7 = llvm.extractvalue %5[1] : !llvm.struct<(i16, array<4 x i8>)> 
    %8 = llvm.alloca %2 x !llvm.array<4 x i8> : (i64) -> !llvm.ptr
    llvm.store %7, %8 : !llvm.array<4 x i8>, !llvm.ptr
    llvm.switch %6 : i16, ^bb3 [
      0: ^bb1(%1 : i16),
      1: ^bb2,
      2: ^bb1(%0 : i16)
    ]
  ^bb5(%9: i16):  // 2 preds: ^bb1, ^bb2
    llvm.br ^bb6(%9 : i16)
  ^bb6(%10: i16):  // pred: ^bb5
    llvm.return %10 : i16
  }
  llvm.func @"_mlir_ciface_enum_match::enum_match::get_my_enum_b"(%arg0: !llvm.struct<(i16, array<4 x i8>)>) -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"enum_match::enum_match::get_my_enum_b"(%arg0) : (!llvm.struct<(i16, array<4 x i8>)>) -> i16
    llvm.return %0 : i16
  }
  llvm.func @"enum_match::enum_match::main"() -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(16 : i16) : i16
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %1 = llvm.call @"enum_init<enum_match::enum_match::MyEnum, 1>"(%0) : (i16) -> !llvm.struct<(i16, array<4 x i8>)>
    %2 = llvm.call @"enum_match::enum_match::get_my_enum_b"(%1) : (!llvm.struct<(i16, array<4 x i8>)>) -> i16
    llvm.return %2 : i16
  }
  llvm.func @"_mlir_ciface_enum_match::enum_match::main"() -> i16 attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(16 : i16) : i16
    llvm.br ^bb1
  ^bb1:  // pred: ^bb0
    %1 = llvm.call @"enum_init<enum_match::enum_match::MyEnum, 1>"(%0) : (i16) -> !llvm.struct<(i16, array<4 x i8>)>
    %2 = llvm.call @"enum_match::enum_match::get_my_enum_b"(%1) : (!llvm.struct<(i16, array<4 x i8>)>) -> i16
    llvm.br ^bb2(%2 : i16)
  ^bb2(%3: i16):  // pred: ^bb1
    llvm.return %3 : i16
  }
}

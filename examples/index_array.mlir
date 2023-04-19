module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"array_new<u8>"() -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %5 = llvm.mlir.constant(8 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"array_append<u8>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i8) -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<packed (i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<packed (i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    llvm.store %arg1, %12 : i8, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"struct_deconstruct<Tuple<u8>>"(%arg0: !llvm.struct<packed (i8)>) -> i8 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i8)> 
    llvm.return %0 : i8
  }
  llvm.func internal @"struct_construct<Tuple<u8>>"(%arg0: i8) -> !llvm.struct<packed (i8)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i8)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i8)> 
    llvm.return %1 : !llvm.struct<packed (i8)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u8)>, 0>"(%arg0: !llvm.struct<packed (i8)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i8)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i8)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::integer::u8)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"struct_deconstruct<Tuple<Box<u8>>>"(%arg0: !llvm.struct<packed (i8)>) -> i8 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i8)> 
    llvm.return %0 : i8
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u8)>, 0>"(%arg0: !llvm.struct<packed (i8)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i8)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i8)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u8)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"struct_construct<Tuple<Box<u8>>>"(%arg0: i8) -> !llvm.struct<packed (i8)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i8)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<packed (i8)> 
    llvm.return %1 : !llvm.struct<packed (i8)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 0>"(%arg0: !llvm.struct<packed (i8)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i8)>)>
    llvm.store %arg0, %4 : !llvm.struct<packed (i8)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func internal @"array_new<felt252>"() -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<packed (i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"array_append<felt252>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i256) -> !llvm.struct<packed (i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<packed (i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<packed (i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<packed (i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    llvm.store %arg1, %12 : i256, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<packed (i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<packed (i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 1>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func @"index_array::index_array::main"() -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4
  ^bb1:  // pred: ^bb4
    %0 = llvm.getelementptr inbounds %26[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i8)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i8)>
    llvm.br ^bb5(%1 : !llvm.struct<packed (i8)>)
  ^bb2:  // pred: ^bb4
    %2 = llvm.getelementptr inbounds %26[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb6(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4:  // pred: ^bb0
    %4 = llvm.call @"array_new<u8>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %5 = llvm.mlir.constant(4 : i8) : i8
    %6 = llvm.call @"array_append<u8>"(%4, %5) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %7 = llvm.mlir.constant(5 : i8) : i8
    %8 = llvm.call @"array_append<u8>"(%6, %7) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %9 = llvm.mlir.constant(4 : i8) : i8
    %10 = llvm.call @"array_append<u8>"(%8, %9) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %11 = llvm.mlir.constant(4 : i8) : i8
    %12 = llvm.call @"array_append<u8>"(%10, %11) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %13 = llvm.mlir.constant(4 : i8) : i8
    %14 = llvm.call @"array_append<u8>"(%12, %13) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %15 = llvm.mlir.constant(1 : i8) : i8
    %16 = llvm.call @"array_append<u8>"(%14, %15) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %17 = llvm.mlir.constant(1 : i8) : i8
    %18 = llvm.call @"array_append<u8>"(%16, %17) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %19 = llvm.mlir.constant(1 : i8) : i8
    %20 = llvm.call @"array_append<u8>"(%18, %19) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %21 = llvm.mlir.constant(2 : i8) : i8
    %22 = llvm.call @"array_append<u8>"(%20, %21) : (!llvm.struct<packed (i32, i32, ptr)>, i8) -> !llvm.struct<packed (i32, i32, ptr)>
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.call @"core::array::ArrayIndex::<core::integer::u8>::index"(%22, %23) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %25 = llvm.mlir.constant(1 : i64) : i64
    %26 = llvm.alloca %25 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %24, %26 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %27 = llvm.getelementptr inbounds %26[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %28 = llvm.load %27 : !llvm.ptr -> i16
    llvm.switch %28 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%29: !llvm.struct<packed (i8)>):  // pred: ^bb1
    %30 = llvm.call @"struct_deconstruct<Tuple<u8>>"(%29) : (!llvm.struct<packed (i8)>) -> i8
    %31 = llvm.call @"struct_construct<Tuple<u8>>"(%30) : (i8) -> !llvm.struct<packed (i8)>
    %32 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u8)>, 0>"(%31) : (!llvm.struct<packed (i8)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %32 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb6(%33: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %34 = llvm.call @"enum_init<core::PanicResult::<(core::integer::u8)>, 1>"(%33) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %34 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_index_array::index_array::main"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"index_array::index_array::main"() : () -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::ArrayIndex::<core::integer::u8>::index"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<packed (i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb4
    %0 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i8)>)>
    %1 = llvm.load %0 : !llvm.ptr -> !llvm.struct<packed (i8)>
    llvm.br ^bb5(%1 : !llvm.struct<packed (i8)>)
  ^bb2:  // pred: ^bb4
    %2 = llvm.getelementptr inbounds %8[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, struct<packed (i32, i32, ptr)>)>
    %3 = llvm.load %2 : !llvm.ptr -> !llvm.struct<packed (i32, i32, ptr)>
    llvm.br ^bb6(%3 : !llvm.struct<packed (i32, i32, ptr)>)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%4: !llvm.struct<packed (i32, i32, ptr)>, %5: i32):  // pred: ^bb0
    %6 = llvm.call @"core::array::array_at::<core::integer::u8>"(%4, %5) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.struct<packed (i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    llvm.store %6, %8 : !llvm.struct<packed (i16, array<16 x i8>)>, !llvm.ptr
    %9 = llvm.getelementptr inbounds %8[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<packed (i16, array<16 x i8>)>
    %10 = llvm.load %9 : !llvm.ptr -> i16
    llvm.switch %10 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%11: !llvm.struct<packed (i8)>):  // pred: ^bb1
    %12 = llvm.call @"struct_deconstruct<Tuple<Box<u8>>>"(%11) : (!llvm.struct<packed (i8)>) -> i8
    %13 = llvm.call @"struct_construct<Tuple<u8>>"(%12) : (i8) -> !llvm.struct<packed (i8)>
    %14 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u8)>, 0>"(%13) : (!llvm.struct<packed (i8)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %14 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb6(%15: !llvm.struct<packed (i32, i32, ptr)>):  // pred: ^bb2
    %16 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u8)>, 1>"(%15) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %16 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u8>::index"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::ArrayIndex::<core::integer::u8>::index"(%arg1, %arg2) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::array_at::<core::integer::u8>"(%arg0: !llvm.struct<packed (i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<packed (i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb2(%arg0, %arg1 : !llvm.struct<packed (i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb2
    %0 = llvm.extractvalue %3[2] : !llvm.struct<packed (i32, i32, ptr)> 
    %1 = llvm.getelementptr %0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, i8
    %2 = llvm.load %1 : !llvm.ptr -> i8
    llvm.br ^bb3(%2 : i8)
  ^bb2(%3: !llvm.struct<packed (i32, i32, ptr)>, %4: i32):  // pred: ^bb0
    %5 = llvm.extractvalue %3[0] : !llvm.struct<packed (i32, i32, ptr)> 
    %6 = llvm.icmp "ult" %4, %5 : i32
    llvm.cond_br %6, ^bb1, ^bb4
  ^bb3(%7: i8):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<Box<u8>>>"(%7) : (i8) -> !llvm.struct<packed (i8)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 0>"(%8) : (!llvm.struct<packed (i8)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %9 : !llvm.struct<packed (i16, array<16 x i8>)>
  ^bb4:  // pred: ^bb2
    %10 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<packed (i32, i32, ptr)>
    %11 = llvm.mlir.constant(1637570914057682275393755530660268060279989363 : i256) : i256
    %12 = llvm.call @"array_append<felt252>"(%10, %11) : (!llvm.struct<packed (i32, i32, ptr)>, i256) -> !llvm.struct<packed (i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u8>)>, 1>"(%12) : (!llvm.struct<packed (i32, i32, ptr)>) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<packed (i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::array_at::<core::integer::u8>"(%arg0: !llvm.ptr<struct<packed (i16, array<16 x i8>)>>, %arg1: !llvm.struct<packed (i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::array_at::<core::integer::u8>"(%arg1, %arg2) : (!llvm.struct<packed (i32, i32, ptr)>, i32) -> !llvm.struct<packed (i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<packed (i16, array<16 x i8>)>>
    llvm.return
  }
}

module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @memmove(!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"array_new<u32>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(32 : i64) : i64
    %1 = llvm.mlir.constant(8 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i32, i32, ptr)> 
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %5[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %3 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.icmp "ult" %2, %3 : i32
    llvm.cond_br %4, ^bb2(%arg0 : !llvm.struct<(i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %5 = llvm.mul %3, %1  : i32
    %6 = llvm.zext %5 : i32 to i64
    %7 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %8 = llvm.call @realloc(%7, %6) : (!llvm.ptr, i64) -> !llvm.ptr
    %9 = llvm.insertvalue %8, %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %10 = llvm.insertvalue %5, %9[1] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb2(%10 : !llvm.struct<(i32, i32, ptr)>)
  ^bb2(%11: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %12 = llvm.extractvalue %11[2] : !llvm.struct<(i32, i32, ptr)> 
    %13 = llvm.getelementptr %12[%2] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %arg1, %13 : i32, !llvm.ptr
    %14 = llvm.add %2, %0  : i32
    %15 = llvm.insertvalue %14, %11[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%arg0: i32) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %0, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : i32, !llvm.ptr
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
  }
  llvm.func internal @"struct_construct<Unit>"() -> !llvm.struct<()> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<()>
    llvm.return %0 : !llvm.struct<()>
  }
  llvm.func internal @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%arg0: !llvm.struct<()>) -> !llvm.struct<(i16, array<4 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i16) : i16
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i16, array<4 x i8>)> : (i64) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %0, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<4 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<()>, !llvm.ptr
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<4 x i8>)>
  }
  llvm.func internal @"array_len<u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_deconstruct<Tuple<u32>>"(%arg0: !llvm.struct<(i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> !llvm.struct<(i32, i32, i32, i32, i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32, i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %2 = llvm.insertvalue %arg1, %1[1] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %3 = llvm.insertvalue %arg2, %2[2] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %4 = llvm.insertvalue %arg3, %3[3] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    %5 = llvm.insertvalue %arg4, %4[4] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.return %5 : !llvm.struct<(i32, i32, i32, i32, i32)>
  }
  llvm.func internal @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(%arg0: !llvm.struct<(i32, i32, i32, i32, i32)>) -> !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)> 
    llvm.return %1 : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(%arg0: !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i16, array<20 x i8>)> : (i64) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %0, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>, !llvm.ptr
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<20 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"struct_deconstruct<Tuple<Box<u32>>>"(%arg0: !llvm.struct<(i32)>) -> i32 attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32)> 
    llvm.return %0 : i32
  }
  llvm.func internal @"struct_construct<Tuple<u32>>"(%arg0: i32) -> !llvm.struct<(i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32)> 
    llvm.return %1 : !llvm.struct<(i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%arg0: !llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32)>, !llvm.ptr
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func internal @"struct_construct<Tuple<Box<u32>>>"(%arg0: i32) -> !llvm.struct<(i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32)> 
    llvm.return %1 : !llvm.struct<(i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%arg0: !llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(0 : i16) : i16
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.alloca %1 x !llvm.struct<(i16, array<16 x i8>)> : (i64) -> !llvm.ptr
    %3 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %2[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32)>, !llvm.ptr
    %5 = llvm.load %2 : !llvm.ptr -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func internal @"array_new<felt252>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(256 : i64) : i64
    %1 = llvm.mlir.constant(8 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %4 = llvm.insertvalue %2, %3[0] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.insertvalue %1, %4[1] : !llvm.struct<(i32, i32, ptr)> 
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %0) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %5[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<felt252>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i256) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %3 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.icmp "ult" %2, %3 : i32
    llvm.cond_br %4, ^bb2(%arg0 : !llvm.struct<(i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %5 = llvm.mul %3, %1  : i32
    %6 = llvm.zext %5 : i32 to i64
    %7 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %8 = llvm.call @realloc(%7, %6) : (!llvm.ptr, i64) -> !llvm.ptr
    %9 = llvm.insertvalue %8, %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %10 = llvm.insertvalue %5, %9[1] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb2(%10 : !llvm.struct<(i32, i32, ptr)>)
  ^bb2(%11: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %12 = llvm.extractvalue %11[2] : !llvm.struct<(i32, i32, ptr)> 
    %13 = llvm.getelementptr %12[%2] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    llvm.store %arg1, %13 : i256, !llvm.ptr
    %14 = llvm.add %2, %0  : i32
    %15 = llvm.insertvalue %14, %11[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    llvm.call_intrinsic "llvm.trap"() : () -> ()
    llvm.unreachable
  }
  llvm.func @"example_array::example_array::main"() -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(4 : i32) : i32
    %1 = llvm.mlir.constant(1 : i64) : i64
    %2 = llvm.mlir.constant(5 : i32) : i32
    %3 = llvm.mlir.constant(7 : i32) : i32
    %4 = llvm.mlir.constant(0 : i32) : i32
    %5 = llvm.mlir.constant(3 : i32) : i32
    %6 = llvm.mlir.constant(2 : i32) : i32
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.mlir.constant(4 : i64) : i64
    llvm.br ^bb9
  ^bb1:  // pred: ^bb9
    %9 = llvm.extractvalue %28[2] : !llvm.struct<(i32, i32, ptr)> 
    %10 = llvm.load %9 : !llvm.ptr -> i32
    %11 = llvm.sub %29, %7  : i32
    %12 = llvm.getelementptr %9[1] : (!llvm.ptr) -> !llvm.ptr, i32
    %13 = llvm.zext %11 : i32 to i64
    %14 = llvm.mul %13, %8  : i64
    %15 = llvm.call @memmove(%9, %12, %14) : (!llvm.ptr, !llvm.ptr, i64) -> !llvm.ptr
    %16 = llvm.insertvalue %11, %28[0] : !llvm.struct<(i32, i32, ptr)> 
    %17 = llvm.insertvalue %15, %16[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb10(%17, %10 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb2:  // pred: ^bb12
    %18 = llvm.load %45 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb13(%41, %18 : !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb3(%19: !llvm.ptr):  // 5 preds: ^bb12, ^bb13, ^bb14, ^bb15, ^bb16
    %20 = llvm.load %19 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb18(%20 : !llvm.struct<(i32, i32, ptr)>)
  ^bb4:  // 5 preds: ^bb12, ^bb13, ^bb14, ^bb15, ^bb16
    llvm.unreachable
  ^bb5:  // pred: ^bb13
    %21 = llvm.load %52 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb14(%48, %46, %21 : i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb6:  // pred: ^bb14
    %22 = llvm.load %60 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb15(%53, %56, %54, %22 : i32, i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb7:  // pred: ^bb15
    %23 = llvm.load %69 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb16(%61, %62, %65, %63, %23 : i32, i32, i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb8:  // pred: ^bb16
    %24 = llvm.load %79 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb17(%70, %71, %72, %75, %24 : i32, i32, i32, i32, !llvm.struct<(i32)>)
  ^bb9:  // pred: ^bb0
    %25 = llvm.call @"array_new<u32>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %26 = llvm.call @"array_append<u32>"(%25, %7) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %27 = llvm.call @"array_append<u32>"(%26, %6) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %28 = llvm.call @"array_append<u32>"(%27, %5) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %29 = llvm.extractvalue %28[0] : !llvm.struct<(i32, i32, ptr)> 
    %30 = llvm.icmp "uge" %29, %7 : i32
    llvm.cond_br %30, ^bb1, ^bb11(%28 : !llvm.struct<(i32, i32, ptr)>)
  ^bb10(%31: !llvm.struct<(i32, i32, ptr)>, %32: i32):  // pred: ^bb1
    %33 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 0>"(%32) : (i32) -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.br ^bb12(%31 : !llvm.struct<(i32, i32, ptr)>)
  ^bb11(%34: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb9
    %35 = llvm.call @"struct_construct<Unit>"() : () -> !llvm.struct<()>
    %36 = llvm.call @"enum_init<core::option::Option::<core::integer::u32>, 1>"(%35) : (!llvm.struct<()>) -> !llvm.struct<(i16, array<4 x i8>)>
    llvm.br ^bb12(%34 : !llvm.struct<(i32, i32, ptr)>)
  ^bb12(%37: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb10, ^bb11
    %38 = llvm.call @"array_append<u32>"(%37, %3) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %39 = llvm.call @"array_append<u32>"(%38, %2) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %40 = llvm.call @"array_len<u32>"(%39) : (!llvm.struct<(i32, i32, ptr)>) -> i32
    %41 = llvm.call @"array_append<u32>"(%39, %40) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %42 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%41, %4) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %43 = llvm.extractvalue %42[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %44 = llvm.extractvalue %42[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %45 = llvm.alloca %1 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %44, %45 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %43 : i16, ^bb4 [
      0: ^bb2,
      1: ^bb3(%45 : !llvm.ptr)
    ]
  ^bb13(%46: !llvm.struct<(i32, i32, ptr)>, %47: !llvm.struct<(i32)>):  // pred: ^bb2
    %48 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%47) : (!llvm.struct<(i32)>) -> i32
    %49 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%46, %7) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %50 = llvm.extractvalue %49[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %51 = llvm.extractvalue %49[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %52 = llvm.alloca %1 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %51, %52 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %50 : i16, ^bb4 [
      0: ^bb5,
      1: ^bb3(%52 : !llvm.ptr)
    ]
  ^bb14(%53: i32, %54: !llvm.struct<(i32, i32, ptr)>, %55: !llvm.struct<(i32)>):  // pred: ^bb5
    %56 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%55) : (!llvm.struct<(i32)>) -> i32
    %57 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%54, %6) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %58 = llvm.extractvalue %57[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %59 = llvm.extractvalue %57[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %60 = llvm.alloca %1 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %59, %60 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %58 : i16, ^bb4 [
      0: ^bb6,
      1: ^bb3(%60 : !llvm.ptr)
    ]
  ^bb15(%61: i32, %62: i32, %63: !llvm.struct<(i32, i32, ptr)>, %64: !llvm.struct<(i32)>):  // pred: ^bb6
    %65 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%64) : (!llvm.struct<(i32)>) -> i32
    %66 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%63, %5) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %67 = llvm.extractvalue %66[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %68 = llvm.extractvalue %66[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %69 = llvm.alloca %1 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %68, %69 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %67 : i16, ^bb4 [
      0: ^bb7,
      1: ^bb3(%69 : !llvm.ptr)
    ]
  ^bb16(%70: i32, %71: i32, %72: i32, %73: !llvm.struct<(i32, i32, ptr)>, %74: !llvm.struct<(i32)>):  // pred: ^bb7
    %75 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%74) : (!llvm.struct<(i32)>) -> i32
    %76 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%73, %0) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    %77 = llvm.extractvalue %76[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %78 = llvm.extractvalue %76[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %79 = llvm.alloca %1 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %78, %79 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %77 : i16, ^bb4 [
      0: ^bb8,
      1: ^bb3(%79 : !llvm.ptr)
    ]
  ^bb17(%80: i32, %81: i32, %82: i32, %83: i32, %84: !llvm.struct<(i32)>):  // pred: ^bb8
    %85 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%84) : (!llvm.struct<(i32)>) -> i32
    %86 = llvm.call @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(%80, %81, %82, %83, %85) : (i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32, i32)>
    %87 = llvm.call @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(%86) : (!llvm.struct<(i32, i32, i32, i32, i32)>) -> !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
    %88 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(%87) : (!llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %88 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb18(%89: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb3
    %90 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%89) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %90 : !llvm.struct<(i16, array<20 x i8>)>
  }
  llvm.func @"_mlir_ciface_example_array::example_array::main"(%arg0: !llvm.ptr<struct<(i16, array<20 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<20 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1637570914057682275393755530660268060279989363 : i256) : i256
    %1 = llvm.mlir.constant(1 : i64) : i64
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb9
    %2 = llvm.load %22 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb10(%2 : !llvm.struct<(i32)>)
  ^bb2:  // pred: ^bb9
    %3 = llvm.load %22 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb11(%3 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb9
    llvm.unreachable
  ^bb4(%4: !llvm.struct<(i32, i32, ptr)>, %5: i32):  // pred: ^bb0
    llvm.br ^bb6(%4, %5 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb5:  // pred: ^bb6
    %6 = llvm.extractvalue %9[2] : !llvm.struct<(i32, i32, ptr)> 
    %7 = llvm.getelementptr %6[%10] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %8 = llvm.load %7 : !llvm.ptr -> i32
    llvm.br ^bb7(%8 : i32)
  ^bb6(%9: !llvm.struct<(i32, i32, ptr)>, %10: i32):  // pred: ^bb4
    %11 = llvm.extractvalue %9[0] : !llvm.struct<(i32, i32, ptr)> 
    %12 = llvm.icmp "ult" %10, %11 : i32
    llvm.cond_br %12, ^bb5, ^bb8
  ^bb7(%13: i32):  // pred: ^bb5
    %14 = llvm.call @"struct_construct<Tuple<Box<u32>>>"(%13) : (i32) -> !llvm.struct<(i32)>
    %15 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%14) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.br ^bb9(%15 : !llvm.struct<(i16, array<16 x i8>)>)
  ^bb8:  // pred: ^bb6
    %16 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %17 = llvm.call @"array_append<felt252>"(%16, %0) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %18 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%17) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.br ^bb9(%18 : !llvm.struct<(i16, array<16 x i8>)>)
  ^bb9(%19: !llvm.struct<(i16, array<16 x i8>)>):  // 2 preds: ^bb7, ^bb8
    %20 = llvm.extractvalue %19[0] : !llvm.struct<(i16, array<16 x i8>)> 
    %21 = llvm.extractvalue %19[1] : !llvm.struct<(i16, array<16 x i8>)> 
    %22 = llvm.alloca %1 x !llvm.array<16 x i8> : (i64) -> !llvm.ptr
    llvm.store %21, %22 : !llvm.array<16 x i8>, !llvm.ptr
    llvm.switch %20 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb10(%23: !llvm.struct<(i32)>):  // pred: ^bb1
    %24 = llvm.call @"struct_deconstruct<Tuple<Box<u32>>>"(%23) : (!llvm.struct<(i32)>) -> i32
    %25 = llvm.call @"struct_construct<Tuple<u32>>"(%24) : (i32) -> !llvm.struct<(i32)>
    %26 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%25) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %26 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb11(%27: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %28 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%27) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %28 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg1, %arg2) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::array_at::<core::integer::u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i16, array<16 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1637570914057682275393755530660268060279989363 : i256) : i256
    llvm.br ^bb2(%arg0, %arg1 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb2
    %1 = llvm.extractvalue %4[2] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.getelementptr %1[%5] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %3 = llvm.load %2 : !llvm.ptr -> i32
    llvm.br ^bb3(%3 : i32)
  ^bb2(%4: !llvm.struct<(i32, i32, ptr)>, %5: i32):  // pred: ^bb0
    %6 = llvm.extractvalue %4[0] : !llvm.struct<(i32, i32, ptr)> 
    %7 = llvm.icmp "ult" %5, %6 : i32
    llvm.cond_br %7, ^bb1, ^bb4
  ^bb3(%8: i32):  // pred: ^bb1
    %9 = llvm.call @"struct_construct<Tuple<Box<u32>>>"(%8) : (i32) -> !llvm.struct<(i32)>
    %10 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%9) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %10 : !llvm.struct<(i16, array<16 x i8>)>
  ^bb4:  // pred: ^bb2
    %11 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %12 = llvm.call @"array_append<felt252>"(%11, %0) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.return %13 : !llvm.struct<(i16, array<16 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::array_at::<core::integer::u32>"(%arg0: !llvm.ptr<struct<(i16, array<16 x i8>)>>, %arg1: !llvm.struct<(i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.mlir.constant(1637570914057682275393755530660268060279989363 : i256) : i256
    llvm.br ^bb2(%arg1, %arg2 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb2
    %1 = llvm.extractvalue %4[2] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.getelementptr %1[%5] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %3 = llvm.load %2 : !llvm.ptr -> i32
    llvm.br ^bb3(%3 : i32)
  ^bb2(%4: !llvm.struct<(i32, i32, ptr)>, %5: i32):  // pred: ^bb0
    %6 = llvm.extractvalue %4[0] : !llvm.struct<(i32, i32, ptr)> 
    %7 = llvm.icmp "ult" %5, %6 : i32
    llvm.cond_br %7, ^bb1, ^bb4
  ^bb3(%8: i32):  // pred: ^bb1
    %9 = llvm.call @"struct_construct<Tuple<Box<u32>>>"(%8) : (i32) -> !llvm.struct<(i32)>
    %10 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%9) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.br ^bb5(%10 : !llvm.struct<(i16, array<16 x i8>)>)
  ^bb4:  // pred: ^bb2
    %11 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %12 = llvm.call @"array_append<felt252>"(%11, %0) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<16 x i8>)>
    llvm.br ^bb5(%13 : !llvm.struct<(i16, array<16 x i8>)>)
  ^bb5(%14: !llvm.struct<(i16, array<16 x i8>)>):  // 2 preds: ^bb3, ^bb4
    llvm.store %14, %arg0 : !llvm.ptr<struct<(i16, array<16 x i8>)>>
    llvm.return
  }
}

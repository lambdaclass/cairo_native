module attributes {llvm.data_layout = ""} {
  llvm.func @realloc(!llvm.ptr, i64) -> !llvm.ptr
  llvm.func @free(!llvm.ptr)
  llvm.func @dprintf(i32, !llvm.ptr, ...) -> i32
  llvm.func internal @"array_new<u32>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.mlir.constant(32 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<(i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<(i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<(i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    llvm.store %arg1, %12 : i32, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
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
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<20 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<20 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<20 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32, i32, ptr)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<20 x i8>)>
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
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%arg0: !llvm.struct<(i32)>) -> !llvm.struct<(i16, array<12 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<12 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<12 x i8>)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<12 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<12 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32, i32, ptr)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<12 x i8>)>
  }
  llvm.func internal @"struct_construct<Tuple<Box<u32>>>"(%arg0: i32) -> !llvm.struct<(i32)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32)>
    %1 = llvm.insertvalue %arg0, %0[0] : !llvm.struct<(i32)> 
    llvm.return %1 : !llvm.struct<(i32)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%arg0: !llvm.struct<(i32)>) -> !llvm.struct<(i16, array<12 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<12 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(0 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<12 x i8>)>
  }
  llvm.func internal @"array_new<felt252>"() -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, ptr)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(8 : i32) : i32
    %3 = llvm.insertvalue %1, %0[0] : !llvm.struct<(i32, i32, ptr)> 
    %4 = llvm.insertvalue %2, %3[1] : !llvm.struct<(i32, i32, ptr)> 
    %5 = llvm.mlir.constant(256 : i64) : i64
    %6 = llvm.mlir.null : !llvm.ptr
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %4[2] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %8 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"array_append<felt252>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i256) -> !llvm.struct<(i32, i32, ptr)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, ptr)> 
    %2 = llvm.icmp "ult" %0, %1 : i32
    llvm.cond_br %2, ^bb2(%arg0 : !llvm.struct<(i32, i32, ptr)>), ^bb1
  ^bb1:  // pred: ^bb0
    %3 = llvm.mlir.constant(2 : i32) : i32
    %4 = llvm.mul %1, %3  : i32
    %5 = llvm.zext %4 : i32 to i64
    %6 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %7 = llvm.call @realloc(%6, %5) : (!llvm.ptr, i64) -> !llvm.ptr
    %8 = llvm.insertvalue %7, %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %9 = llvm.insertvalue %4, %8[1] : !llvm.struct<(i32, i32, ptr)> 
    llvm.br ^bb2(%9 : !llvm.struct<(i32, i32, ptr)>)
  ^bb2(%10: !llvm.struct<(i32, i32, ptr)>):  // 2 preds: ^bb0, ^bb1
    %11 = llvm.extractvalue %10[2] : !llvm.struct<(i32, i32, ptr)> 
    %12 = llvm.getelementptr %11[%0] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    llvm.store %arg1, %12 : i256, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.add %0, %13  : i32
    %15 = llvm.insertvalue %14, %10[0] : !llvm.struct<(i32, i32, ptr)> 
    llvm.return %15 : !llvm.struct<(i32, i32, ptr)>
  }
  llvm.func internal @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%arg0: !llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<12 x i8>)> attributes {llvm.dso_local, passthrough = ["norecurse", "alwaysinline", "nounwind"]} {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.alloca %0 x !llvm.struct<(i16, array<12 x i8>)> : (i64) -> !llvm.ptr
    %2 = llvm.mlir.constant(1 : i16) : i16
    %3 = llvm.getelementptr inbounds %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %2, %3 : i16, !llvm.ptr
    %4 = llvm.getelementptr inbounds %1[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %arg0, %4 : !llvm.struct<(i32, i32, ptr)>, !llvm.ptr
    %5 = llvm.load %1 : !llvm.ptr -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %5 : !llvm.struct<(i16, array<12 x i8>)>
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
  llvm.func internal @"print_Tuple<u32, u32, u32, u32, u32>"(%arg0: !llvm.struct<(i32, i32, i32, i32, i32)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%0) : (i32) -> ()
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%1) : (i32) -> ()
    %2 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%2) : (i32) -> ()
    %3 = llvm.extractvalue %arg0[3] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%3) : (i32) -> ()
    %4 = llvm.extractvalue %arg0[4] : !llvm.struct<(i32, i32, i32, i32, i32)> 
    llvm.call @print_u32(%4) : (i32) -> ()
    llvm.return
  }
  llvm.func internal @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(%arg0: !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)> 
    llvm.call @"print_Tuple<u32, u32, u32, u32, u32>"(%0) : (!llvm.struct<(i32, i32, i32, i32, i32)>) -> ()
    llvm.return
  }
  llvm.func internal @print_felt252(%arg0: i256) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.mlir.constant(224 : i256) : i256
    %1 = llvm.ashr %arg0, %0  : i256
    %2 = llvm.trunc %1 : i256 to i32
    %3 = llvm.mlir.constant(5 : i64) : i64
    %4 = llvm.alloca %3 x i8 : (i64) -> !llvm.ptr
    %5 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %5, %4 : !llvm.array<5 x i8>, !llvm.ptr
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.call @dprintf(%6, %4, %2) : (i32, !llvm.ptr, i32) -> i32
    %8 = llvm.mlir.constant(192 : i256) : i256
    %9 = llvm.ashr %arg0, %8  : i256
    %10 = llvm.trunc %9 : i256 to i32
    %11 = llvm.mlir.constant(5 : i64) : i64
    %12 = llvm.alloca %11 x i8 : (i64) -> !llvm.ptr
    %13 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %13, %12 : !llvm.array<5 x i8>, !llvm.ptr
    %14 = llvm.mlir.constant(1 : i32) : i32
    %15 = llvm.call @dprintf(%14, %12, %10) : (i32, !llvm.ptr, i32) -> i32
    %16 = llvm.mlir.constant(160 : i256) : i256
    %17 = llvm.ashr %arg0, %16  : i256
    %18 = llvm.trunc %17 : i256 to i32
    %19 = llvm.mlir.constant(5 : i64) : i64
    %20 = llvm.alloca %19 x i8 : (i64) -> !llvm.ptr
    %21 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %21, %20 : !llvm.array<5 x i8>, !llvm.ptr
    %22 = llvm.mlir.constant(1 : i32) : i32
    %23 = llvm.call @dprintf(%22, %20, %18) : (i32, !llvm.ptr, i32) -> i32
    %24 = llvm.mlir.constant(128 : i256) : i256
    %25 = llvm.ashr %arg0, %24  : i256
    %26 = llvm.trunc %25 : i256 to i32
    %27 = llvm.mlir.constant(5 : i64) : i64
    %28 = llvm.alloca %27 x i8 : (i64) -> !llvm.ptr
    %29 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %29, %28 : !llvm.array<5 x i8>, !llvm.ptr
    %30 = llvm.mlir.constant(1 : i32) : i32
    %31 = llvm.call @dprintf(%30, %28, %26) : (i32, !llvm.ptr, i32) -> i32
    %32 = llvm.mlir.constant(96 : i256) : i256
    %33 = llvm.ashr %arg0, %32  : i256
    %34 = llvm.trunc %33 : i256 to i32
    %35 = llvm.mlir.constant(5 : i64) : i64
    %36 = llvm.alloca %35 x i8 : (i64) -> !llvm.ptr
    %37 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %37, %36 : !llvm.array<5 x i8>, !llvm.ptr
    %38 = llvm.mlir.constant(1 : i32) : i32
    %39 = llvm.call @dprintf(%38, %36, %34) : (i32, !llvm.ptr, i32) -> i32
    %40 = llvm.mlir.constant(64 : i256) : i256
    %41 = llvm.ashr %arg0, %40  : i256
    %42 = llvm.trunc %41 : i256 to i32
    %43 = llvm.mlir.constant(5 : i64) : i64
    %44 = llvm.alloca %43 x i8 : (i64) -> !llvm.ptr
    %45 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %45, %44 : !llvm.array<5 x i8>, !llvm.ptr
    %46 = llvm.mlir.constant(1 : i32) : i32
    %47 = llvm.call @dprintf(%46, %44, %42) : (i32, !llvm.ptr, i32) -> i32
    %48 = llvm.mlir.constant(32 : i256) : i256
    %49 = llvm.ashr %arg0, %48  : i256
    %50 = llvm.trunc %49 : i256 to i32
    %51 = llvm.mlir.constant(5 : i64) : i64
    %52 = llvm.alloca %51 x i8 : (i64) -> !llvm.ptr
    %53 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %53, %52 : !llvm.array<5 x i8>, !llvm.ptr
    %54 = llvm.mlir.constant(1 : i32) : i32
    %55 = llvm.call @dprintf(%54, %52, %50) : (i32, !llvm.ptr, i32) -> i32
    %56 = llvm.mlir.constant(0 : i256) : i256
    %57 = llvm.trunc %arg0 : i256 to i32
    %58 = llvm.mlir.constant(5 : i64) : i64
    %59 = llvm.alloca %58 x i8 : (i64) -> !llvm.ptr
    %60 = llvm.mlir.constant(dense<[37, 48, 56, 88, 0]> : tensor<5xi8>) : !llvm.array<5 x i8>
    llvm.store %60, %59 : !llvm.array<5 x i8>, !llvm.ptr
    %61 = llvm.mlir.constant(1 : i32) : i32
    %62 = llvm.call @dprintf(%61, %59, %57) : (i32, !llvm.ptr, i32) -> i32
    %63 = llvm.mlir.constant(2 : i64) : i64
    %64 = llvm.alloca %63 x i8 : (i64) -> !llvm.ptr
    %65 = llvm.mlir.constant(dense<[10, 0]> : tensor<2xi8>) : !llvm.array<2 x i8>
    llvm.store %65, %64 : !llvm.array<2 x i8>, !llvm.ptr
    %66 = llvm.mlir.constant(1 : i32) : i32
    %67 = llvm.call @dprintf(%66, %64) : (i32, !llvm.ptr) -> i32
    llvm.return
  }
  llvm.func internal @"print_Array<felt252>"(%arg0: !llvm.struct<(i32, i32, ptr)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[2] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.extractvalue %arg0[0] : !llvm.struct<(i32, i32, ptr)> 
    %3 = llvm.mlir.constant(1 : i32) : i32
    llvm.br ^bb1(%1 : i32)
  ^bb1(%4: i32):  // 2 preds: ^bb0, ^bb2
    %5 = llvm.icmp "ult" %4, %2 : i32
    llvm.cond_br %5, ^bb2(%4 : i32), ^bb3
  ^bb2(%6: i32):  // pred: ^bb1
    %7 = llvm.getelementptr %0[%6] : (!llvm.ptr, i32) -> !llvm.ptr, i256
    %8 = llvm.load %7 : !llvm.ptr -> i256
    llvm.call @print_felt252(%8) : (i256) -> ()
    %9 = llvm.add %6, %3  : i32
    llvm.br ^bb1(%9 : i32)
  ^bb3:  // pred: ^bb1
    llvm.return
  }
  llvm.func internal @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(%arg0: !llvm.struct<(i16, array<20 x i8>)>) attributes {llvm.dso_local, passthrough = ["norecurse", "nounwind"]} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(i16, array<20 x i8>)> 
    %1 = llvm.zext %0 : i16 to i32
    %2 = llvm.mlir.constant(4 : i64) : i64
    %3 = llvm.alloca %2 x i8 : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(dense<[37, 88, 10, 0]> : tensor<4xi8>) : !llvm.array<4 x i8>
    llvm.store %4, %3 : !llvm.array<4 x i8>, !llvm.ptr
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.call @dprintf(%5, %3, %1) : (i32, !llvm.ptr, i32) -> i32
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.array<20 x i8> : (i64) -> !llvm.ptr
    %9 = llvm.extractvalue %arg0[1] : !llvm.struct<(i16, array<20 x i8>)> 
    llvm.store %9, %8 : !llvm.array<20 x i8>, !llvm.ptr
    llvm.switch %0 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    %10 = llvm.mlir.constant(55 : i64) : i64
    %11 = llvm.alloca %10 x i8 : (i64) -> !llvm.ptr
    %12 = llvm.mlir.constant(dense<[48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 48, 10, 0]> : tensor<55xi8>) : !llvm.array<55 x i8>
    llvm.store %12, %11 : !llvm.array<55 x i8>, !llvm.ptr
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.call @dprintf(%13, %11) : (i32, !llvm.ptr) -> i32
    %15 = llvm.load %8 : !llvm.ptr -> !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
    llvm.call @"print_Tuple<Tuple<u32, u32, u32, u32, u32>>"(%15) : (!llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) -> ()
    llvm.return
  ^bb2:  // pred: ^bb0
    %16 = llvm.load %8 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.call @"print_Array<felt252>"(%16) : (!llvm.struct<(i32, i32, ptr)>) -> ()
    llvm.return
  ^bb3:  // pred: ^bb0
    llvm.return
  }
  llvm.func @main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.call @"print_core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>"(%0) : (!llvm.struct<(i16, array<20 x i8>)>) -> ()
    llvm.return
  }
  llvm.func @_mlir_ciface_main() attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.call @main() : () -> ()
    llvm.return
  }
  llvm.func @"example_array::example_array::main"() -> !llvm.struct<(i16, array<20 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb16
  ^bb1:  // pred: ^bb16
    %0 = llvm.load %28 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb17(%22, %0 : !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb2:  // pred: ^bb16
    %1 = llvm.load %28 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb26(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb16
    llvm.unreachable
  ^bb4:  // pred: ^bb17
    %2 = llvm.load %37 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb18(%31, %29, %2 : i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb5:  // pred: ^bb17
    %3 = llvm.load %37 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb25(%3 : !llvm.struct<(i32, i32, ptr)>)
  ^bb6:  // pred: ^bb17
    llvm.unreachable
  ^bb7:  // pred: ^bb18
    %4 = llvm.load %47 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb19(%38, %41, %39, %4 : i32, i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb8:  // pred: ^bb18
    %5 = llvm.load %47 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb24(%5 : !llvm.struct<(i32, i32, ptr)>)
  ^bb9:  // pred: ^bb18
    llvm.unreachable
  ^bb10:  // pred: ^bb19
    %6 = llvm.load %58 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb20(%48, %49, %52, %50, %6 : i32, i32, i32, !llvm.struct<(i32, i32, ptr)>, !llvm.struct<(i32)>)
  ^bb11:  // pred: ^bb19
    %7 = llvm.load %58 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb23(%7 : !llvm.struct<(i32, i32, ptr)>)
  ^bb12:  // pred: ^bb19
    llvm.unreachable
  ^bb13:  // pred: ^bb20
    %8 = llvm.load %70 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb21(%59, %60, %61, %64, %8 : i32, i32, i32, i32, !llvm.struct<(i32)>)
  ^bb14:  // pred: ^bb20
    %9 = llvm.load %70 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb22(%9 : !llvm.struct<(i32, i32, ptr)>)
  ^bb15:  // pred: ^bb20
    llvm.unreachable
  ^bb16:  // pred: ^bb0
    %10 = llvm.call @"array_new<u32>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %11 = llvm.mlir.constant(1 : i32) : i32
    %12 = llvm.call @"array_append<u32>"(%10, %11) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.mlir.constant(2 : i32) : i32
    %14 = llvm.call @"array_append<u32>"(%12, %13) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %15 = llvm.mlir.constant(3 : i32) : i32
    %16 = llvm.call @"array_append<u32>"(%14, %15) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %17 = llvm.mlir.constant(4 : i32) : i32
    %18 = llvm.call @"array_append<u32>"(%16, %17) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %19 = llvm.mlir.constant(5 : i32) : i32
    %20 = llvm.call @"array_append<u32>"(%18, %19) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %21 = llvm.call @"array_len<u32>"(%20) : (!llvm.struct<(i32, i32, ptr)>) -> i32
    %22 = llvm.call @"array_append<u32>"(%20, %21) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i32, i32, ptr)>
    %23 = llvm.mlir.constant(0 : i32) : i32
    %24 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%22, %23) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    %25 = llvm.extractvalue %24[0] : !llvm.struct<(i16, array<12 x i8>)> 
    %26 = llvm.extractvalue %24[1] : !llvm.struct<(i16, array<12 x i8>)> 
    %27 = llvm.mlir.constant(1 : i64) : i64
    %28 = llvm.alloca %27 x !llvm.array<12 x i8> : (i64) -> !llvm.ptr
    llvm.store %26, %28 : !llvm.array<12 x i8>, !llvm.ptr
    llvm.switch %25 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb17(%29: !llvm.struct<(i32, i32, ptr)>, %30: !llvm.struct<(i32)>):  // pred: ^bb1
    %31 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%30) : (!llvm.struct<(i32)>) -> i32
    %32 = llvm.mlir.constant(1 : i32) : i32
    %33 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%29, %32) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    %34 = llvm.extractvalue %33[0] : !llvm.struct<(i16, array<12 x i8>)> 
    %35 = llvm.extractvalue %33[1] : !llvm.struct<(i16, array<12 x i8>)> 
    %36 = llvm.mlir.constant(1 : i64) : i64
    %37 = llvm.alloca %36 x !llvm.array<12 x i8> : (i64) -> !llvm.ptr
    llvm.store %35, %37 : !llvm.array<12 x i8>, !llvm.ptr
    llvm.switch %34 : i16, ^bb6 [
      0: ^bb4,
      1: ^bb5
    ]
  ^bb18(%38: i32, %39: !llvm.struct<(i32, i32, ptr)>, %40: !llvm.struct<(i32)>):  // pred: ^bb4
    %41 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%40) : (!llvm.struct<(i32)>) -> i32
    %42 = llvm.mlir.constant(2 : i32) : i32
    %43 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%39, %42) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    %44 = llvm.extractvalue %43[0] : !llvm.struct<(i16, array<12 x i8>)> 
    %45 = llvm.extractvalue %43[1] : !llvm.struct<(i16, array<12 x i8>)> 
    %46 = llvm.mlir.constant(1 : i64) : i64
    %47 = llvm.alloca %46 x !llvm.array<12 x i8> : (i64) -> !llvm.ptr
    llvm.store %45, %47 : !llvm.array<12 x i8>, !llvm.ptr
    llvm.switch %44 : i16, ^bb9 [
      0: ^bb7,
      1: ^bb8
    ]
  ^bb19(%48: i32, %49: i32, %50: !llvm.struct<(i32, i32, ptr)>, %51: !llvm.struct<(i32)>):  // pred: ^bb7
    %52 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%51) : (!llvm.struct<(i32)>) -> i32
    %53 = llvm.mlir.constant(3 : i32) : i32
    %54 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%50, %53) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    %55 = llvm.extractvalue %54[0] : !llvm.struct<(i16, array<12 x i8>)> 
    %56 = llvm.extractvalue %54[1] : !llvm.struct<(i16, array<12 x i8>)> 
    %57 = llvm.mlir.constant(1 : i64) : i64
    %58 = llvm.alloca %57 x !llvm.array<12 x i8> : (i64) -> !llvm.ptr
    llvm.store %56, %58 : !llvm.array<12 x i8>, !llvm.ptr
    llvm.switch %55 : i16, ^bb12 [
      0: ^bb10,
      1: ^bb11
    ]
  ^bb20(%59: i32, %60: i32, %61: i32, %62: !llvm.struct<(i32, i32, ptr)>, %63: !llvm.struct<(i32)>):  // pred: ^bb10
    %64 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%63) : (!llvm.struct<(i32)>) -> i32
    %65 = llvm.mlir.constant(4 : i32) : i32
    %66 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%62, %65) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    %67 = llvm.extractvalue %66[0] : !llvm.struct<(i16, array<12 x i8>)> 
    %68 = llvm.extractvalue %66[1] : !llvm.struct<(i16, array<12 x i8>)> 
    %69 = llvm.mlir.constant(1 : i64) : i64
    %70 = llvm.alloca %69 x !llvm.array<12 x i8> : (i64) -> !llvm.ptr
    llvm.store %68, %70 : !llvm.array<12 x i8>, !llvm.ptr
    llvm.switch %67 : i16, ^bb15 [
      0: ^bb13,
      1: ^bb14
    ]
  ^bb21(%71: i32, %72: i32, %73: i32, %74: i32, %75: !llvm.struct<(i32)>):  // pred: ^bb13
    %76 = llvm.call @"struct_deconstruct<Tuple<u32>>"(%75) : (!llvm.struct<(i32)>) -> i32
    %77 = llvm.call @"struct_construct<Tuple<u32, u32, u32, u32, u32>>"(%71, %72, %73, %74, %76) : (i32, i32, i32, i32, i32) -> !llvm.struct<(i32, i32, i32, i32, i32)>
    %78 = llvm.call @"struct_construct<Tuple<Tuple<u32, u32, u32, u32, u32>>>"(%77) : (!llvm.struct<(i32, i32, i32, i32, i32)>) -> !llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>
    %79 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 0>"(%78) : (!llvm.struct<(struct<(i32, i32, i32, i32, i32)>)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %79 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb22(%80: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb14
    %81 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%80) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %81 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb23(%82: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb11
    %83 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%82) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %83 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb24(%84: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb8
    %85 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%84) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %85 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb25(%86: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb5
    %87 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%86) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %87 : !llvm.struct<(i16, array<20 x i8>)>
  ^bb26(%88: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %89 = llvm.call @"enum_init<core::PanicResult::<((core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32, core::integer::u32))>, 1>"(%88) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.return %89 : !llvm.struct<(i16, array<20 x i8>)>
  }
  llvm.func @"_mlir_ciface_example_array::example_array::main"(%arg0: !llvm.ptr<struct<(i16, array<20 x i8>)>>) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"example_array::example_array::main"() : () -> !llvm.struct<(i16, array<20 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<20 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i16, array<12 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb4(%arg0, %arg1 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb4
    %0 = llvm.load %8 : !llvm.ptr -> !llvm.struct<(i32)>
    llvm.br ^bb5(%0 : !llvm.struct<(i32)>)
  ^bb2:  // pred: ^bb4
    %1 = llvm.load %8 : !llvm.ptr -> !llvm.struct<(i32, i32, ptr)>
    llvm.br ^bb6(%1 : !llvm.struct<(i32, i32, ptr)>)
  ^bb3:  // pred: ^bb4
    llvm.unreachable
  ^bb4(%2: !llvm.struct<(i32, i32, ptr)>, %3: i32):  // pred: ^bb0
    %4 = llvm.call @"core::array::array_at::<core::integer::u32>"(%2, %3) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    %5 = llvm.extractvalue %4[0] : !llvm.struct<(i16, array<12 x i8>)> 
    %6 = llvm.extractvalue %4[1] : !llvm.struct<(i16, array<12 x i8>)> 
    %7 = llvm.mlir.constant(1 : i64) : i64
    %8 = llvm.alloca %7 x !llvm.array<12 x i8> : (i64) -> !llvm.ptr
    llvm.store %6, %8 : !llvm.array<12 x i8>, !llvm.ptr
    llvm.switch %5 : i16, ^bb3 [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb5(%9: !llvm.struct<(i32)>):  // pred: ^bb1
    %10 = llvm.call @"struct_deconstruct<Tuple<Box<u32>>>"(%9) : (!llvm.struct<(i32)>) -> i32
    %11 = llvm.call @"struct_construct<Tuple<u32>>"(%10) : (i32) -> !llvm.struct<(i32)>
    %12 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 0>"(%11) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %12 : !llvm.struct<(i16, array<12 x i8>)>
  ^bb6(%13: !llvm.struct<(i32, i32, ptr)>):  // pred: ^bb2
    %14 = llvm.call @"enum_init<core::PanicResult::<(@core::integer::u32)>, 1>"(%13) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %14 : !llvm.struct<(i16, array<12 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::ArrayIndex::<core::integer::u32>::index"(%arg0: !llvm.ptr<struct<(i16, array<12 x i8>)>>, %arg1: !llvm.struct<(i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::ArrayIndex::<core::integer::u32>::index"(%arg1, %arg2) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<12 x i8>)>>
    llvm.return
  }
  llvm.func @"core::array::array_at::<core::integer::u32>"(%arg0: !llvm.struct<(i32, i32, ptr)>, %arg1: i32) -> !llvm.struct<(i16, array<12 x i8>)> attributes {llvm.dso_local, llvm.emit_c_interface} {
    llvm.br ^bb2(%arg0, %arg1 : !llvm.struct<(i32, i32, ptr)>, i32)
  ^bb1:  // pred: ^bb2
    %0 = llvm.extractvalue %3[2] : !llvm.struct<(i32, i32, ptr)> 
    %1 = llvm.getelementptr %0[%4] : (!llvm.ptr, i32) -> !llvm.ptr, i32
    %2 = llvm.load %1 : !llvm.ptr -> i32
    llvm.br ^bb3(%2 : i32)
  ^bb2(%3: !llvm.struct<(i32, i32, ptr)>, %4: i32):  // pred: ^bb0
    %5 = llvm.extractvalue %3[0] : !llvm.struct<(i32, i32, ptr)> 
    %6 = llvm.icmp "ult" %4, %5 : i32
    llvm.cond_br %6, ^bb1, ^bb4
  ^bb3(%7: i32):  // pred: ^bb1
    %8 = llvm.call @"struct_construct<Tuple<Box<u32>>>"(%7) : (i32) -> !llvm.struct<(i32)>
    %9 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 0>"(%8) : (!llvm.struct<(i32)>) -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %9 : !llvm.struct<(i16, array<12 x i8>)>
  ^bb4:  // pred: ^bb2
    %10 = llvm.call @"array_new<felt252>"() : () -> !llvm.struct<(i32, i32, ptr)>
    %11 = llvm.mlir.constant(1637570914057682275393755530660268060279989363 : i256) : i256
    %12 = llvm.call @"array_append<felt252>"(%10, %11) : (!llvm.struct<(i32, i32, ptr)>, i256) -> !llvm.struct<(i32, i32, ptr)>
    %13 = llvm.call @"enum_init<core::PanicResult::<(core::box::Box::<@core::integer::u32>)>, 1>"(%12) : (!llvm.struct<(i32, i32, ptr)>) -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.return %13 : !llvm.struct<(i16, array<12 x i8>)>
  }
  llvm.func @"_mlir_ciface_core::array::array_at::<core::integer::u32>"(%arg0: !llvm.ptr<struct<(i16, array<12 x i8>)>>, %arg1: !llvm.struct<(i32, i32, ptr)>, %arg2: i32) attributes {llvm.dso_local, llvm.emit_c_interface} {
    %0 = llvm.call @"core::array::array_at::<core::integer::u32>"(%arg1, %arg2) : (!llvm.struct<(i32, i32, ptr)>, i32) -> !llvm.struct<(i16, array<12 x i8>)>
    llvm.store %0, %arg0 : !llvm.ptr<struct<(i16, array<12 x i8>)>>
    llvm.return
  }
}

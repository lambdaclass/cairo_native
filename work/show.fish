for file in *.sierra.json
    set name $(path change-extension '' $(path change-extension '' $file))
    echo "Tx: $name..."
    jq '
        def round(precision):.*pow(10;precision)|round/pow(10;precision);
        def unit(s):.|tostring+" "+s;
        def to_s(old; new):.[new]=(.[old]/1000|round(2)|unit("S")) | del(.[old]);
        def to_mb(old; new):.[new]=(.[old]/1024/1024|round(2)|unit("MB")) | del(.[old]);
        def sort_by_freq:
            to_entries
            | sort_by(-.value)
            | (map(.value) | add) as $total
            | map(.value = (.value / $total * 100 | round(2)))
            | map(select(.value >= 1))
            | from_entries
        ;

        .
        | to_s("compilation_total_time_ms";"compilation_total_time")
        | to_s("compilation_sierra_to_mlir_time_ms";"compilation_sierra_to_mlir_time")
        | to_s("compilation_mlir_passes_time_ms";"compilation_mlir_passes_time")
        | to_s("compilation_mlir_to_llvm_time_ms";"compilation_mlir_to_llvm_time")
        | to_s("compilation_llvm_passes_time_ms";"compilation_llvm_passes_time")
        | to_s("compilation_llvm_to_object_time_ms";"compilation_llvm_to_object_time")
        | to_s("compilation_linking_time_ms";"compilation_linking_time")
        | to_mb("object_size_bytes";"object_size")
        | .sierra_libfunc_frequency=(.sierra_libfunc_frequency // {} | sort_by_freq)
        ' "$name.stats.json"
end

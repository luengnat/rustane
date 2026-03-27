//! Test ANE with exact working project approach

use std::ffi::{c_char, c_void, CString};

#[test]
#[cfg(target_vendor = "apple")]
fn test_ane_exact_approach() {
    println!("\n=== Testing ANE with Exact Working Approach ===\n");

    // Try loading the dylib directly
    let lib_path = CString::new("/Users/nat/dev/ANE/bridge/libane_bridge.dylib").unwrap();

    unsafe {
        let handle = libc::dlopen(lib_path.as_ptr(), libc::RTLD_NOW);
        if handle.is_null() {
            println!("❌ Failed to load ANE bridge library");
            return;
        }
        println!("✅ ANE bridge library loaded");

        // Get function pointers
        type InitFn = unsafe extern "C" fn() -> i32;
        type CompileFn = unsafe extern "C" fn(
            mil_text: *const c_char,
            mil_len: usize,
            weight_names: *mut *const c_char,
            weight_datas: *mut *const u8,
            weight_lens: *const usize,
            n_weights: i32,
            n_inputs: i32,
            input_sizes: *const usize,
            n_outputs: i32,
            output_sizes: *const usize,
        ) -> *mut c_void;

        let init_ptr = libc::dlsym(handle, b"ane_bridge_init\0".as_ptr() as *const c_char);
        let compile_ptr = libc::dlsym(
            handle,
            b"ane_bridge_compile_multi_weights\0".as_ptr() as *const c_char,
        );

        if init_ptr.is_null() || compile_ptr.is_null() {
            println!("❌ Failed to get function pointers");
            libc::dlclose(handle);
            return;
        }

        let init: InitFn = std::mem::transmute(init_ptr);
        let compile: CompileFn = std::mem::transmute(compile_ptr);

        // Initialize
        let ret = init();
        println!(
            "ANE Bridge Init: {}",
            if ret == 0 {
                "✅ SUCCESS"
            } else {
                "❌ FAILED"
            }
        );

        if ret != 0 {
            libc::dlclose(handle);
            return;
        }

        // Try compiling with full program wrapper (matching working ANE project)
        let mil = r#"program(1.3)
[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3510.2.1"}, {"coremlc-version", "3505.4.1"}, {"coremltools-component-milinternal", ""}, {"coremltools-version", "9.0"}})]
{
    func main<ios18>(tensor<fp16, [1, 4, 1, 4]> x) {
        tensor<fp16, [1,4,1,4]> out = identity(x=x)[name=string("out")];
    } -> (out);
}
"#;
        let mil_cstring = CString::new(mil).unwrap();
        let input_sizes: [usize; 1] = [32]; // 4*4*2 bytes for fp16
        let output_sizes: [usize; 1] = [32];

        println!("\nCompiling MIL:\n{}", mil);

        let kernel = compile(
            mil_cstring.as_ptr(),
            mil.len(),
            std::ptr::null_mut(), // weight_names
            std::ptr::null_mut(), // weight_datas
            std::ptr::null(),     // weight_lens
            0,                    // n_weights
            1,                    // n_inputs
            input_sizes.as_ptr(),
            1, // n_outputs
            output_sizes.as_ptr(),
        );

        if !kernel.is_null() {
            println!("✅ COMPILED SUCCESSFULLY!");

            // Try to evaluate
            let eval_ptr = libc::dlsym(handle, b"ane_bridge_eval\0".as_ptr() as *const c_char);

            if !eval_ptr.is_null() {
                type EvalFn = unsafe extern "C" fn(kernel: *mut c_void) -> i32;
                type WriteFn = unsafe extern "C" fn(
                    kernel: *mut c_void,
                    idx: i32,
                    data: *const c_void,
                    bytes: usize,
                ) -> i32;
                type ReadFn = unsafe extern "C" fn(
                    kernel: *mut c_void,
                    idx: i32,
                    data: *mut c_void,
                    bytes: usize,
                ) -> i32;

                let eval: EvalFn = std::mem::transmute(eval_ptr);

                // Write input
                let write_ptr = libc::dlsym(
                    handle,
                    b"ane_bridge_write_input\0".as_ptr() as *const c_char,
                );
                if !write_ptr.is_null() {
                    let write: WriteFn = std::mem::transmute(write_ptr);

                    // fp16 1.0 = 0x3C00 in little-endian: [0x00, 0x3C]
                    let input_bytes: Vec<u8> =
                        (0..32).map(|_| 0x3C00u16.to_le_bytes()).flatten().collect(); // 16 fp16 values of 1.0

                    write(
                        kernel,
                        0,
                        input_bytes.as_ptr() as *const c_void,
                        input_bytes.len(),
                    );

                    // Eval
                    let eval_ret = eval(kernel);
                    println!(
                        "Eval: {}",
                        if eval_ret != 0 {
                            "✅ SUCCESS"
                        } else {
                            "❌ FAILED"
                        }
                    );

                    // Read output
                    if eval_ret != 0 {
                        let read_ptr = libc::dlsym(
                            handle,
                            b"ane_bridge_read_output\0".as_ptr() as *const c_char,
                        );
                        if !read_ptr.is_null() {
                            let read: ReadFn = std::mem::transmute(read_ptr);

                            let mut output_bytes = vec![0u8; 32];
                            read(
                                kernel,
                                0,
                                output_bytes.as_mut_ptr() as *mut c_void,
                                output_bytes.len(),
                            );

                            // Check first 4 values (should be 1.0 fp16 = 0x3C00)
                            let first_val = u16::from_le_bytes([output_bytes[0], output_bytes[1]]);
                            println!(
                                "First output value (fp16 bits): 0x{:04X} (expected 0x3C00)",
                                first_val
                            );
                        }
                    }
                }
            }

            // Free kernel
            let free_ptr = libc::dlsym(handle, b"ane_bridge_free\0".as_ptr() as *const c_char);
            if !free_ptr.is_null() {
                type FreeFn = unsafe extern "C" fn(kernel: *mut c_void) -> i32;
                let free_kernel: FreeFn = std::mem::transmute(free_ptr);
                free_kernel(kernel);
            }
        } else {
            println!("❌ Compilation returned null kernel");
        }

        libc::dlclose(handle);
    }

    println!("\n═══════════════════════════════════════");
}

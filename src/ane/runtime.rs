#[cfg(target_vendor = "apple")]
mod apple {
    use core::ffi::{c_char, c_int, c_void, CStr};
    use core::ptr::{self, NonNull};
    use libc::{dlopen, RTLD_NOW};
    use objc2::rc::{autoreleasepool, Retained};
    use objc2::runtime::{AnyClass, AnyObject, Bool, NSObject, NSObjectProtocol};
    use objc2::{extern_class, extern_conformance, extern_methods, msg_send, AnyThread, ClassType};
    use objc2_core_foundation::{CFDictionary, CFRetained};
    use objc2_foundation::{NSArray, NSData, NSDictionary, NSError, NSNumber, NSString};
    use objc2_io_surface::{IOSurfaceLockOptions, IOSurfaceRef};
    use std::ffi::CString;
    use std::fs;
    use std::path::PathBuf;
    use std::slice;
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::OnceLock;
    use std::thread;
    use std::time::Duration;

    const ANE_FRAMEWORK_PATH: &[u8] =
        b"/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine\0";
    const CLASS_DESC: &[u8] = b"_ANEInMemoryModelDescriptor\0";
    const CLASS_INMEM: &[u8] = b"_ANEInMemoryModel\0";
    const CLASS_REQ: &[u8] = b"_ANERequest\0";
    const CLASS_IO: &[u8] = b"_ANEIOSurfaceObject\0";
    const ANE_QOS: u32 = 21;

    extern_class!(
        #[unsafe(super(NSObject))]
        #[thread_kind = AnyThread]
        #[name = "_ANEInMemoryModelDescriptor"]
        struct ANEInMemoryModelDescriptor;
    );

    extern_conformance!(
        unsafe impl NSObjectProtocol for ANEInMemoryModelDescriptor {}
    );

    #[allow(non_snake_case)]
    impl ANEInMemoryModelDescriptor {
        extern_methods!(
            #[unsafe(method(modelWithMILText:weights:optionsPlist:))]
            #[unsafe(method_family = none)]
            pub unsafe fn modelWithMILText_weights_optionsPlist(
                mil_text: &NSData,
                weights: *const NSDictionary<NSString, AnyObject>,
                options_plist: *const AnyObject,
            ) -> Option<Retained<Self>>;
        );
    }

    extern_class!(
        #[unsafe(super(NSObject))]
        #[thread_kind = AnyThread]
        #[name = "_ANEInMemoryModel"]
        struct ANEInMemoryModel;
    );

    extern_conformance!(
        unsafe impl NSObjectProtocol for ANEInMemoryModel {}
    );

    #[allow(non_snake_case)]
    impl ANEInMemoryModel {
        extern_methods!(
            #[unsafe(method(inMemoryModelWithDescriptor:))]
            #[unsafe(method_family = none)]
            pub unsafe fn inMemoryModelWithDescriptor(
                descriptor: &ANEInMemoryModelDescriptor,
            ) -> Option<Retained<Self>>;

            #[unsafe(method(hexStringIdentifier))]
            #[unsafe(method_family = none)]
            pub unsafe fn hexStringIdentifier(&self) -> Retained<NSString>;

            #[unsafe(method(compileWithQoS:options:error:))]
            #[unsafe(method_family = none)]
            pub unsafe fn compileWithQoS_options_error(
                &self,
                qos: u32,
                options: &NSDictionary<AnyObject, AnyObject>,
                error: *mut *mut NSError,
            ) -> Bool;

            #[unsafe(method(loadWithQoS:options:error:))]
            #[unsafe(method_family = none)]
            pub unsafe fn loadWithQoS_options_error(
                &self,
                qos: u32,
                options: &NSDictionary<AnyObject, AnyObject>,
                error: *mut *mut NSError,
            ) -> Bool;

            #[unsafe(method(evaluateWithQoS:options:request:error:))]
            #[unsafe(method_family = none)]
            pub unsafe fn evaluateWithQoS_options_request_error(
                &self,
                qos: u32,
                options: &NSDictionary<AnyObject, AnyObject>,
                request: &ANERequest,
                error: *mut *mut NSError,
            ) -> Bool;

            #[unsafe(method(unloadWithQoS:error:))]
            #[unsafe(method_family = none)]
            pub unsafe fn unloadWithQoS_error(&self, qos: u32, error: *mut *mut NSError) -> Bool;
        );
    }

    extern_class!(
        #[unsafe(super(NSObject))]
        #[thread_kind = AnyThread]
        #[name = "_ANERequest"]
        struct ANERequest;
    );

    extern_conformance!(
        unsafe impl NSObjectProtocol for ANERequest {}
    );

    #[allow(non_snake_case)]
    impl ANERequest {
        extern_methods!(
            #[unsafe(method(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:))]
            #[unsafe(method_family = none)]
            pub unsafe fn requestWithInputs_inputIndices_outputs_outputIndices_weightsBuffer_perfStats_procedureIndex(
                inputs: &NSArray<AnyObject>,
                input_indices: &NSArray<NSNumber>,
                outputs: &NSArray<AnyObject>,
                output_indices: &NSArray<NSNumber>,
                weights_buffer: *const AnyObject,
                perf_stats: *const AnyObject,
                procedure_index: &NSNumber,
            ) -> Option<Retained<Self>>;
        );
    }

    extern_class!(
        #[unsafe(super(NSObject))]
        #[thread_kind = AnyThread]
        #[name = "_ANEIOSurfaceObject"]
        struct ANEIOSurfaceObject;
    );

    extern_conformance!(
        unsafe impl NSObjectProtocol for ANEIOSurfaceObject {}
    );

    #[allow(non_snake_case)]
    impl ANEIOSurfaceObject {
        extern_methods!(
            #[unsafe(method(objectWithIOSurface:))]
            #[unsafe(method_family = none)]
            pub unsafe fn objectWithIOSurface(surface: &IOSurfaceRef) -> Option<Retained<Self>>;
        );
    }

    static CLASSES: OnceLock<Option<()>> = OnceLock::new();
    static COMPILE_COUNT: AtomicI32 = AtomicI32::new(0);

    pub struct ANEKernelHandle {
        model: Retained<ANEInMemoryModel>,
        io_inputs: Vec<CFRetained<IOSurfaceRef>>,
        io_outputs: Vec<CFRetained<IOSurfaceRef>>,
        request: Retained<ANERequest>,
        tmp_dir: PathBuf,
        /// Stored MIL source text for delta compilation (reload with new weights)
        mil_text: Vec<u8>,
        n_inputs: i32,
        n_outputs: i32,
        input_bytes: Vec<usize>,
        output_bytes: Vec<usize>,
    }

    struct ANEPrivateRuntime;

    impl ANEPrivateRuntime {
        fn initialize() -> Option<()> {
            CLASSES
                .get_or_init(|| {
                    let handle = unsafe { dlopen(cstr(ANE_FRAMEWORK_PATH).as_ptr(), RTLD_NOW) };
                    if handle.is_null() {
                        return None;
                    }

                    AnyClass::get(cstr(CLASS_DESC))?;
                    AnyClass::get(cstr(CLASS_INMEM))?;
                    AnyClass::get(cstr(CLASS_REQ))?;
                    AnyClass::get(cstr(CLASS_IO))?;

                    Some(())
                })
                .as_ref()
                .map(|_| ())
        }

        fn create_surface(bytes: usize) -> Option<CFRetained<IOSurfaceRef>> {
            let keys = [
                NSString::from_str("IOSurfaceWidth"),
                NSString::from_str("IOSurfaceHeight"),
                NSString::from_str("IOSurfaceBytesPerElement"),
                NSString::from_str("IOSurfaceBytesPerRow"),
                NSString::from_str("IOSurfaceAllocSize"),
                NSString::from_str("IOSurfacePixelFormat"),
            ];
            let values = vec![
                NSNumber::new_usize(bytes).into(),
                NSNumber::new_u8(1).into(),
                NSNumber::new_u8(1).into(),
                NSNumber::new_usize(bytes).into(),
                NSNumber::new_usize(bytes).into(),
                NSNumber::new_u8(0).into(),
            ];
            let key_refs = keys.iter().map(|key| &**key).collect::<Vec<_>>();
            let props =
                NSDictionary::<NSString, AnyObject>::from_retained_objects(&key_refs, &values);

            // Cast NSDictionary to CFDictionary for IOSurfaceRef::new
            let cf_props: &CFDictionary = unsafe {
                &*(&*props as *const NSDictionary<NSString, AnyObject> as *const CFDictionary)
            };
            unsafe { IOSurfaceRef::new(cf_props) }
        }

        fn create_weight_dictionary(
            weight_names: *mut *const c_char,
            weight_datas: *mut *const u8,
            weight_lens: *const usize,
            n_weights: i32,
        ) -> Option<Retained<NSDictionary<NSString, AnyObject>>> {
            if n_weights <= 0 {
                return None;
            }

            let mut keys = Vec::with_capacity(n_weights as usize);
            let mut values = Vec::with_capacity(n_weights as usize);

            for idx in 0..(n_weights as isize) {
                let name_ptr = unsafe { *weight_names.offset(idx) };
                let data_ptr = unsafe { *weight_datas.offset(idx) };
                let data_len = unsafe { *weight_lens.offset(idx) };
                if name_ptr.is_null() || data_ptr.is_null() {
                    return None;
                }

                let name = unsafe { CStr::from_ptr(name_ptr) }
                    .to_string_lossy()
                    .into_owned();
                let data = NSData::with_bytes(unsafe { slice::from_raw_parts(data_ptr, data_len) });

                let inner_keys = [NSString::from_str("offset"), NSString::from_str("data")];
                let inner_values = vec![NSNumber::new_i32(0).into(), data.into()];
                let inner_key_refs = inner_keys.iter().map(|key| &**key).collect::<Vec<_>>();
                let entry = NSDictionary::<NSString, AnyObject>::from_retained_objects(
                    &inner_key_refs,
                    &inner_values,
                );

                keys.push(NSString::from_str(&name));
                values.push(entry.into());
            }

            let key_refs = keys.iter().map(|key| &**key).collect::<Vec<_>>();
            Some(NSDictionary::<NSString, AnyObject>::from_retained_objects(
                &key_refs, &values,
            ))
        }

        fn write_model_files(
            model: &ANEInMemoryModel,
            mil_data: &NSData,
            weight_names: *mut *const c_char,
            weight_datas: *mut *const u8,
            weight_lens: *const usize,
            n_weights: i32,
        ) -> Option<PathBuf> {
            let hex_id = unsafe { model.hexStringIdentifier() };
            let tmp_dir = std::env::temp_dir().join(hex_id.to_string());
            let weights_dir = tmp_dir.join("weights");
            fs::create_dir_all(&weights_dir).ok()?;
            fs::write(tmp_dir.join("model.mil"), mil_data.to_vec()).ok()?;

            for idx in 0..(n_weights as isize) {
                let name_ptr = unsafe { *weight_names.offset(idx) };
                let data_ptr = unsafe { *weight_datas.offset(idx) };
                let data_len = unsafe { *weight_lens.offset(idx) };
                if name_ptr.is_null() || data_ptr.is_null() {
                    return None;
                }

                let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy();
                let rel = name
                    .strip_prefix("@model_path/")
                    .unwrap_or(name.as_ref())
                    .to_string();
                let full_path = tmp_dir.join(rel);
                if let Some(parent) = full_path.parent() {
                    fs::create_dir_all(parent).ok()?;
                }
                let bytes = unsafe { slice::from_raw_parts(data_ptr, data_len) };
                fs::write(full_path, bytes).ok()?;
            }

            Some(tmp_dir)
        }

        unsafe fn create_request(
            io_inputs: &[CFRetained<IOSurfaceRef>],
            io_outputs: &[CFRetained<IOSurfaceRef>],
        ) -> Option<Retained<ANERequest>> {
            let mut input_objects = Vec::with_capacity(io_inputs.len());
            let mut input_indices = Vec::with_capacity(io_inputs.len());
            for (idx, surface) in io_inputs.iter().enumerate() {
                let obj = unsafe { ANEIOSurfaceObject::objectWithIOSurface(&**surface) }?;
                input_objects.push(obj.into());
                input_indices.push(NSNumber::new_usize(idx));
            }

            let mut output_objects = Vec::with_capacity(io_outputs.len());
            let mut output_indices = Vec::with_capacity(io_outputs.len());
            for (idx, surface) in io_outputs.iter().enumerate() {
                let obj = unsafe { ANEIOSurfaceObject::objectWithIOSurface(&**surface) }?;
                output_objects.push(obj.into());
                output_indices.push(NSNumber::new_usize(idx));
            }

            let inputs = NSArray::<AnyObject>::from_retained_slice(&input_objects);
            let input_ids = NSArray::<NSNumber>::from_retained_slice(&input_indices);
            let outputs = NSArray::<AnyObject>::from_retained_slice(&output_objects);
            let output_ids = NSArray::<NSNumber>::from_retained_slice(&output_indices);
            let procedure = NSNumber::new_u8(0);

            unsafe {
                ANERequest::requestWithInputs_inputIndices_outputs_outputIndices_weightsBuffer_perfStats_procedureIndex(
                    &inputs,
                    &input_ids,
                    &outputs,
                    &output_ids,
                    ptr::null::<AnyObject>(),
                    ptr::null::<AnyObject>(),
                    &procedure,
                )
            }
        }
    }

    pub unsafe fn ane_bridge_init() -> c_int {
        if ANEPrivateRuntime::initialize().is_some() {
            0
        } else {
            -1
        }
    }

    pub unsafe fn ane_bridge_compile_multi_weights(
        mil_text: *const c_char,
        mil_len: usize,
        weight_names: *mut *const c_char,
        weight_datas: *mut *const u8,
        weight_lens: *const usize,
        n_weights: c_int,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle {
        autoreleasepool(|_| {
            let Some(()) = ANEPrivateRuntime::initialize() else {
                return ptr::null_mut();
            };
            if mil_text.is_null() || input_sizes.is_null() || output_sizes.is_null() {
                return ptr::null_mut();
            }

            let mil_bytes = unsafe { slice::from_raw_parts(mil_text.cast::<u8>(), mil_len) };
            let mil_data = NSData::with_bytes(mil_bytes);
            let weights = ANEPrivateRuntime::create_weight_dictionary(
                weight_names,
                weight_datas,
                weight_lens,
                n_weights,
            );

            // Create empty dictionary if no weights provided
            let empty_weights: Retained<NSDictionary<NSString, AnyObject>> =
                unsafe { msg_send![NSDictionary::<NSString, AnyObject>::class(), new] };
            let desc = unsafe {
                ANEInMemoryModelDescriptor::modelWithMILText_weights_optionsPlist(
                    &mil_data,
                    weights
                        .as_ref()
                        .map(|dict| &**dict as *const NSDictionary<NSString, AnyObject>)
                        .unwrap_or(&*empty_weights),
                    ptr::null::<AnyObject>(),
                )
            };
            let Some(desc) = desc else {
                return ptr::null_mut();
            };

            let model = unsafe { ANEInMemoryModel::inMemoryModelWithDescriptor(&desc) };
            let Some(model) = model else {
                return ptr::null_mut();
            };

            let Some(tmp_dir) = ANEPrivateRuntime::write_model_files(
                &model,
                &mil_data,
                weight_names,
                weight_datas,
                weight_lens,
                n_weights,
            ) else {
                return ptr::null_mut();
            };

            let options = empty_dictionary();
            let mut error: *mut NSError = ptr::null_mut();
            let compiled =
                unsafe { model.compileWithQoS_options_error(ANE_QOS, &options, &mut error) };
            if !compiled.as_bool() {
                let _ = fs::remove_dir_all(&tmp_dir);
                eprintln!("ane_bridge: ANE compile failed: {}", ns_error_string(error));
                return ptr::null_mut();
            }

            let mut error: *mut NSError = ptr::null_mut();
            let mut loaded =
                unsafe { model.loadWithQoS_options_error(ANE_QOS, &options, &mut error) };
            if !loaded.as_bool() {
                thread::sleep(Duration::from_millis(100));
                let mut retry_error: *mut NSError = ptr::null_mut();
                loaded =
                    unsafe { model.loadWithQoS_options_error(ANE_QOS, &options, &mut retry_error) };
                if !loaded.as_bool() {
                    let _ = fs::remove_dir_all(&tmp_dir);
                    eprintln!(
                        "ane_bridge: ANE load failed after retry: {}",
                        ns_error_string(retry_error)
                    );
                    return ptr::null_mut();
                }
            }

            let input_bytes =
                unsafe { slice::from_raw_parts(input_sizes, n_inputs as usize) }.to_vec();
            let output_bytes =
                unsafe { slice::from_raw_parts(output_sizes, n_outputs as usize) }.to_vec();

            let io_inputs = input_bytes
                .iter()
                .copied()
                .map(ANEPrivateRuntime::create_surface)
                .collect::<Option<Vec<_>>>();
            let io_outputs = output_bytes
                .iter()
                .copied()
                .map(ANEPrivateRuntime::create_surface)
                .collect::<Option<Vec<_>>>();
            let (Some(io_inputs), Some(io_outputs)) = (io_inputs, io_outputs) else {
                let _ = fs::remove_dir_all(&tmp_dir);
                return ptr::null_mut();
            };

            let Some(request) =
                (unsafe { ANEPrivateRuntime::create_request(&io_inputs, &io_outputs) })
            else {
                let _ = fs::remove_dir_all(&tmp_dir);
                return ptr::null_mut();
            };

            COMPILE_COUNT.fetch_add(1, Ordering::Relaxed);

            // Store MIL text for delta compilation
            let mil_text = mil_bytes.to_vec();

            Box::into_raw(Box::new(ANEKernelHandle {
                model,
                io_inputs,
                io_outputs,
                request,
                tmp_dir,
                mil_text,
                n_inputs,
                n_outputs,
                input_bytes,
                output_bytes,
            }))
        })
    }

    pub unsafe fn ane_bridge_compile(
        mil_text: *const c_char,
        mil_len: usize,
        weight_data: *const u8,
        weight_len: usize,
        n_inputs: c_int,
        input_sizes: *const usize,
        n_outputs: c_int,
        output_sizes: *const usize,
    ) -> *mut ANEKernelHandle {
        if !weight_data.is_null() && weight_len > 0 {
            let name = b"@model_path/weights/weight.bin\0";
            let mut weight_name = name.as_ptr().cast::<c_char>();
            let mut weight_data_ptr = weight_data;
            let weight_len_arr = [weight_len];
            unsafe {
                ane_bridge_compile_multi_weights(
                    mil_text,
                    mil_len,
                    &mut weight_name,
                    &mut weight_data_ptr,
                    weight_len_arr.as_ptr(),
                    1,
                    n_inputs,
                    input_sizes,
                    n_outputs,
                    output_sizes,
                )
            }
        } else {
            unsafe {
                ane_bridge_compile_multi_weights(
                    mil_text,
                    mil_len,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    ptr::null(),
                    0,
                    n_inputs,
                    input_sizes,
                    n_outputs,
                    output_sizes,
                )
            }
        }
    }

    pub unsafe fn ane_bridge_eval(kernel: *mut ANEKernelHandle) -> bool {
        autoreleasepool(|_| {
            let Some(kernel) = (unsafe { kernel.as_ref() }) else {
                return false;
            };
            let options = empty_dictionary();
            let mut error: *mut NSError = ptr::null_mut();
            let ok = unsafe {
                kernel.model.evaluateWithQoS_options_request_error(
                    ANE_QOS,
                    &options,
                    &kernel.request,
                    &mut error,
                )
            };
            if !ok.as_bool() {
                eprintln!("ane_bridge: ANE eval failed: {}", ns_error_string(error));
            }
            ok.as_bool()
        })
    }

    pub unsafe fn ane_bridge_write_input(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        data: *const c_void,
        bytes: usize,
    ) {
        let Some(kernel) = (unsafe { kernel.as_ref() }) else {
            return;
        };
        if idx < 0 || idx >= kernel.n_inputs || data.is_null() {
            return;
        }
        let surface = &kernel.io_inputs[idx as usize];
        unsafe {
            surface.lock(IOSurfaceLockOptions(0), ptr::null_mut());
            ptr::copy_nonoverlapping(
                data.cast::<u8>(),
                surface.base_address().as_ptr().cast(),
                bytes,
            );
            surface.unlock(IOSurfaceLockOptions(0), ptr::null_mut());
        }
    }

    pub unsafe fn ane_bridge_read_output(
        kernel: *mut ANEKernelHandle,
        idx: c_int,
        data: *mut c_void,
        bytes: usize,
    ) {
        let Some(kernel) = (unsafe { kernel.as_ref() }) else {
            return;
        };
        if idx < 0 || idx >= kernel.n_outputs || data.is_null() {
            return;
        }
        let surface = &kernel.io_outputs[idx as usize];
        unsafe {
            surface.lock(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
            ptr::copy_nonoverlapping(
                surface.base_address().as_ptr().cast::<u8>(),
                data.cast(),
                bytes,
            );
            surface.unlock(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
        }
    }

    pub unsafe fn ane_bridge_free(kernel: *mut ANEKernelHandle) {
        autoreleasepool(|_| {
            let Some(mut kernel) =
                NonNull::new(kernel).map(|ptr| unsafe { Box::from_raw(ptr.as_ptr()) })
            else {
                return;
            };

            let mut error: *mut NSError = ptr::null_mut();
            let _ = unsafe { kernel.model.unloadWithQoS_error(ANE_QOS, &mut error) };
            let _ = fs::remove_dir_all(&kernel.tmp_dir);

            kernel.io_inputs.clear();
            kernel.io_outputs.clear();
            kernel.input_bytes.clear();
            kernel.output_bytes.clear();
        })
    }

    pub unsafe fn ane_bridge_get_compile_count() -> c_int {
        COMPILE_COUNT.load(Ordering::Relaxed)
    }

    pub unsafe fn ane_bridge_reset_compile_count() {
        COMPILE_COUNT.store(0, Ordering::Relaxed);
    }

    /// Reload weights by creating a new model with updated weights (delta compilation)
    ///
    /// The ANE's `_ANEInMemoryModel` is a one-shot compile object — after compilation,
    /// it can only be loaded/unloaded/evaluated. The compiled weights are stored in an
    /// internal `data` file (~60x the size of raw weights) that `loadWithQoS` reads from.
    /// There is no way to force it to re-read from `weights/weight.bin`.
    ///
    /// Therefore, delta compilation works by:
    /// 1. Unloading the old model
    /// 2. Creating a new model descriptor with updated weights
    /// 3. Creating a new model from the descriptor
    /// 4. Compiling and loading the new model
    /// 5. Updating the kernel handle with the new model
    ///
    /// This is still faster than a full pipeline compile because the MIL graph
    /// is the same — only the weight data changes. The ANE's internal compiler
    /// can skip graph optimization and just regenerate the weight data.
    ///
    /// This increments the compile count.
    ///
    /// # Safety
    /// - kernel must be a valid, loaded ANEKernelHandle
    /// - weight_files must be (name, path) pairs matching MIL references
    /// - weight data must match expected sizes
    pub unsafe fn ane_bridge_reload_weights(
        kernel: *mut ANEKernelHandle,
        weight_files: &[(&str, &[u8])],
    ) -> bool {
        autoreleasepool(|_| {
            let Some(kernel) = (unsafe { kernel.as_ref() }) else {
                return false;
            };

            // Step 1: Unload the old model
            let mut error: *mut NSError = ptr::null_mut();
            let unloaded = unsafe { kernel.model.unloadWithQoS_error(ANE_QOS, &mut error) };
            if !unloaded.as_bool() {
                eprintln!("ane_bridge: unload failed: {}", ns_error_string(error));
                return false;
            }

            // Step 2: Get the MIL source from the stored copy in the kernel handle
            let mil_data = NSData::with_bytes(&kernel.mil_text);

            // Step 3: Create new weight dictionary with updated weights
            // Convert weight_files to C pointers for create_weight_dictionary
            let n_weights = weight_files.len() as i32;
            let mut c_names: Vec<*const c_char> = Vec::with_capacity(weight_files.len());
            let mut c_datas: Vec<*const u8> = Vec::with_capacity(weight_files.len());
            let mut c_lens: Vec<usize> = Vec::with_capacity(weight_files.len());

            // We need the name strings to live long enough
            let mut name_cstrings: Vec<CString> = Vec::with_capacity(weight_files.len());

            for (name, data) in weight_files {
                let clean_name = name.strip_prefix("@model_path/").unwrap_or(name);
                // Re-add @model_path/ prefix since write_model_files expects it
                let full_name = format!("@model_path/{}", clean_name);
                let c_name =
                    CString::new(full_name).unwrap_or_else(|_| CString::new(clean_name).unwrap());
                c_names.push(c_name.as_ptr());
                c_datas.push(data.as_ptr());
                c_lens.push(data.len());
                name_cstrings.push(c_name);
            }

            let weights = if n_weights > 0 {
                ANEPrivateRuntime::create_weight_dictionary(
                    c_names.as_mut_ptr(),
                    c_datas.as_mut_ptr(),
                    c_lens.as_ptr(),
                    n_weights,
                )
            } else {
                None
            };

            let empty_weights: Retained<NSDictionary<NSString, AnyObject>> =
                unsafe { msg_send![NSDictionary::<NSString, AnyObject>::class(), new] };

            // Step 4: Create new model descriptor and model
            let desc = unsafe {
                ANEInMemoryModelDescriptor::modelWithMILText_weights_optionsPlist(
                    &mil_data,
                    weights
                        .as_ref()
                        .map(|dict| &**dict as *const NSDictionary<NSString, AnyObject>)
                        .unwrap_or(&*empty_weights),
                    ptr::null::<AnyObject>(),
                )
            };
            let Some(desc) = desc else {
                eprintln!("ane_bridge: failed to create model descriptor for reload");
                return false;
            };

            let new_model = unsafe { ANEInMemoryModel::inMemoryModelWithDescriptor(&desc) };
            let Some(new_model) = new_model else {
                eprintln!("ane_bridge: failed to create new model for reload");
                return false;
            };

            // Step 5: Write new model files (same tmp_dir, updated weights)
            let Some(tmp_dir) = ANEPrivateRuntime::write_model_files(
                &new_model,
                &mil_data,
                c_names.as_mut_ptr(),
                c_datas.as_mut_ptr(),
                c_lens.as_ptr(),
                n_weights,
            ) else {
                eprintln!("ane_bridge: failed to write model files for reload");
                return false;
            };

            // Step 6: Compile the new model
            let options = empty_dictionary();
            let mut compile_error: *mut NSError = ptr::null_mut();
            let compiled = unsafe {
                new_model.compileWithQoS_options_error(ANE_QOS, &options, &mut compile_error)
            };
            if !compiled.as_bool() {
                eprintln!(
                    "ane_bridge: recompile failed: {}",
                    ns_error_string(compile_error)
                );
                let _ = fs::remove_dir_all(&tmp_dir);
                return false;
            }

            // Step 7: Load the new model
            let mut load_error: *mut NSError = ptr::null_mut();
            let loaded =
                unsafe { new_model.loadWithQoS_options_error(ANE_QOS, &options, &mut load_error) };
            if !loaded.as_bool() {
                thread::sleep(Duration::from_millis(100));
                let mut retry_error: *mut NSError = ptr::null_mut();
                let retry_loaded = unsafe {
                    new_model.loadWithQoS_options_error(ANE_QOS, &options, &mut retry_error)
                };
                if !retry_loaded.as_bool() {
                    eprintln!(
                        "ane_bridge: reload (load) failed: {}",
                        ns_error_string(retry_error)
                    );
                    let _ = fs::remove_dir_all(&tmp_dir);
                    return false;
                }
            }

            COMPILE_COUNT.fetch_add(1, Ordering::Relaxed);

            // Step 8: Rebuild the request and update the kernel handle
            let Some(new_request) =
                ANEPrivateRuntime::create_request(&kernel.io_inputs, &kernel.io_outputs)
            else {
                return false;
            };

            // Update the kernel handle with new model, request, and tmp_dir
            let kernel_ptr = kernel as *const ANEKernelHandle as *mut ANEKernelHandle;
            unsafe {
                (*kernel_ptr).model = new_model;
                (*kernel_ptr).request = new_request;
                (*kernel_ptr).tmp_dir = tmp_dir;
            }

            true
        })
    }

    fn cstr(bytes: &'static [u8]) -> &'static CStr {
        unsafe { CStr::from_bytes_with_nul_unchecked(bytes) }
    }

    fn ns_error_string(error: *mut NSError) -> String {
        if let Some(error) = unsafe { Retained::from_raw(error) } {
            let description: Retained<NSString> = unsafe { msg_send![&*error, description] };
            description.to_string()
        } else {
            "unknown".to_string()
        }
    }

    fn empty_dictionary() -> Retained<NSDictionary<AnyObject, AnyObject>> {
        unsafe { msg_send![NSDictionary::<AnyObject, AnyObject>::class(), new] }
    }
}

#[cfg(not(target_vendor = "apple"))]
mod apple {
    use core::ffi::{c_char, c_int, c_void};
    use core::ptr;

    pub enum ANEKernelHandle {}

    pub unsafe fn ane_bridge_init() -> c_int {
        -1
    }

    pub unsafe fn ane_bridge_compile(
        _mil_text: *const c_char,
        _mil_len: usize,
        _weight_data: *const u8,
        _weight_len: usize,
        _n_inputs: c_int,
        _input_sizes: *const usize,
        _n_outputs: c_int,
        _output_sizes: *const usize,
    ) -> *mut ANEKernelHandle {
        ptr::null_mut()
    }

    pub unsafe fn ane_bridge_compile_multi_weights(
        _mil_text: *const c_char,
        _mil_len: usize,
        _weight_names: *mut *const c_char,
        _weight_datas: *mut *const u8,
        _weight_lens: *const usize,
        _n_weights: c_int,
        _n_inputs: c_int,
        _input_sizes: *const usize,
        _n_outputs: c_int,
        _output_sizes: *const usize,
    ) -> *mut ANEKernelHandle {
        ptr::null_mut()
    }

    pub unsafe fn ane_bridge_eval(_kernel: *mut ANEKernelHandle) -> bool {
        false
    }

    pub unsafe fn ane_bridge_write_input(
        _kernel: *mut ANEKernelHandle,
        _idx: c_int,
        _data: *const c_void,
        _bytes: usize,
    ) {
    }

    pub unsafe fn ane_bridge_read_output(
        _kernel: *mut ANEKernelHandle,
        _idx: c_int,
        _data: *mut c_void,
        _bytes: usize,
    ) {
    }

    pub unsafe fn ane_bridge_free(_kernel: *mut ANEKernelHandle) {}

    pub unsafe fn ane_bridge_get_compile_count() -> c_int {
        0
    }

    pub unsafe fn ane_bridge_reset_compile_count() {}

    pub unsafe fn ane_bridge_reload_weights(
        _kernel: *mut ANEKernelHandle,
        _weight_files: &[(&str, &[u8])],
    ) -> bool {
        false
    }
}

pub(crate) use apple::*;

use std::collections::HashMap;

/// Request to compile MIL code into an ANE kernel
///
/// This struct encapsulates all the information needed to compile a MIL program
/// into an executable ANE kernel. It holds the MIL source code, weight data,
/// and tensor size specifications.
///
/// # Example
///
/// ```
/// use rustane::ane::ANECompileRequest;
/// use std::collections::HashMap;
///
/// let req = ANECompileRequest {
///     mil_text: "func main(x: (1, 1, 1, 16)) -> (1, 1, 1, 16) { return x }".to_string(),
///     weights: HashMap::new(),
///     input_sizes: vec![16],
///     output_sizes: vec![16],
/// };
/// ```
#[derive(Clone)]
pub struct ANECompileRequest {
    /// MIL program text (Model Intermediate Language)
    pub mil_text: String,

    /// Weight data indexed by variable name (as raw byte vectors)
    pub weights: HashMap<String, Vec<u8>>,

    /// Input tensor sizes in bytes
    pub input_sizes: Vec<usize>,

    /// Output tensor sizes in bytes
    pub output_sizes: Vec<usize>,
}

fn sort_named_weights(weights: HashMap<String, Vec<u8>>) -> Vec<(String, Vec<u8>)> {
    let mut sorted_weights: Vec<(String, Vec<u8>)> = weights.into_iter().collect();
    sorted_weights.sort_by(|(lhs, _), (rhs, _)| lhs.cmp(rhs));
    sorted_weights
}

impl ANECompileRequest {
    /// Create a new compile request with no weights.
    pub fn new(
        mil_text: impl Into<String>,
        input_sizes: impl Into<Vec<usize>>,
        output_sizes: impl Into<Vec<usize>>,
    ) -> Self {
        Self {
            mil_text: mil_text.into(),
            weights: HashMap::new(),
            input_sizes: input_sizes.into(),
            output_sizes: output_sizes.into(),
        }
    }

    /// Add a named weight payload from raw bytes.
    pub fn with_weight_bytes(mut self, name: impl Into<String>, data: impl Into<Vec<u8>>) -> Self {
        self.weights.insert(name.into(), data.into());
        self
    }

    /// Add a named ANE weight blob to the request.
    pub fn with_weight_blob(self, name: impl Into<String>, blob: &crate::ane::WeightBlob) -> Self {
        self.with_weight_bytes(name, blob.as_bytes().to_vec())
    }

    /// Extend the request with multiple named raw weight payloads.
    pub fn with_weights<I, N, D>(mut self, weights: I) -> Self
    where
        I: IntoIterator<Item = (N, D)>,
        N: Into<String>,
        D: Into<Vec<u8>>,
    {
        for (name, data) in weights {
            self.weights.insert(name.into(), data.into());
        }
        self
    }

    /// Compile MIL code and weights into an ANE kernel
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The ANE framework is not available
    /// - The MIL code is invalid
    /// - Compilation fails on the ANE
    pub fn compile(self) -> crate::Result<crate::wrapper::ANEExecutor> {
        let _runtime = crate::wrapper::ANERuntime::init()?;
        let mut compiler = crate::wrapper::ANECompiler::new();

        if self.weights.is_empty() {
            return compiler.compile_single(
                &self.mil_text,
                None,
                &self.input_sizes,
                &self.output_sizes,
            );
        }

        let sorted_weights = sort_named_weights(self.weights);
        let weight_names: Vec<&str> = sorted_weights
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        let weight_datas: Vec<&[u8]> = sorted_weights
            .iter()
            .map(|(_, data)| data.as_slice())
            .collect();
        let weight_lens: Vec<usize> = weight_datas.iter().map(|data| data.len()).collect();

        compiler.compile_multi(
            &self.mil_text,
            &weight_names,
            &weight_datas,
            &weight_lens,
            &self.input_sizes,
            &self.output_sizes,
        )
    }
}

/// Initialize the ANE runtime
///
/// Loads the private AppleNeuralEngine framework and resolves required classes.
/// This must be called before any other ANE operations.
///
/// # Errors
///
/// Returns an error if:
/// - Running on non-Apple Silicon hardware
/// - The ANE framework cannot be loaded
/// - Required private APIs are not available
///
/// # Example
///
/// ```no_run
/// use rustane::ane::runtime;
///
/// fn main() -> rustane::Result<()> {
///     runtime::ane_init()?;
///     // ANE operations can now be performed
///     Ok(())
/// }
/// ```
pub fn ane_init() -> crate::Result<()> {
    // Safety: ane_bridge_init is safe to call and returns 0 on success, -1 on failure
    let result = unsafe { apple::ane_bridge_init() };
    if result == 0 {
        Ok(())
    } else {
        Err(crate::Error::from(crate::ane::ANEError::FrameworkNotFound))
    }
}

/// Compile budget monitor for tracking ANE compilation usage
///
/// The ANE has a limit of ~119 compilations per process before memory issues.
/// This monitor tracks compilation count and provides warnings/budget info.
///
/// # Example
///
/// ```no_run
/// use rustane::ane::runtime::CompileBudgetMonitor;
///
/// let monitor = CompileBudgetMonitor::new();
/// assert!(monitor.check_budget(10), "Should have budget for 10 compiles");
///
/// // After compiling
/// let count = monitor.get_compile_count();
/// println!("Compiles used: {}/{}", count, monitor.budget_limit());
/// ```
pub struct CompileBudgetMonitor {
    budget_limit: i32,
    warning_threshold: i32,
}

impl CompileBudgetMonitor {
    /// ANE compile budget limit (from Orion research)
    pub const DEFAULT_BUDGET: i32 = 110;
    /// Warning threshold (90% of budget)
    pub const DEFAULT_WARNING_THRESHOLD: i32 = 99;

    /// Create a new budget monitor with default limits
    pub fn new() -> Self {
        Self {
            budget_limit: Self::DEFAULT_BUDGET,
            warning_threshold: Self::DEFAULT_WARNING_THRESHOLD,
        }
    }

    /// Create monitor with custom budget
    pub fn with_budget(budget: i32) -> Self {
        Self {
            budget_limit: budget,
            warning_threshold: (budget as f32 * 0.9) as i32,
        }
    }

    /// Get current compile count from ANE runtime
    pub fn get_compile_count(&self) -> i32 {
        unsafe { apple::ane_bridge_get_compile_count() }
    }

    /// Check if there's enough budget for N more compiles
    pub fn check_budget(&self, needed: i32) -> bool {
        let used = self.get_compile_count();
        used + needed <= self.budget_limit
    }

    /// Get remaining budget
    pub fn remaining(&self) -> i32 {
        let used = self.get_compile_count();
        (self.budget_limit - used).max(0)
    }

    /// Get budget limit
    pub fn budget_limit(&self) -> i32 {
        self.budget_limit
    }

    /// Check if we're in warning zone (90% of budget used)
    pub fn is_warning_zone(&self) -> bool {
        self.get_compile_count() >= self.warning_threshold
    }

    /// Check if budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.get_compile_count() >= self.budget_limit
    }

    /// Get budget status as a struct
    pub fn status(&self) -> BudgetStatus {
        let used = self.get_compile_count();
        BudgetStatus {
            used,
            limit: self.budget_limit,
            remaining: (self.budget_limit - used).max(0),
            percent_used: (used as f32 / self.budget_limit as f32 * 100.0).min(100.0),
            warning: self.is_warning_zone(),
            exhausted: self.is_exhausted(),
        }
    }
}

impl Default for CompileBudgetMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Budget status information
#[derive(Debug, Clone)]
pub struct BudgetStatus {
    /// Compiles used
    pub used: i32,
    /// Budget limit
    pub limit: i32,
    /// Remaining budget
    pub remaining: i32,
    /// Percentage used (0-100)
    pub percent_used: f32,
    /// In warning zone
    pub warning: bool,
    /// Budget exhausted
    pub exhausted: bool,
}

#[cfg(test)]
mod tests {
    use super::{sort_named_weights, ANECompileRequest};
    use crate::ane::WeightBlob;
    use std::collections::HashMap;

    #[test]
    fn test_sort_named_weights_orders_by_name() {
        let mut weights = HashMap::new();
        weights.insert("@model_path/weights/wk.bin".to_string(), vec![2u8; 2]);
        weights.insert("@model_path/weights/wq.bin".to_string(), vec![1u8; 3]);
        weights.insert("@model_path/weights/wv.bin".to_string(), vec![3u8; 1]);

        let sorted = sort_named_weights(weights);
        let names: Vec<&str> = sorted.iter().map(|(name, _)| name.as_str()).collect();

        assert_eq!(
            names,
            vec![
                "@model_path/weights/wk.bin",
                "@model_path/weights/wq.bin",
                "@model_path/weights/wv.bin",
            ]
        );
    }

    #[test]
    fn test_sort_named_weights_preserves_payloads() {
        let mut weights = HashMap::new();
        weights.insert("@model_path/weights/wq.bin".to_string(), vec![1u8, 2, 3]);
        weights.insert("@model_path/weights/wk.bin".to_string(), vec![4u8, 5]);

        let sorted = sort_named_weights(weights);

        assert_eq!(sorted[0].0, "@model_path/weights/wk.bin");
        assert_eq!(sorted[0].1, vec![4u8, 5]);
        assert_eq!(sorted[1].0, "@model_path/weights/wq.bin");
        assert_eq!(sorted[1].1, vec![1u8, 2, 3]);
    }

    #[test]
    fn test_compile_request_clone_preserves_weights() {
        let mut weights = HashMap::new();
        weights.insert("@model_path/weights/w.bin".to_string(), vec![7u8; 4]);
        let request = ANECompileRequest {
            mil_text: "program(1.0) { }".to_string(),
            weights,
            input_sizes: vec![16],
            output_sizes: vec![16],
        };

        let cloned = request.clone();
        assert_eq!(cloned.mil_text, request.mil_text);
        assert_eq!(cloned.input_sizes, request.input_sizes);
        assert_eq!(cloned.output_sizes, request.output_sizes);
        assert_eq!(
            cloned.weights.get("@model_path/weights/w.bin"),
            Some(&vec![7u8; 4])
        );
    }

    #[test]
    fn test_compile_request_builder_starts_empty() {
        let request = ANECompileRequest::new("program(1.0) { }", vec![16], vec![8]);

        assert_eq!(request.mil_text, "program(1.0) { }");
        assert!(request.weights.is_empty());
        assert_eq!(request.input_sizes, vec![16]);
        assert_eq!(request.output_sizes, vec![8]);
    }

    #[test]
    fn test_compile_request_with_weight_bytes() {
        let request = ANECompileRequest::new("program(1.0) { }", vec![16], vec![8])
            .with_weight_bytes("@model_path/weights/w.bin", vec![1u8, 2, 3]);

        assert_eq!(
            request.weights.get("@model_path/weights/w.bin"),
            Some(&vec![1u8, 2, 3])
        );
    }

    #[test]
    fn test_compile_request_with_weight_blob() {
        let blob = WeightBlob::from_f32(&[1.0f32, 2.0, 3.0, 4.0], 2, 2).unwrap();
        let request = ANECompileRequest::new("program(1.0) { }", vec![16], vec![8])
            .with_weight_blob("@model_path/weights/w.bin", &blob);

        assert_eq!(
            request.weights.get("@model_path/weights/w.bin"),
            Some(&blob.as_bytes().to_vec())
        );
    }

    #[test]
    fn test_compile_request_with_weights_overwrites_by_name() {
        let request = ANECompileRequest::new("program(1.0) { }", vec![16], vec![8]).with_weights([
            ("@model_path/weights/wq.bin", vec![1u8, 2]),
            ("@model_path/weights/wk.bin", vec![3u8]),
            ("@model_path/weights/wq.bin", vec![4u8, 5, 6]),
        ]);

        assert_eq!(request.weights.len(), 2);
        assert_eq!(
            request.weights.get("@model_path/weights/wq.bin"),
            Some(&vec![4u8, 5, 6])
        );
        assert_eq!(
            request.weights.get("@model_path/weights/wk.bin"),
            Some(&vec![3u8])
        );
    }

    #[test]
    fn test_compile_budget_monitor_defaults() {
        let monitor = super::CompileBudgetMonitor::new();
        assert_eq!(monitor.budget_limit(), 110);
        assert!(monitor.check_budget(10));
        assert!(!monitor.is_exhausted());
    }

    #[test]
    fn test_compile_budget_monitor_custom() {
        let monitor = super::CompileBudgetMonitor::with_budget(50);
        assert_eq!(monitor.budget_limit(), 50);
        assert!(monitor.check_budget(10));
        assert!(!monitor.check_budget(100));
    }

    #[test]
    fn test_compile_budget_monitor_status() {
        let monitor = super::CompileBudgetMonitor::new();
        let status = monitor.status();

        assert_eq!(status.limit, 110);
        assert!(status.percent_used >= 0.0);
        assert!(status.percent_used <= 100.0);
    }
}

#[cfg(target_vendor = "apple")]
mod apple {
    use core::ffi::{c_char, c_int, c_void, CStr};
    use core::ptr::{self, NonNull};
    use libc::{dlopen, RTLD_NOW};
    use objc2::rc::{autoreleasepool, Retained};
    use objc2::runtime::{AnyClass, AnyObject, Bool};
    use objc2::{class, msg_send};
    use objc2_core_foundation::CFRetained;
    use objc2_foundation::{NSArray, NSData, NSDictionary, NSError, NSNumber, NSString};
    use objc2_io_surface::IOSurfaceRef;
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
    const IOSURFACE_LOCK_READ_ONLY: u32 = 0x0000_0001;
    const ANE_QOS: u32 = 21;

    #[link(name = "IOSurface", kind = "framework")]
    unsafe extern "C" {
        fn IOSurfaceCreate(properties: *const c_void) -> *const IOSurfaceRef;
        fn IOSurfaceLock(buffer: &IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
        fn IOSurfaceUnlock(buffer: &IOSurfaceRef, options: u32, seed: *mut u32) -> i32;
        fn IOSurfaceGetBaseAddress(buffer: &IOSurfaceRef) -> *mut c_void;
    }

    struct RuntimeClasses {
        descriptor: &'static AnyClass,
        in_memory_model: &'static AnyClass,
        request: &'static AnyClass,
        io_surface_object: &'static AnyClass,
    }

    static CLASSES: OnceLock<Option<RuntimeClasses>> = OnceLock::new();
    static COMPILE_COUNT: AtomicI32 = AtomicI32::new(0);

    pub struct ANEKernelHandle {
        pub(crate) model: Retained<AnyObject>,
        pub(crate) io_inputs: Vec<CFRetained<IOSurfaceRef>>,
        pub(crate) io_outputs: Vec<CFRetained<IOSurfaceRef>>,
        pub(crate) request: Retained<AnyObject>,
        pub(crate) tmp_dir: PathBuf,
        pub(crate) n_inputs: i32,
        pub(crate) n_outputs: i32,
        pub(crate) input_bytes: Vec<usize>,
        pub(crate) output_bytes: Vec<usize>,
    }

    struct ANEPrivateRuntime;

    impl ANEPrivateRuntime {
        fn initialize() -> Option<&'static RuntimeClasses> {
            CLASSES
                .get_or_init(|| {
                    let handle = unsafe { dlopen(cstr(ANE_FRAMEWORK_PATH).as_ptr(), RTLD_NOW) };
                    if handle.is_null() {
                        return None;
                    }

                    Some(RuntimeClasses {
                        descriptor: AnyClass::get(cstr(CLASS_DESC))?,
                        in_memory_model: AnyClass::get(cstr(CLASS_INMEM))?,
                        request: AnyClass::get(cstr(CLASS_REQ))?,
                        io_surface_object: AnyClass::get(cstr(CLASS_IO))?,
                    })
                })
                .as_ref()
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

            let raw =
                unsafe { IOSurfaceCreate((&*props as *const NSDictionary<NSString, AnyObject>).cast()) };
            let raw = NonNull::new(raw as *mut IOSurfaceRef)?;
            Some(unsafe { CFRetained::from_raw(raw) })
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

                let name = unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned();
                let data = NSData::with_bytes(unsafe { slice::from_raw_parts(data_ptr, data_len) });

                let inner_keys = [NSString::from_str("offset"), NSString::from_str("data")];
                let inner_values = vec![NSNumber::new_u8(0).into(), data.into()];
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
                &key_refs,
                &values,
            ))
        }

        fn write_model_files(
            model: &AnyObject,
            mil_data: &NSData,
            weight_names: *mut *const c_char,
            weight_datas: *mut *const u8,
            weight_lens: *const usize,
            n_weights: i32,
        ) -> Option<PathBuf> {
            let hex_id: Retained<NSString> = unsafe { msg_send![model, hexStringIdentifier] };
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
            classes: &RuntimeClasses,
            io_inputs: &[CFRetained<IOSurfaceRef>],
            io_outputs: &[CFRetained<IOSurfaceRef>],
        ) -> Option<Retained<AnyObject>> {
            let mut input_objects = Vec::with_capacity(io_inputs.len());
            let mut input_indices = Vec::with_capacity(io_inputs.len());
            for (idx, surface) in io_inputs.iter().enumerate() {
                let obj: Option<Retained<AnyObject>> = unsafe {
                    msg_send![classes.io_surface_object, objectWithIOSurface: &**surface]
                };
                input_objects.push(obj?);
                input_indices.push(NSNumber::new_usize(idx));
            }

            let mut output_objects = Vec::with_capacity(io_outputs.len());
            let mut output_indices = Vec::with_capacity(io_outputs.len());
            for (idx, surface) in io_outputs.iter().enumerate() {
                let obj: Option<Retained<AnyObject>> = unsafe {
                    msg_send![classes.io_surface_object, objectWithIOSurface: &**surface]
                };
                output_objects.push(obj?);
                output_indices.push(NSNumber::new_usize(idx));
            }

            let inputs = NSArray::<AnyObject>::from_retained_slice(&input_objects);
            let input_ids = NSArray::<NSNumber>::from_retained_slice(&input_indices);
            let outputs = NSArray::<AnyObject>::from_retained_slice(&output_objects);
            let output_ids = NSArray::<NSNumber>::from_retained_slice(&output_indices);
            let procedure = NSNumber::new_u8(0);

            unsafe {
                msg_send![
                    classes.request,
                    requestWithInputs: &*inputs,
                    inputIndices: &*input_ids,
                    outputs: &*outputs,
                    outputIndices: &*output_ids,
                    weightsBuffer: ptr::null::<AnyObject>(),
                    perfStats: ptr::null::<AnyObject>(),
                    procedureIndex: &*procedure
                ]
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
            let Some(classes) = ANEPrivateRuntime::initialize() else {
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

            let desc: Option<Retained<AnyObject>> = unsafe {
                msg_send![
                    classes.descriptor,
                    modelWithMILText: &*mil_data,
                    weights: weights
                        .as_ref()
                        .map(|dict| &**dict as *const NSDictionary<NSString, AnyObject>)
                        .unwrap_or(ptr::null()),
                    optionsPlist: ptr::null::<AnyObject>()
                ]
            };
            let Some(desc) = desc else {
                return ptr::null_mut();
            };

            let model: Option<Retained<AnyObject>> =
                unsafe { msg_send![classes.in_memory_model, inMemoryModelWithDescriptor: &*desc] };
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
            let compiled: Bool = unsafe {
                msg_send![&*model, compileWithQoS: ANE_QOS, options: &*options, error: &mut error]
            };
            if !compiled.as_bool() {
                let _ = fs::remove_dir_all(&tmp_dir);
                eprintln!("ane_bridge: ANE compile failed: {}", ns_error_string(error));
                return ptr::null_mut();
            }

            let mut error: *mut NSError = ptr::null_mut();
            let mut loaded: Bool = unsafe {
                msg_send![&*model, loadWithQoS: ANE_QOS, options: &*options, error: &mut error]
            };
            if !loaded.as_bool() {
                thread::sleep(Duration::from_millis(100));
                let mut retry_error: *mut NSError = ptr::null_mut();
                loaded = unsafe {
                    msg_send![
                        &*model,
                        loadWithQoS: ANE_QOS,
                        options: &*options,
                        error: &mut retry_error
                    ]
                };
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
                (unsafe { ANEPrivateRuntime::create_request(classes, &io_inputs, &io_outputs) })
            else {
                let _ = fs::remove_dir_all(&tmp_dir);
                return ptr::null_mut();
            };

            COMPILE_COUNT.fetch_add(1, Ordering::Relaxed);

            Box::into_raw(Box::new(ANEKernelHandle {
                model,
                io_inputs,
                io_outputs,
                request,
                tmp_dir,
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
            let ok: Bool = unsafe {
                msg_send![
                    &*kernel.model,
                    evaluateWithQoS: ANE_QOS,
                    options: &*options,
                    request: &*kernel.request,
                    error: &mut error
                ]
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
            IOSurfaceLock(&**surface, 0, ptr::null_mut());
            ptr::copy_nonoverlapping(
                data.cast::<u8>(),
                IOSurfaceGetBaseAddress(&**surface).cast(),
                bytes,
            );
            IOSurfaceUnlock(&**surface, 0, ptr::null_mut());
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
            IOSurfaceLock(&**surface, IOSURFACE_LOCK_READ_ONLY, ptr::null_mut());
            ptr::copy_nonoverlapping(
                IOSurfaceGetBaseAddress(&**surface).cast::<u8>(),
                data.cast(),
                bytes,
            );
            IOSurfaceUnlock(&**surface, IOSURFACE_LOCK_READ_ONLY, ptr::null_mut());
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
            let _: Bool =
                unsafe { msg_send![&*kernel.model, unloadWithQoS: ANE_QOS, error: &mut error] };
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
        unsafe { msg_send![class!(NSDictionary), new] }
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

impl ANECompileRequest {
    /// Compile MIL code and weights into an ANE kernel
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The ANE framework is not available
    /// - The MIL code is invalid
    /// - Compilation fails on the ANE
    pub fn compile(self) -> crate::Result<crate::wrapper::ANEExecutor> {
        // TODO: Implement actual ANE compilation
        // This will use the low-level ane_bridge_compile FFI
        Err(crate::Error::Other("ANE compilation not yet implemented".to_string()))
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

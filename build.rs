use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=ANE_BRIDGE_LIB_PATH");
    println!("cargo:rerun-if-env-changed=ANE_BRIDGE_INCLUDE_PATH");

    // Locate the ANE bridge library
    let (lib_dir, lib_file) = find_ane_bridge_lib();
    println!("cargo:rustc-link-search={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=ane_bridge");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=IOSurface");

    // Copy library to target directory for runtime loading
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../../"); // Go up to target/debug or target/release
    let target_lib = target_dir.join(&lib_file);

    if !target_lib.exists() {
        fs::copy(&lib_dir.join(&lib_file), &target_lib)
            .expect("Failed to copy ANE bridge library to target directory");
        eprintln!("Copied {} to {:?}", lib_file, target_lib);
    }

    // Locate the ANE bridge header
    let include_path = find_ane_bridge_include();
    println!("cargo:rerun-if-changed={}", include_path.display());

    // Generate bindings
    generate_bindings(&include_path);
}

fn find_ane_bridge_lib() -> (PathBuf, String) {
    let lib_file = "libane_bridge.dylib";

    // Check environment variable first
    if let Ok(path) = env::var("ANE_BRIDGE_LIB_PATH") {
        let path = PathBuf::from(&path);
        if path.exists() {
            // Check if it's a directory or full path to the library
            if path.is_dir() {
                let lib_path = path.join(lib_file);
                if lib_path.exists() {
                    return (path, lib_file.to_string());
                }
            } else if path.ends_with(lib_file) {
                return (path.parent().unwrap().to_path_buf(), lib_file.to_string());
            }
        }
        panic!(
            "ANE_BRIDGE_LIB_PATH set but path does not exist: {:?}",
            path
        );
    }

    // Check default locations
    let default_locations = vec![
        // Relative to project root during development
        PathBuf::from("../ANE/bridge"),
        PathBuf::from("../../ANE/bridge"),
        PathBuf::from("~/dev/ANE/bridge"),
        // System install locations
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/opt/homebrew/lib"),
        PathBuf::from("/usr/lib"),
    ];

    for location in default_locations {
        if location.exists() {
            let lib_path = location.join(lib_file);
            if lib_path.exists() {
                eprintln!("Found ANE bridge library at: {:?}", lib_path);
                return (location, lib_file.to_string());
            }
        }
    }

    panic!(
        "Could not find {}. \
        Please build the ANE bridge library from ~/dev/ANE/bridge/ \
        or set ANE_BRIDGE_LIB_PATH environment variable.",
        lib_file
    );
}

fn find_ane_bridge_include() -> PathBuf {
    // Check environment variable first
    if let Ok(path_str) = env::var("ANE_BRIDGE_INCLUDE_PATH") {
        let path = PathBuf::from(&path_str);
        if path.exists() {
            let header_path = if path.is_dir() {
                path.join("ane_bridge.h")
            } else {
                path
            };
            if header_path.exists() {
                return header_path;
            }
        }
        panic!(
            "ANE_BRIDGE_INCLUDE_PATH set but path does not exist: {:?}",
            path_str
        );
    }

    // Check default locations
    let default_locations = vec![
        PathBuf::from("../ANE/bridge/ane_bridge.h"),
        PathBuf::from("../../ANE/bridge/ane_bridge.h"),
        PathBuf::from("~/dev/ANE/bridge/ane_bridge.h"),
        PathBuf::from("/usr/local/include/ane_bridge.h"),
        PathBuf::from("/opt/homebrew/include/ane_bridge.h"),
    ];

    for location in default_locations {
        if location.exists() {
            eprintln!("Found ANE bridge header at: {:?}", location);
            return location;
        }
    }

    panic!(
        "Could not find ane_bridge.h. \
        Please ensure the ANE bridge source is available \
        or set ANE_BRIDGE_INCLUDE_PATH environment variable."
    );
}

fn generate_bindings(header_path: &PathBuf) {
    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
        .allowlist_function("ane_bridge_.*")
        .allowlist_type("ANEKernelHandle")
        .opaque_type("ANEKernelHandle")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

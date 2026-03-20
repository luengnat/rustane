fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(target_vendor = "apple")]
    {
        println!("cargo:rustc-link-lib=framework=IOSurface");
    }
}

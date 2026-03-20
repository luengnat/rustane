#[test]
fn test_ane_error_to_rustane_error_conversion() {
    use rustane::ane::error::ANEError;
    use rustane::Error;

    let ane_err = ANEError::FrameworkNotFound;
    let rustane_err: Error = ane_err.into();

    match rustane_err {
        Error::Other(msg) => assert!(msg.contains("FrameworkNotFound")),
        _ => panic!("Wrong error type"),
    }
}

#[test]
fn test_ane_error_debug_output() {
    use rustane::ane::error::ANEError;

    let err = ANEError::CompileFailed("test compilation".to_string());
    let msg = format!("{:?}", err);
    assert!(msg.contains("CompileFailed"));
    assert!(msg.contains("test compilation"));
}

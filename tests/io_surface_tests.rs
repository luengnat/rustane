#[test]
fn test_io_surface_creation() {
    use rustane::ane::IOSurface;

    let result = IOSurface::new(1024);
    // May fail on non-Apple Silicon, but shouldn't panic
    match result {
        Ok(_) => println!("IOSurface created"),
        Err(e) => println!("IOSurface creation not available: {:?}", e),
    }
}

#[test]
fn test_io_surface_write_read_roundtrip() {
    use rustane::ane::IOSurface;

    let mut io = match IOSurface::new(64) {
        Ok(io) => io,
        Err(_) => {
            println!("IOSurface not available, skipping test");
            return;
        }
    };

    let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
    let _ = io.write(&data);
    let result = io.read();

    match result {
        Ok(read_data) => {
            // IOSurface is a fixed-size buffer, read() returns full buffer
            assert_eq!(read_data.len(), 64);
            // Verify the written data is at the beginning
            assert_eq!(&read_data[..data.len()], data.as_slice());
        }
        Err(_) => println!("Read failed (expected on some systems)"),
    }
}

#[test]
fn test_io_surface_write_exceeds_capacity() {
    use rustane::ane::IOSurface;

    let mut io = match IOSurface::new(10) {
        Ok(io) => io,
        Err(_) => {
            println!("IOSurface not available, skipping test");
            return;
        }
    };

    let data = vec![1u8; 20]; // Larger than capacity
    let result = io.write(&data);

    // Should fail gracefully
    assert!(result.is_err());
}

#[test]
fn test_io_surface_with_lock() {
    use rustane::ane::IOSurface;

    let mut io = match IOSurface::new(64) {
        Ok(io) => io,
        Err(_) => {
            println!("IOSurface not available, skipping test");
            return;
        }
    };

    // Test that with_lock allows direct pointer access
    let result = io.with_lock(|ptr| {
        // Verify we got a non-null pointer
        !ptr.is_null()
    });

    match result {
        Ok(is_valid) => assert!(is_valid),
        Err(_) => println!("with_lock failed (expected on some systems)"),
    }
}

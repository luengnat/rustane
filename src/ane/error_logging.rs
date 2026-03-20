//! Structured Error Logging and Reporting
//!
//! Provides structured logging infrastructure for ANE errors with
//! severity levels, context capture, and error aggregation.

use crate::ane::error_diagnostics::{ErrorDiagnostic, ErrorCategory};
use std::fmt::Write as FmtWrite;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Error severity levels for logging
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Debug information
    Debug = 0,
    /// Informational message
    Info = 1,
    /// Warning - operation succeeded but with issues
    Warning = 2,
    /// Error - operation failed
    Error = 3,
    /// Critical error - training cannot continue
    Critical = 4,
}

impl ErrorSeverity {
    /// Get severity name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warning => "WARNING",
            Self::Error => "ERROR",
            Self::Critical => "CRITICAL",
        }
    }

    /// Get severity as emoji for display
    pub fn emoji(&self) -> &'static str {
        match self {
            Self::Debug => "🔍",
            Self::Info => "ℹ️",
            Self::Warning => "⚠️",
            Self::Error => "❌",
            Self::Critical => "🚨",
        }
    }

    /// Map error category to severity
    pub fn from_category(category: ErrorCategory) -> Self {
        match category {
            ErrorCategory::Configuration => Self::Critical,
            ErrorCategory::Resource => Self::Error,
            ErrorCategory::Compilation => Self::Error,
            ErrorCategory::Data => Self::Warning,
            ErrorCategory::Runtime => Self::Warning,
        }
    }
}

/// Single error log entry
#[derive(Debug, Clone)]
pub struct ErrorLogEntry {
    /// Timestamp when error occurred
    pub timestamp: SystemTime,
    /// Error severity
    pub severity: ErrorSeverity,
    /// Error category
    pub category: ErrorCategory,
    /// Human-readable description
    pub description: String,
    /// Operation that failed
    pub operation: Option<String>,
    /// Layer index (if applicable)
    pub layer_idx: Option<usize>,
    /// Full error message
    pub error_message: String,
}

impl ErrorLogEntry {
    /// Create new log entry from diagnostic
    pub fn from_diagnostic(diag: &ErrorDiagnostic, operation: Option<&str>) -> Self {
        let severity = ErrorSeverity::from_category(diag.category);
        let timestamp = SystemTime::now();

        Self {
            timestamp,
            severity,
            category: diag.category,
            description: diag.root_cause.clone(),
            operation: operation.map(|s| s.to_string()),
            layer_idx: diag.context.get("layer_idx").and_then(|s| s.parse().ok()),
            error_message: diag.error.to_string(),
        }
    }

    /// Format as single-line log message
    pub fn format_short(&self) -> String {
        let _timestamp_ms = self
            .timestamp
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);

        let op_str = self.operation.as_deref().unwrap_or("unknown");
        let layer_str = self.layer_idx.map(|l| format!("[layer {}] ", l)).unwrap_or_default();

        format!(
            "{} [{:?}] {}{} - {}",
            self.severity.emoji(),
            self.category,
            layer_str,
            op_str,
            self.description
        )
    }

    /// Format as detailed log message
    pub fn format_detailed(&self) -> String {
        let mut output = String::new();

        writeln!(output, "{} {}", self.severity.emoji(), self.severity.name()).ok();
        writeln!(output, "  Category: {:?}", self.category).ok();

        if let Some(ref operation) = self.operation {
            writeln!(output, "  Operation: {}", operation).ok();
        }

        if let Some(layer_idx) = self.layer_idx {
            writeln!(output, "  Layer: {}", layer_idx).ok();
        }

        writeln!(output, "  Description: {}", self.description).ok();
        writeln!(output, "  Error: {}", self.error_message).ok();

        output
    }
}

/// Structured error log
#[derive(Debug, Clone, Default)]
pub struct ErrorLog {
    entries: Vec<ErrorLogEntry>,
}

impl ErrorLog {
    /// Create new empty log
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add entry to log
    pub fn log(&mut self, entry: ErrorLogEntry) {
        self.entries.push(entry);
    }

    /// Add diagnostic to log
    pub fn log_diagnostic(&mut self, diag: &ErrorDiagnostic, operation: Option<&str>) {
        let entry = ErrorLogEntry::from_diagnostic(diag, operation);
        self.log(entry);
    }

    /// Get all entries
    pub fn entries(&self) -> &[ErrorLogEntry] {
        &self.entries
    }

    /// Get entry count by severity
    pub fn count_by_severity(&self, severity: ErrorSeverity) -> usize {
        self.entries.iter().filter(|e| e.severity == severity).count()
    }

    /// Get entry count by category
    pub fn count_by_category(&self, category: ErrorCategory) -> usize {
        self.entries.iter().filter(|e| e.category == category).count()
    }

    /// Check if log has any critical errors
    pub fn has_critical_errors(&self) -> bool {
        self.entries.iter().any(|e| e.severity == ErrorSeverity::Critical)
    }

    /// Get most recent n entries
    pub fn recent(&self, n: usize) -> &[ErrorLogEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Format summary report
    pub fn format_summary(&self) -> String {
        if self.entries.is_empty() {
            return "No errors logged".to_string();
        }

        let mut output = String::new();

        writeln!(output, "Error Log Summary").ok();
        writeln!(output, "================").ok();
        writeln!(output, "Total errors: {}", self.entries.len()).ok();

        // By severity
        writeln!(output).ok();
        writeln!(output, "By Severity:").ok();
        for severity in [
            ErrorSeverity::Critical,
            ErrorSeverity::Error,
            ErrorSeverity::Warning,
            ErrorSeverity::Info,
            ErrorSeverity::Debug,
        ] {
            let count = self.count_by_severity(severity);
            if count > 0 {
                writeln!(output, "  {}: {}", severity.name(), count).ok();
            }
        }

        // By category
        writeln!(output).ok();
        writeln!(output, "By Category:").ok();
        for category in [
            ErrorCategory::Configuration,
            ErrorCategory::Resource,
            ErrorCategory::Compilation,
            ErrorCategory::Data,
            ErrorCategory::Runtime,
        ] {
            let count = self.count_by_category(category);
            if count > 0 {
                writeln!(output, "  {}: {}", category.name(), count).ok();
            }
        }

        // Recent errors
        if !self.entries.is_empty() {
            writeln!(output).ok();
            writeln!(output, "Recent Errors (last 5):").ok();
            for entry in self.recent(5) {
                writeln!(output, "  {}", entry.format_short()).ok();
            }
        }

        output
    }
}

/// Error reporter with configurable output
pub struct ErrorReporter {
    log: ErrorLog,
    verbose: bool,
    output: Box<dyn Write + Send>,
}

impl ErrorReporter {
    /// Create new error reporter
    pub fn new(verbose: bool) -> Self {
        Self {
            log: ErrorLog::new(),
            verbose,
            output: Box::new(std::io::stdout()),
        }
    }

    /// Create with custom output
    pub fn with_output<W: Write + Send + 'static>(verbose: bool, output: W) -> Self {
        Self {
            log: ErrorLog::new(),
            verbose,
            output: Box::new(output),
        }
    }

    /// Report an error
    pub fn report(&mut self, diag: &ErrorDiagnostic, operation: Option<&str>) {
        self.log.log_diagnostic(diag, operation);

        let entry = ErrorLogEntry::from_diagnostic(diag, operation);

        if self.verbose || entry.severity >= ErrorSeverity::Warning {
            let _ = writeln!(self.output, "{}", entry.format_short());

            if entry.severity >= ErrorSeverity::Error || self.verbose {
                let _ = writeln!(self.output, "{}", entry.format_detailed());
            }
        }
    }

    /// Get log reference
    pub fn log(&self) -> &ErrorLog {
        &self.log
    }

    /// Get mutable log reference
    pub fn log_mut(&mut self) -> &mut ErrorLog {
        &mut self.log
    }

    /// Print summary report
    pub fn print_summary(&mut self) {
        let summary = self.log.format_summary();
        let _ = writeln!(self.output, "{}", summary);
    }

    /// Check if any critical errors were logged
    pub fn has_critical_errors(&self) -> bool {
        self.log.has_critical_errors()
    }
}

impl Default for ErrorReporter {
    fn default() -> Self {
        Self::new(false)
    }
}

/// Global error reporter (lazy static)
static GLOBAL_REPORTER: std::sync::Mutex<Option<ErrorReporter>> = std::sync::Mutex::new(None);

/// Initialize global error reporter
pub fn init_global_reporter(verbose: bool) {
    *GLOBAL_REPORTER.lock().unwrap() = Some(ErrorReporter::new(verbose));
}

/// Report error to global reporter
pub fn report_error(diag: &ErrorDiagnostic, operation: Option<&str>) {
    if let Some(ref mut reporter) = *GLOBAL_REPORTER.lock().unwrap() {
        reporter.report(diag, operation);
    }
}

/// Print global error summary
pub fn print_global_summary() {
    if let Some(ref mut reporter) = *GLOBAL_REPORTER.lock().unwrap() {
        reporter.print_summary();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ane::ANEError;

    #[test]
    fn test_error_severity() {
        assert_eq!(ErrorSeverity::Debug.name(), "DEBUG");
        assert_eq!(ErrorSeverity::Error.emoji(), "❌");
        assert!(ErrorSeverity::Critical > ErrorSeverity::Error);
    }

    #[test]
    fn test_error_log() {
        let mut log = ErrorLog::new();
        assert_eq!(log.entries().len(), 0);
        assert!(!log.has_critical_errors());

        let entry = ErrorLogEntry {
            timestamp: SystemTime::now(),
            severity: ErrorSeverity::Warning,
            category: ErrorCategory::Runtime,
            description: "test".to_string(),
            operation: Some("test_op".to_string()),
            layer_idx: Some(0),
            error_message: "test error".to_string(),
        };

        log.log(entry);
        assert_eq!(log.entries().len(), 1);
    }

    #[test]
    fn test_error_reporter() {
        let mut reporter = ErrorReporter::new(false);
        let error = ANEError::EvalFailed("test".to_string());
        let diag = ErrorDiagnostic::from_error(error);

        reporter.report(&diag, Some("test_op"));
        assert_eq!(reporter.log().entries().len(), 1);
    }
}

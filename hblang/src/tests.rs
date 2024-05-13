#![cfg(test)]

pub fn run_test(name: &'static str, input: &'static str, test: fn(&'static str, &mut String)) {
    use std::{io::Write, path::PathBuf};

    let filter = std::env::var("PT_FILTER").unwrap_or_default();
    if !filter.is_empty() && !name.contains(&filter) {
        return;
    }

    let mut output = String::new();
    test(input, &mut output);

    let mut root = PathBuf::from(std::env::var("PT_TEST_ROOT").unwrap_or("tests".to_string()));
    root.push(
        name.replace("::", "_")
            .replace(concat!(env!("CARGO_PKG_NAME"), "_"), ""),
    );
    root.set_extension("txt");

    let expected = std::fs::read_to_string(&root).unwrap_or_default();

    if output == expected {
        return;
    }

    if std::env::var("PT_UPDATE").is_ok() {
        std::fs::write(&root, output).unwrap();
        return;
    }

    if !root.exists() {
        std::fs::create_dir_all(root.parent().unwrap()).unwrap();
        std::fs::write(&root, vec![]).unwrap();
    }

    let mut proc = std::process::Command::new("diff")
        .arg("-u")
        .arg("--color")
        .arg(&root)
        .arg("-")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::inherit())
        .spawn()
        .unwrap();

    proc.stdin
        .as_mut()
        .unwrap()
        .write_all(output.as_bytes())
        .unwrap();

    proc.wait().unwrap();

    panic!();
}

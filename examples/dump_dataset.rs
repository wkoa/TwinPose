// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
// Developer fixture exporter for native C++/Rust differential validation.
fn main() -> twinpose::Result<()> {
    let path = std::env::args().nth(1).ok_or("dataset.json required")?;
    let path = std::path::Path::new(&path);
    let c: twinpose::dataset::Config = serde_json::from_value(twinpose::dataset::json(path)?)?;
    let data = twinpose::dataset::load(path, &c)?;
    use std::io::Write;
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for (frame_id, views) in data.frames {
        serde_json::to_writer(
            &mut out,
            &serde_json::json!({"frame_id":frame_id,"views":views}),
        )?;
        writeln!(out)?;
    }
    Ok(())
}

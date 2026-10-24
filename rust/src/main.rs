// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use serde::{Deserialize, Serialize};
use std::{
    io::{BufRead, BufReader, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
use twinpose::{
    Result,
    dataset::{self, Config},
    evaluation,
    model::*,
    pose::{Params, Tracker},
};
#[derive(Serialize)]
struct Output {
    frame_id: usize,
    poses: MultiPose3D,
    ground_truth: MultiPose3D,
    metrics: std::collections::BTreeMap<usize, evaluation::Metrics>,
    algorithm_ms: f64,
    rejected_geometry: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    frame_id: usize,
    views: MultiView,
}
fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "help".into());
    if command == "help" || command == "--help" || command == "-h" {
        println!(
            "TwinPose Rust\n\n  twinpose run --dataset <dataset.json> --algorithm <algorithm.json> --output <new.jsonl> [--limit N]\n  twinpose stream --algorithm <algorithm.json>  < frames.jsonl > poses.jsonl\n\nrun loads upstream dataset configurations. stream reads calibrated multi-view frames,\nwith strictly increasing frame_id; missing observations are never fabricated.\nOutput poses use dataset camera coordinates and units, without a display remap.\nNo detectors/models are downloaded. Errors go to stderr with nonzero exit status."
        );
        return Ok(());
    }
    let mut dataset_path = None;
    let mut algorithm_path = None;
    let mut output_path = None;
    let mut limit = None;
    while let Some(arg) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--dataset" => dataset_path = Some(PathBuf::from(value)),
            "--algorithm" => algorithm_path = Some(PathBuf::from(value)),
            "--output" => output_path = Some(PathBuf::from(value)),
            "--limit" => {
                let n: usize = value.parse()?;
                if n == 0 {
                    return Err("limit must be positive".into());
                }
                limit = Some(n)
            }
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    let algorithm_path = algorithm_path.ok_or("--algorithm required")?;
    let mut params: Params = serde_json::from_value(dataset::json(&algorithm_path)?)?;
    match command.as_str() {
        "run" => {
            let path = dataset_path.ok_or("--dataset required")?;
            let c: Config = serde_json::from_value(dataset::json(&path)?)?;
            if params.max_person_num == 0 {
                params.max_person_num = c.detection_max_person_num;
            }
            let mut tracker = Tracker::new(params)?;
            let data = dataset::load(&path, &c)?;
            let out = output_path.ok_or("--output required")?;
            // Refuse to truncate an existing result or raw source file.
            let file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&out)?;
            let mut writer = BufWriter::new(file);
            let mut count = 0;
            for ((frame_id, views), gt) in data
                .frames
                .into_iter()
                .zip(data.ground_truth)
                .take(limit.unwrap_or(usize::MAX))
            {
                let start = Instant::now();
                let (poses, debug) = tracker.process_corrected(&views, &c.result_correction)?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.;
                let metrics = gt
                    .iter()
                    .map(|(&id, pose)| (id, evaluation::evaluate(pose, &poses)))
                    .collect();
                serde_json::to_writer(
                    &mut writer,
                    &Output {
                        frame_id,
                        poses,
                        ground_truth: gt,
                        metrics,
                        algorithm_ms: elapsed,
                        rejected_geometry: debug.rejected_geometry,
                    },
                )?;
                writeln!(writer)?;
                count += 1;
            }
            writer.flush()?;
            eprintln!("completed {count} frames: {}", out.display());
        }
        "stream" => {
            if dataset_path.is_some() || output_path.is_some() || limit.is_some() {
                return Err("stream accepts only --algorithm".into());
            }
            let mut tracker = Tracker::new(params)?;
            let mut last = None;
            let stdin = std::io::stdin();
            let mut stdout = BufWriter::new(std::io::stdout().lock());
            for line in BufReader::new(stdin.lock()).lines() {
                let line = line?;
                let input: Input = serde_json::from_str(&line)?;
                if last.is_some_and(|v| input.frame_id <= v) {
                    return Err(
                        "frame_id must strictly increase; start a new stream to reset history"
                            .into(),
                    );
                }
                let start = Instant::now();
                let (poses, debug) = tracker.process(&input.views)?;
                serde_json::to_writer(
                    &mut stdout,
                    &Output {
                        frame_id: input.frame_id,
                        poses,
                        ground_truth: MultiPose3D::new(),
                        metrics: Default::default(),
                        algorithm_ms: start.elapsed().as_secs_f64() * 1000.,
                        rejected_geometry: debug.rejected_geometry,
                    },
                )?;
                writeln!(stdout)?;
                stdout.flush()?;
                last = Some(input.frame_id);
            }
        }
        _ => return Err(format!("unknown command {command}").into()),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("TwinPose: {e}");
        std::process::exit(1);
    }
}

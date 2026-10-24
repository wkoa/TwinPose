// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};
use twinpose::dataset::*;
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let p = std::env::temp_dir().join(format!(
            "twinpose-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
    fn write(&self, name: &str, value: Value) -> PathBuf {
        let p = self.0.join(name);
        fs::write(&p, serde_json::to_vec(&value).unwrap()).unwrap();
        p
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn frame(id: usize) -> Value {
    json!({"frame_id":id,"views":[{"camera":{"k":[[1,0,0],[0,1,0],[0,0,1]],"r":[[1,0,0],[0,1,0],[0,0,1]],"t":[0,0,0]},"joints":{}},{"camera":{"k":[[1,0,0],[0,1,0],[0,0,1]],"r":[[1,0,0],[0,1,0],[0,0,1]],"t":[-1,0,0]},"joints":{}}]})
}
fn cli(args: &[&str], input: &str) -> std::process::Output {
    let mut process = Command::new(env!("CARGO_BIN_EXE_twinpose"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    process
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    process.wait_with_output().unwrap()
}
#[test]
fn stream_keeps_identity_and_empty_frame() {
    let t = Temp::new();
    let p = t.write("algorithm.json", json!({}));
    let output = cli(
        &["stream", "--algorithm", p.to_str().unwrap()],
        &format!("{}\n{}\n", frame(12), frame(14)),
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines: String = String::from_utf8(output.stdout).unwrap();
    let rows: Vec<Value> = lines
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["frame_id"], 12);
    assert_eq!(rows[1]["frame_id"], 14);
    assert_eq!(rows[0]["poses"], json!({}));
}
#[test]
fn stream_rejects_reordered_frame_without_fake_output() {
    let t = Temp::new();
    let p = t.write("algorithm.json", json!({}));
    let o = cli(
        &["stream", "--algorithm", p.to_str().unwrap()],
        &format!("{}\n{}\n", frame(12), frame(11)),
    );
    assert!(!o.status.success());
    assert_eq!(String::from_utf8(o.stdout).unwrap().lines().count(), 1);
    assert!(
        String::from_utf8(o.stderr)
            .unwrap()
            .contains("strictly increase")
    );
}
#[test]
fn stream_rejects_truncated_json_and_singular_camera() {
    let t = Temp::new();
    let p = t.write("algorithm.json", json!({}));
    for text in ["{\"frame_id\":1".to_owned(), {
        let mut v = frame(0);
        v["views"][0]["camera"]["k"] = json!([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
        v.to_string()
    }] {
        let o = cli(
            &["stream", "--algorithm", p.to_str().unwrap()],
            &(text + "\n"),
        );
        assert!(!o.status.success());
        assert!(o.stdout.is_empty());
    }
}
#[test]
fn cli_does_not_overwrite_output_or_raw_input() {
    let t = Temp::new();
    let output = t.0.join("output.jsonl");
    fs::write(&output, "preserve me").unwrap();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let o = cli(
        &[
            "run",
            "--dataset",
            root.join("dataset/shelf/dataset.json").to_str().unwrap(),
            "--algorithm",
            root.join("dataset/shelf/algorithm.json").to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
            "--limit",
            "1",
        ],
        "",
    );
    assert!(!o.status.success());
    assert_eq!(fs::read_to_string(output).unwrap(), "preserve me");
}
#[test]
fn cli_rejects_unknown_options_and_configs() {
    let t = Temp::new();
    let p = t.write("bad.json", json!({"pretend_parameter":true}));
    for args in [
        vec!["stream", "--algorithm", p.to_str().unwrap()],
        vec!["run", "--limit", "0"],
        vec!["run", "--unknown", "a"],
    ] {
        let o = cli(&args, "");
        assert!(!o.status.success());
        assert!(o.stdout.is_empty());
    }
}
fn mm(ids: &[usize]) -> Value {
    json!({"instance_info":ids.iter().map(|&id|json!({"frame_id":id,"instances":[]})).collect::<Vec<_>>()})
}
#[test]
fn mmpose_rejects_missing_or_duplicate_cross_camera_frame() {
    let t = Temp::new();
    let c = Config {
        detection_view_num: 2,
        detection_frame_range: [1, 3],
        ..Default::default()
    };
    t.write("0.json", mm(&[1, 2, 3]));
    t.write("1.json", mm(&[1, 3]));
    assert!(
        load_detections(&c, &t.0)
            .unwrap_err()
            .to_string()
            .contains("differ")
    );
    t.write("1.json", mm(&[1, 1, 3]));
    assert!(load_detections(&c, &t.0).is_err());
}
#[test]
fn mmpose_does_not_fill_a_missing_camera() {
    let t = Temp::new();
    let c = Config {
        detection_view_num: 2,
        detection_frame_range: [1, 3],
        ..Default::default()
    };
    t.write("0.json", mm(&[1, 2, 3]));
    assert!(load_detections(&c, &t.0).is_err());
}
#[test]
fn readers_reject_truncated_ground_truth_and_invalid_camera_count() {
    let t = Temp::new();
    let p = t.0.join("gt.txt");
    fs::write(&p, "2 1 0.1").unwrap();
    let c = Config {
        ground_truth_loader: "4DA".into(),
        ground_truth_joint_type_num: 21,
        ..Default::default()
    };
    assert!(load_ground_truth(&c, &p).is_err());
    let p = t.write("cameras.json", json!({}));
    assert!(load_cameras(&p, "4DA", 2).is_err());
}
#[test]
fn openpose_and_osd_preserve_frame_and_candidate_order() {
    let t = Temp::new();
    let c = Config {
        detection_loader: "OpenPose".into(),
        detection_view_num: 2,
        detection_frame_range: [9, 9],
        detection_joint_type_num: 1,
        ..Default::default()
    };
    for v in 0..2 {
        t.write(
            &format!("{v}_000000000009_keypoints.json"),
            json!({"people":[{"pose_keypoints_2d":[10,20,0.9]}]}),
        );
    }
    let f = load_detections(&c, &t.0).unwrap();
    assert_eq!(f[0].0, 9);
    assert_eq!(f[0].1[0].joints[&0][0].point, [10., 20.]);
    let p = t.0.join("osd.txt");
    fs::write(
        &p,
        "frame 9\nview 0\np 1 10 20 0.9\nview 1\np 1 11 21 0.8\n",
    )
    .unwrap();
    let c = Config {
        detection_loader: "OSD".into(),
        ..c
    };
    let f = load_detections(&c, &p).unwrap();
    assert_eq!(f[0].0, 9);
    assert_eq!(f[0].1[1].joints[&0][0].point, [11., 21.]);
}
#[test]
fn four_da_text_candidates_and_paf_are_parsed() {
    let t = Temp::new();
    for v in 0..2 {
        fs::write(
            t.0.join(format!("{v}.txt")),
            format!("10\n{}", "1 0.5 0.25 0.8\n".repeat(10)),
        )
        .unwrap();
    }
    let c = Config {
        detection_loader: "4DA".into(),
        detection_skel_type: "Body25".into(),
        detection_view_num: 2,
        detection_joint_type_num: 1,
        detection_image_width: 101,
        detection_image_height: 201,
        ..Default::default()
    };
    let frames = load_detections(&c, &t.0).unwrap();
    assert_eq!(frames.len(), 10);
    assert_eq!(frames[0].1[0].joints[&0][0].point, [50., 50.]);
}
#[test]
fn campus_and_osd_camera_readers_match_projection() {
    let t = Temp::new();
    let campus = t.0.join("campus.txt");
    fs::write(
        &campus,
        "1 0 0 -1 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n",
    )
    .unwrap();
    let a = load_cameras(&campus, "Campus", 2).unwrap();
    let osd = t.0.join("osd.txt");
    fs::write(
        &osd,
        "1 0 0 0 1 0 0 0 1 1 0 0 0 1 0 0 0 1 -1 0 0\n1 0 0 0 1 0 0 0 1 1 0 0 0 1 0 0 0 1 1 0 0\n",
    )
    .unwrap();
    let b = load_cameras(&osd, "OSD", 2).unwrap();
    assert_eq!(a[0].t, b[0].t);
    assert_eq!(a[1].position(), b[1].position());
}
#[test]
fn ground_truth_readers_keep_explicit_ids_and_invalid_joints_absent() {
    let t = Temp::new();
    let p = t.0.join("shelf.txt");
    fs::write(&p, "2 1 1 2 1 2 3 4 5 6 1 0").unwrap();
    let c = Config {
        ground_truth_loader: "4DA Shelf".into(),
        ground_truth_skel_type: "Body25".into(),
        ground_truth_max_person_num: 3,
        ..Default::default()
    };
    let frames = load_ground_truth(&c, &p).unwrap();
    assert_eq!(frames[0][&2].len(), 1);
    assert_eq!(frames[0][&2][&0].0, [1., 3., 5.]);
    let p = t.0.join("campus.txt");
    fs::write(&p, "frame 0\np 1 1 2 3\n").unwrap();
    let c = Config {
        ground_truth_loader: "Campus".into(),
        ground_truth_joint_type_num: 1,
        ..c
    };
    let frames = load_ground_truth(&c, &p).unwrap();
    assert_eq!(frames[0][&0][&0].0, [1., 2., 3.]);
}
#[test]
fn panoptic_missing_file_is_an_empty_annotation_frame() {
    let t = Temp::new();
    t.write(
        "body3DScene_00000001.json",
        json!({"bodies":[{"id":7,"joints19":vec![1.;76]}]}),
    );
    let c = Config {
        ground_truth_loader: "Panoptic".into(),
        ground_truth_skel_type: "Coco19".into(),
        ground_truth_frame_range: [1, 2],
        ..Default::default()
    };
    let frames = load_ground_truth(&c, &t.0).unwrap();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0][&7].len(), 19);
    assert!(frames[1].is_empty());
}

// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use crate::{Result, geometry::*, model::*};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub camera_loader: String,
    pub camera_load_path: PathBuf,
    pub detection_loader: String,
    pub detection_skel_type: String,
    pub detection_load_path: PathBuf,
    pub detection_joint_type_num: usize,
    pub detection_view_num: usize,
    pub detection_max_person_num: usize,
    pub detection_frame_range: [usize; 2],
    pub detection_image_width: usize,
    pub detection_image_height: usize,
    pub detection_joint_2d_scale: f32,
    pub ground_truth_loader: String,
    pub ground_truth_skel_type: String,
    pub ground_truth_load_path: PathBuf,
    pub ground_truth_joint_type_num: usize,
    pub ground_truth_max_person_num: usize,
    pub ground_truth_frame_range: [usize; 2],
    pub sync_points_path: PathBuf,
    pub result_correction: String,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            camera_loader: "4DA".into(),
            camera_load_path: PathBuf::new(),
            detection_loader: "MMPose".into(),
            detection_skel_type: "Coco17".into(),
            detection_load_path: PathBuf::new(),
            detection_joint_type_num: 17,
            detection_view_num: 5,
            detection_max_person_num: 4,
            detection_frame_range: [300, 600],
            detection_image_width: 1032,
            detection_image_height: 776,
            detection_joint_2d_scale: 1.,
            ground_truth_loader: "None".into(),
            ground_truth_skel_type: "Body25".into(),
            ground_truth_load_path: PathBuf::new(),
            ground_truth_joint_type_num: 25,
            ground_truth_max_person_num: 3,
            ground_truth_frame_range: [8500, 10500],
            sync_points_path: PathBuf::new(),
            result_correction: "None".into(),
        }
    }
}
pub struct Dataset {
    pub frames: Vec<(usize, MultiView)>,
    pub ground_truth: Vec<MultiPose3D>,
}
pub fn json(path: &Path) -> Result<Value> {
    let f = fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(serde_json::from_reader(std::io::BufReader::new(f))?)
}
fn number(v: &Value) -> Result<f32> {
    let n = v.as_f64().ok_or("expected number")? as f32;
    if !n.is_finite() {
        return Err("non-finite number".into());
    }
    Ok(n)
}
fn array(v: &Value) -> Result<&Vec<Value>> {
    Ok(v.as_array().ok_or("expected array")?)
}
fn index(v: &Value) -> Result<usize> {
    Ok(usize::try_from(
        v.as_u64().ok_or("expected unsigned integer")?,
    )?)
}
fn matrix(v: &Value) -> Result<Mat3> {
    let a = array(v)?;
    if a.len() != 9 {
        return Err("expected 9 matrix elements".into());
    }
    let mut out = [[0.; 3]; 3];
    for i in 0..9 {
        out[i / 3][i % 3] = number(&a[i])?;
    }
    Ok(out)
}
fn vector(v: &Value) -> Result<Vec3> {
    let a = array(v)?;
    if a.len() != 3 {
        return Err("expected 3 vector elements".into());
    }
    Ok(Vec3([number(&a[0])?, number(&a[1])?, number(&a[2])?]))
}
struct Tokens {
    data: Vec<String>,
    offset: usize,
}
impl Tokens {
    fn open(p: &Path) -> Result<Self> {
        Ok(Self {
            data: fs::read_to_string(p)?
                .split_whitespace()
                .map(str::to_owned)
                .collect(),
            offset: 0,
        })
    }
    fn next<T: std::str::FromStr>(&mut self) -> Result<T> {
        let v = self.data.get(self.offset).ok_or("truncated text dataset")?;
        self.offset += 1;
        v.parse().map_err(|_| format!("invalid token {v}").into())
    }
    fn count(&mut self) -> Result<usize> {
        let n: usize = self.next()?;
        if n > 1_000_000 {
            return Err("dataset count too large".into());
        }
        Ok(n)
    }
    fn done(&self) -> Result<()> {
        if self.offset != self.data.len() {
            return Err("trailing text dataset values".into());
        }
        Ok(())
    }
}
// Text records can span several physical lines (the original loaders use >>).
// Ignore unrelated comment/header lines while retaining numeric continuations.
fn text_records(path: &Path) -> Result<Vec<(String, Vec<String>)>> {
    let mut records: Vec<(String, Vec<String>)> = Vec::new();
    for line in fs::read_to_string(path)?.lines() {
        let words: Vec<String> = line.split_whitespace().map(str::to_owned).collect();
        let Some(first) = words.first() else { continue };
        if matches!(first.as_str(), "frame" | "view" | "p") {
            records.push((first.clone(), words[1..].to_vec()));
        } else if words.iter().all(|v| v.parse::<f64>().is_ok())
            && let Some((_, data)) = records.last_mut()
        {
            data.extend(words);
        }
    }
    Ok(records)
}
fn empty_view() -> View {
    View {
        camera: Camera {
            k: IDENTITY,
            r: IDENTITY,
            t: Vec3::default(),
        },
        joints: BTreeMap::new(),
        bone_weights: BTreeMap::new(),
    }
}
pub const COCO17: [i32; 17] = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11];
fn type_map(name: &str) -> Result<Vec<i32>> {
    Ok(match name {
        "Body25" => (0..25).collect(),
        "Coco17" => COCO17.to_vec(),
        "Coco18" => vec![
            0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        ],
        "Coco19" => vec![
            1, 0, 8, 5, 6, 7, 12, 13, 14, 2, 3, 4, 9, 10, 11, 15, 16, 17, 18,
        ],
        "Shelf14" => vec![11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 6, 7, 1, 0],
        "OptiTrack21" => vec![
            8, -1, -1, -1, 0, -1, 5, 6, 7, -1, 2, 3, 4, 12, 13, 14, 9, 10, 11, 19, 22,
        ],
        "Skel19" => vec![
            8, 1, 9, 12, 0, 2, 5, 10, 13, 17, 18, 3, 6, 11, 14, 4, 7, 19, 22,
        ],
        _ => return Err(format!("unsupported skeleton {name}").into()),
    })
}
pub fn convert_pose(source: &MultiPose3D, name: &str) -> Result<MultiPose3D> {
    let map = type_map(name)?;
    let mut output = MultiPose3D::new();
    for (&id, pose) in source {
        let mut dst = Pose3D::new();
        for (&kind, &v) in pose {
            if !v.finite() {
                return Err("non-finite ground truth coordinate".into());
            }
            if let Some(&new) = map.get(kind)
                && new >= 0
            {
                dst.insert(new as usize, v);
            }
        }
        let neck = matches!(name, "Coco17" | "Shelf14" | "OptiTrack21" | "Skel19");
        let hip = matches!(name, "Coco17" | "Coco18" | "Shelf14");
        for (enable, a, b, c) in [(neck, 2, 5, 1), (hip, 9, 12, 8)] {
            if enable && let (Some(&a), Some(&b)) = (dst.get(&a), dst.get(&b)) {
                dst.insert(c, (a + b) * 0.5);
            }
        }
        output.insert(id, dst);
    }
    Ok(output)
}
pub fn convert_coco17(view: &mut View) -> Result<()> {
    let source = std::mem::take(&mut view.joints);
    let mut uid = source
        .values()
        .flatten()
        .map(|j| j.uid)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or("UID overflow")?;
    for (old, js) in source {
        let &new = COCO17.get(old).ok_or("invalid Coco17 kind")?;
        view.joints.insert(new as usize, js);
    }
    for (a, b, c) in [(2, 5, 1), (9, 12, 8)] {
        let aa = view.joints.get(&a).cloned().unwrap_or_default();
        let bb = view.joints.get(&b).cloned().unwrap_or_default();
        if aa.len() != bb.len() {
            return Err("incomplete COCO person".into());
        }
        for (a, b) in aa.iter().zip(bb) {
            view.add(
                c,
                Joint2D {
                    point: [
                        (a.point[0] + b.point[0]) * 0.5,
                        (a.point[1] + b.point[1]) * 0.5,
                    ],
                    conf: (a.conf + b.conf) * 0.5,
                    uid,
                },
            );
            uid = uid.checked_add(1).ok_or("UID overflow")?;
        }
    }
    Ok(())
}
/// Brown-Conrady inverse, five fixed-point iterations, matching upstream's default OpenCV call.
pub fn undistort_normalized(uv: [f64; 2], k: Mat3, d: [f64; 5]) -> Result<[f64; 2]> {
    let x0 = (uv[0] - k[0][2] as f64) / k[0][0] as f64;
    let y0 = (uv[1] - k[1][2] as f64) / k[1][1] as f64;
    let (mut x, mut y) = (x0, y0);
    let [k1, k2, p1, p2, k3] = d;
    for _ in 0..5 {
        let r = x * x + y * y;
        let radial = 1. + ((k3 * r + k2) * r + k1) * r;
        if radial <= 0. || !radial.is_finite() {
            return Err("non-invertible distortion".into());
        }
        let dx = 2. * p1 * x * y + p2 * (r + 2. * x * x);
        let dy = p1 * (r + 2. * y * y) + 2. * p2 * x * y;
        x = (x0 - dx) / radial;
        y = (y0 - dy) / radial;
    }
    if !x.is_finite() || !y.is_finite() {
        return Err("invalid distortion result".into());
    }
    Ok([x, y])
}
/// Estimate a rectified pinhole viewport from the nine-point-per-edge border grid.
pub fn optimal_camera(k: Mat3, d: [f64; 5], size: [f64; 2], alpha: f64) -> Result<Mat3> {
    if size.iter().any(|v| !v.is_finite() || *v <= 1.)
        || !alpha.is_finite()
        || !(0.0..=1.0).contains(&alpha)
    {
        return Err("invalid camera image size/alpha".into());
    }
    let mut inner = [
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::INFINITY,
    ];
    let mut outer = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for y in 0..9 {
        for x in 0..9 {
            if x != 0 && x != 8 && y != 0 && y != 8 {
                continue;
            }
            let [u, v] = undistort_normalized(
                [
                    x as f64 * (size[0] - 1.) / 8.,
                    y as f64 * (size[1] - 1.) / 8.,
                ],
                k,
                d,
            )?;
            outer[0] = outer[0].min(u);
            outer[1] = outer[1].min(v);
            outer[2] = outer[2].max(u);
            outer[3] = outer[3].max(v);
            if x == 0 {
                inner[0] = inner[0].max(u)
            }
            if x == 8 {
                inner[2] = inner[2].min(u)
            }
            if y == 0 {
                inner[1] = inner[1].max(v)
            }
            if y == 8 {
                inner[3] = inner[3].min(v)
            }
        }
    }
    let mut output = k;
    for axis in 0..2 {
        let f0 = (size[axis] - 1.) / (inner[axis + 2] - inner[axis]);
        let f1 = (size[axis] - 1.) / (outer[axis + 2] - outer[axis]);
        output[axis][axis] = (f0 * (1. - alpha) + f1 * alpha) as f32;
        output[axis][2] = (-f0 * inner[axis] * (1. - alpha) - f1 * outer[axis] * alpha) as f32;
    }
    inverse(output)?;
    Ok(output)
}
pub fn load_cameras(path: &Path, kind: &str, views: usize) -> Result<Vec<Camera>> {
    let mut out = Vec::new();
    match kind {
        "4DA" => {
            let data = json(path)?;
            let obj = data.as_object().ok_or("expected camera object")?;
            for cam in obj.values() {
                let mut k = matrix(&cam["K"])?;
                let (mut r, mut t) = (IDENTITY, Vec3::default());
                if let Some(v) = cam.get("R") {
                    r = if array(v)?.len() == 3 {
                        rodrigues(vector(v)?)
                    } else {
                        matrix(v)?
                    };
                }
                if let Some(v) = cam.get("T") {
                    t = vector(v)?;
                }
                if let Some(rt) = cam.get("RT") {
                    let a = array(rt)?;
                    if a.len() != 12 {
                        return Err("RT must have 12 elements".into());
                    }
                    for i in 0..3 {
                        for j in 0..3 {
                            r[i][j] = number(&a[i * 4 + j])?;
                        }
                        t.0[i] = number(&a[i * 4 + 3])?;
                    }
                }
                let size = array(&cam["imgSize"])?;
                if size.len() != 2 {
                    return Err("invalid imgSize".into());
                }
                let mut d = [0.; 5];
                if let Some(coeffs) = cam.get("distCoeff") {
                    let values = array(coeffs)?;
                    if values.len() > 5 {
                        return Err("only Brown 5-coefficient distortion supported".into());
                    }
                    for (i, v) in values.iter().enumerate() {
                        d[i] = number(v)? as f64;
                    }
                }
                k = optimal_camera(
                    k,
                    d,
                    [number(&size[0])? as f64, number(&size[1])? as f64],
                    cam.get("rectifyAlpha")
                        .map(number)
                        .transpose()?
                        .unwrap_or(0.) as f64,
                )?;
                out.push(Camera::new(k, r, t)?);
            }
        }
        "Campus" | "OSD" => {
            let mut ts = Tokens::open(path)?;
            for _ in 0..views {
                let mut k = [[0.; 3]; 3];
                let mut r = IDENTITY;
                let mut t = Vec3::default();
                if kind == "Campus" {
                    for (i, row) in k.iter_mut().enumerate() {
                        for v in row {
                            *v = ts.next()?;
                        }
                        t.0[i] = ts.next()?;
                    }
                    t = mat_vec(inverse(k)?, t);
                } else {
                    for v in k.iter_mut().flatten() {
                        *v = ts.next()?
                    }
                    for v in r.iter_mut().flatten() {
                        *v = ts.next()?
                    }
                    for v in &mut t.0 {
                        *v = ts.next()?
                    }
                }
                out.push(Camera::new(k, r, t)?);
            }
            ts.done()?;
        }
        _ => return Err(format!("unsupported camera loader {kind}").into()),
    }
    if out.len() != views {
        return Err("camera count differs from detection views".into());
    }
    Ok(out)
}
pub fn load_detections(c: &Config, path: &Path) -> Result<Vec<(usize, MultiView)>> {
    let mut frames: BTreeMap<usize, MultiView> = BTreeMap::new();
    let mut uid = 0_u32;
    let mut add = |view: &mut View, kind, point, conf| -> Result<()> {
        view.add(kind, Joint2D { point, conf, uid });
        uid = uid.checked_add(1).ok_or("UID overflow")?;
        Ok(())
    };
    match c.detection_loader.as_str() {
        "MMPose" => {
            let mut expected = None;
            for view_id in 0..c.detection_view_num {
                let data = json(&path.join(format!("{view_id}.json")))?;
                let mut ids = Vec::new();
                for info in array(&data["instance_info"])? {
                    let id = index(&info["frame_id"])?;
                    if id < c.detection_frame_range[0] || id > c.detection_frame_range[1] {
                        continue;
                    }
                    if ids.last().is_some_and(|&old| id <= old) {
                        return Err("unordered/duplicate MMPose frame".into());
                    }
                    ids.push(id);
                    let mut view = empty_view();
                    for person in array(&info["instances"])?
                        .iter()
                        .take(c.detection_max_person_num)
                    {
                        let pts = array(&person["keypoints"])?;
                        let scores = array(&person["keypoint_scores"])?;
                        if pts.len() < c.detection_joint_type_num
                            || scores.len() < c.detection_joint_type_num
                        {
                            return Err("truncated MMPose joints".into());
                        }
                        for kind in 0..c.detection_joint_type_num {
                            let pt = array(&pts[kind])?;
                            if pt.len() < 2 {
                                return Err("truncated keypoint".into());
                            }
                            add(
                                &mut view,
                                kind,
                                [number(&pt[0])?, number(&pt[1])?],
                                number(&scores[kind])?,
                            )?;
                        }
                    }
                    frames.entry(id).or_default().push(view);
                }
                if expected.as_ref().is_some_and(|old| old != &ids) {
                    return Err("MMPose frame IDs differ between cameras".into());
                }
                expected = Some(ids);
            }
        }
        "OpenPose" => {
            for frame in c.detection_frame_range[0]..=c.detection_frame_range[1] {
                let mut views = Vec::new();
                for view_id in 0..c.detection_view_num {
                    let data = json(&path.join(format!("{view_id}_{frame:012}_keypoints.json")))?;
                    let mut view = empty_view();
                    for person in array(&data["people"])?
                        .iter()
                        .take(c.detection_max_person_num)
                    {
                        let pts = array(&person["pose_keypoints_2d"])?;
                        if pts.len() < c.detection_joint_type_num * 3 {
                            return Err("truncated OpenPose joints".into());
                        }
                        for kind in 0..c.detection_joint_type_num {
                            add(
                                &mut view,
                                kind,
                                [number(&pts[3 * kind])?, number(&pts[3 * kind + 1])?],
                                number(&pts[3 * kind + 2])?,
                            )?;
                        }
                    }
                    views.push(view);
                }
                frames.insert(frame, views);
            }
        }
        "OSD" => {
            let mut current = None;
            let mut last_view = None;
            for (keyword, words) in text_records(path)? {
                let mut ts = Tokens {
                    data: words,
                    offset: 0,
                };
                match keyword.as_str() {
                    "frame" => {
                        let id = ts.count()?;
                        if frames.contains_key(&id) || current.is_some_and(|old| id <= old) {
                            return Err("unordered OSD frame".into());
                        }
                        frames.insert(id, Vec::new());
                        current = Some(id);
                        last_view = None;
                    }
                    "view" => {
                        let id = ts.count()?;
                        let views = frames
                            .get_mut(&current.ok_or("OSD view before frame")?)
                            .unwrap();
                        if id != views.len() {
                            return Err("nonsequential OSD view".into());
                        }
                        views.push(empty_view());
                        last_view = Some(id);
                    }
                    "p" => {
                        let score: f32 = ts.next()?;
                        if score < 0.01 {
                            continue;
                        }
                        let view = &mut frames
                            .get_mut(&current.ok_or("OSD person before frame")?)
                            .unwrap()[last_view.ok_or("OSD person before view")?];
                        for kind in 0..c.detection_joint_type_num {
                            add(view, kind, [ts.next()?, ts.next()?], ts.next()?)?;
                        }
                        ts.done()?;
                    }
                    _ => {}
                }
            }
        }
        "4DA" => {
            let bones = [
                (1, 8),
                (9, 10),
                (10, 11),
                (8, 9),
                (8, 12),
                (12, 13),
                (13, 14),
                (1, 2),
                (2, 3),
                (3, 4),
                (2, 17),
                (1, 5),
                (5, 6),
                (6, 7),
                (5, 18),
                (1, 0),
                (0, 15),
                (0, 16),
                (15, 17),
                (16, 18),
                (14, 19),
                (19, 20),
                (14, 21),
                (11, 22),
                (22, 23),
                (11, 24),
            ];
            let mut count = None;
            for view_id in 0..c.detection_view_num {
                let mut ts = Tokens::open(&path.join(format!("{view_id}.txt")))?;
                let mut n = ts.count()?;
                while n < 10 {
                    n = ts.count()?;
                }
                if count.is_some_and(|v| v != n) {
                    return Err("4DA frame counts differ".into());
                }
                count = Some(n);
                for frame in 0..n {
                    let mut view = empty_view();
                    for kind in 0..c.detection_joint_type_num {
                        let num = ts.count()?;
                        if num > 255 {
                            return Err("4DA candidate count exceeds byte capacity".into());
                        }
                        let mut values = vec![[0.; 3]; num];
                        for axis in 0..3 {
                            for v in &mut values {
                                v[axis] = ts.next()?;
                            }
                        }
                        for v in values {
                            add(
                                &mut view,
                                kind,
                                [
                                    v[0] * (c.detection_image_width as f32 - 1.),
                                    v[1] * (c.detection_image_height as f32 - 1.),
                                ],
                                v[2],
                            )?;
                        }
                    }
                    for &(a, b) in &bones {
                        for ja in view.joints.get(&a).into_iter().flatten() {
                            for jb in view.joints.get(&b).into_iter().flatten() {
                                let w: f32 = ts.next()?;
                                view.bone_weights.insert(
                                    format!("{}:{}", ja.uid.min(jb.uid), ja.uid.max(jb.uid)),
                                    w,
                                );
                            }
                        }
                    }
                    frames.entry(frame).or_default().push(view);
                }
                ts.done()?;
            }
        }
        other => return Err(format!("unsupported detection loader {other}").into()),
    }
    if frames.is_empty() {
        return Err("no detection frames in selected range".into());
    }
    for views in frames.values() {
        if views.len() != c.detection_view_num {
            return Err("incomplete multi-view frame".into());
        }
    }
    Ok(frames.into_iter().collect())
}
pub fn load_ground_truth(c: &Config, path: &Path) -> Result<Vec<MultiPose3D>> {
    let mut frames = Vec::new();
    match c.ground_truth_loader.as_str() {
        "None" => return Ok(frames),
        "4DA" | "4DA Shelf" => {
            let shelf = c.ground_truth_loader == "4DA Shelf";
            let mut ts = Tokens::open(path)?;
            let kinds = if shelf {
                ts.count()?
            } else {
                c.ground_truth_joint_type_num
            };
            let n = ts.count()?;
            if kinds > 25 {
                return Err("ground truth has >25 joints".into());
            }
            for _ in 0..n {
                let persons = ts.count()?;
                let mut poses = MultiPose3D::new();
                for person in 0..persons {
                    let id = if shelf { ts.count()? } else { person };
                    let mut points = vec![[0.; 3]; kinds];
                    for axis in 0..3 {
                        for p in &mut points {
                            p[axis] = ts.next()?;
                        }
                    }
                    let mut pose = Pose3D::new();
                    for (kind, p) in points.into_iter().enumerate() {
                        let valid: f32 = ts.next()?;
                        if valid >= 1e-4 {
                            pose.insert(kind, Vec3(p));
                        }
                    }
                    if id < c.ground_truth_max_person_num {
                        poses.insert(id, pose);
                    }
                }
                frames.push(poses);
            }
            ts.done()?;
        }
        "Campus" => {
            for (keyword, words) in text_records(path)? {
                match keyword.as_str() {
                    "frame" => frames.push(MultiPose3D::new()),
                    "p" => {
                        let mut ts = Tokens {
                            data: words,
                            offset: 0,
                        };
                        let id: usize =
                            ts.count()?.checked_sub(1).ok_or("Campus IDs start at 1")?;
                        let mut pose = Pose3D::new();
                        for k in 0..c.ground_truth_joint_type_num {
                            pose.insert(k, Vec3([ts.next()?, ts.next()?, ts.next()?]));
                        }
                        ts.done()?;
                        if id < c.ground_truth_max_person_num {
                            frames
                                .last_mut()
                                .ok_or("Campus person before frame")?
                                .insert(id, pose);
                        }
                    }
                    _ => {}
                }
            }
        }
        "Panoptic" => {
            for id in c.ground_truth_frame_range[0]..=c.ground_truth_frame_range[1] {
                let path = path.join(format!("body3DScene_{id:08}.json"));
                let mut poses = MultiPose3D::new();
                if path.exists() {
                    let data = json(&path)?;
                    for body in array(&data["bodies"])? {
                        let id = index(&body["id"])?;
                        let a = array(&body["joints19"])?;
                        if a.len() != 76 {
                            return Err("expected 19 Panoptic joints".into());
                        }
                        let mut pose = Pose3D::new();
                        for (k, v) in a.chunks_exact(4).enumerate() {
                            pose.insert(k, Vec3([number(&v[0])?, number(&v[1])?, number(&v[2])?]));
                        }
                        poses.insert(id, pose);
                    }
                }
                frames.push(poses);
            }
        }
        other => return Err(format!("unsupported ground truth loader {other}").into()),
    }
    frames
        .into_iter()
        .map(|p| convert_pose(&p, &c.ground_truth_skel_type))
        .collect()
}
pub fn synchronize(source: &[MultiPose3D], points: &[(usize, usize)]) -> Result<Vec<MultiPose3D>> {
    if points.len() < 2
        || points
            .windows(2)
            .any(|v| v[0].0 >= v[1].0 || v[0].1 >= v[1].1)
    {
        return Err("sync points must be strictly increasing".into());
    }
    let end = points.last().unwrap().0;
    if end > 1_000_000 {
        return Err("sync range too large".into());
    }
    let mut out = vec![MultiPose3D::new(); end];
    let mut k = 1;
    for (frame, slot) in out.iter_mut().enumerate().skip(points[0].0) {
        while frame >= points[k].0 {
            k += 1;
        }
        let (a, b) = points[k - 1];
        let (c, d) = points[k];
        let index = (b as f32 + (frame - a) as f32 * (d - b) as f32 / (c - a) as f32) as usize;
        *slot = source
            .get(index)
            .ok_or("sync point outside source")?
            .clone();
    }
    Ok(out)
}
pub fn correct(poses: &mut MultiPose3D, kind: &str) -> Result<()> {
    match kind {
        "None" => {}
        "Shelf" => {
            for p in poses.values_mut() {
                if let Some(nose) = p.get_mut(&0) {
                    nose.0[2] += 0.05;
                }
            }
        }
        "4DA" => {
            for p in poses.values_mut() {
                if let (Some(&nose), Some(&neck), Some(&hip)) = (p.get(&0), p.get(&1), p.get(&8)) {
                    let axis = (neck - hip).normalized()?;
                    p.insert(0, neck + axis * (nose - neck).dot(axis));
                }
            }
        }
        _ => return Err("unknown result correction".into()),
    }
    Ok(())
}
pub fn load(config_path: &Path, c: &Config) -> Result<Dataset> {
    if !(2..=16).contains(&c.detection_view_num)
        || c.detection_joint_type_num == 0
        || c.detection_joint_type_num > 25
        || c.detection_max_person_num == 0
        || c.detection_max_person_num > 255
        || !c.detection_joint_2d_scale.is_finite()
        || c.detection_joint_2d_scale <= 0.
        || c.detection_frame_range[0] > c.detection_frame_range[1]
        || c.ground_truth_frame_range[0] > c.ground_truth_frame_range[1]
    {
        return Err("invalid dataset config".into());
    }
    if c.detection_skel_type != "Body25" && c.detection_skel_type != "Coco17" {
        return Err("2D detections support Body25 or Coco17, matching upstream".into());
    }
    let base = config_path.parent().unwrap_or(Path::new("."));
    let mut frames = load_detections(c, &base.join(&c.detection_load_path))?;
    let cameras = load_cameras(
        &base.join(&c.camera_load_path),
        &c.camera_loader,
        c.detection_view_num,
    )?;
    for (_, views) in &mut frames {
        for (i, v) in views.iter_mut().enumerate() {
            for j in v.joints.values_mut().flatten() {
                j.point[0] /= c.detection_joint_2d_scale;
                j.point[1] /= c.detection_joint_2d_scale;
            }
            if c.detection_skel_type == "Coco17" {
                convert_coco17(v)?;
            }
            v.camera = cameras[i].clone();
            v.validate()?;
        }
    }
    let mut ground_truth = load_ground_truth(c, &base.join(&c.ground_truth_load_path))?;
    if !c.sync_points_path.as_os_str().is_empty() {
        let mut ts = Tokens::open(&base.join(&c.sync_points_path))?;
        let n = ts.count()?;
        let mut points = Vec::new();
        for _ in 0..n {
            points.push((ts.count()?, ts.count()?));
        }
        ts.done()?;
        ground_truth = synchronize(&ground_truth, &points)?;
    }
    ground_truth.resize(frames.len(), MultiPose3D::new());
    Ok(Dataset {
        frames,
        ground_truth,
    })
}

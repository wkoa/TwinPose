// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use crate::model::*;
use serde::Serialize;
pub const EVAL_JOINTS: [usize; 13] = [0, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14];
pub const EVAL_BONES: [(usize, usize); 10] = [
    (1, 8),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (9, 10),
    (10, 11),
    (12, 13),
    (13, 14),
    (0, 1),
];
#[derive(Debug, Serialize)]
pub struct Metrics {
    pub matched_person: Option<usize>,
    pub pcp: Option<f32>,
    pub pck: Option<f32>,
    pub mpjpe_mm: Option<f32>,
    pub valid_gt_joints: usize,
    pub predicted_gt_joints: usize,
    pub valid_gt_bones: usize,
}
/// Same capped-distance person matching as upstream. Missing errors stay null,
/// rather than becoming a misleading zero-millimeter MPJPE.
pub fn evaluate(gt: &Pose3D, poses: &MultiPose3D) -> Metrics {
    let best = poses.iter().min_by(|(_, a), (_, b)| {
        let score = |pose: &Pose3D| {
            EVAL_JOINTS
                .iter()
                .map(|k| match (gt.get(k), pose.get(k)) {
                    (Some(a), Some(b)) => a.distance(*b).min(0.5),
                    _ => 0.5,
                })
                .sum::<f32>()
        };
        score(a).total_cmp(&score(b))
    });
    let valid_gt_joints = EVAL_JOINTS.iter().filter(|k| gt.contains_key(k)).count();
    let valid_gt_bones = EVAL_BONES
        .iter()
        .filter(|(a, b)| gt.contains_key(a) && gt.contains_key(b))
        .count();
    let mut m = Metrics {
        matched_person: best.map(|(id, _)| *id),
        pcp: None,
        pck: None,
        mpjpe_mm: None,
        valid_gt_joints,
        predicted_gt_joints: 0,
        valid_gt_bones,
    };
    if let Some((_, pose)) = best {
        let mut sum = 0.;
        let mut correct = 0;
        for kind in EVAL_JOINTS {
            if let (Some(a), Some(b)) = (gt.get(&kind), pose.get(&kind)) {
                let d = a.distance(*b);
                sum += d;
                m.predicted_gt_joints += 1;
                if d < 0.2 {
                    correct += 1;
                }
            }
        }
        if valid_gt_joints > 0 {
            m.pck = Some(correct as f32 / valid_gt_joints as f32);
        }
        if m.predicted_gt_joints == valid_gt_joints && valid_gt_joints > 0 {
            m.mpjpe_mm = Some(sum * 1000. / valid_gt_joints as f32);
        }
        let mut correct = 0;
        for (a, b) in EVAL_BONES {
            if let (Some(ga), Some(gb), Some(pa), Some(pb)) =
                (gt.get(&a), gt.get(&b), pose.get(&a), pose.get(&b))
                && ga.distance(*gb) * 0.5 > (pa.distance(*ga) + pb.distance(*gb)) * 0.5
            {
                correct += 1;
            }
        }
        if valid_gt_bones > 0 {
            m.pcp = Some(correct as f32 / valid_gt_bones as f32);
        }
    }
    m
}

// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use crate::{
    Result,
    geometry::{Camera, Vec3},
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Joint2D {
    pub point: [f32; 2],
    pub conf: f32,
    pub uid: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct View {
    pub camera: Camera,
    pub joints: BTreeMap<usize, Vec<Joint2D>>,
    #[serde(default)]
    pub bone_weights: BTreeMap<String, f32>,
}
impl View {
    pub fn add(&mut self, kind: usize, joint: Joint2D) {
        self.joints.entry(kind).or_default().push(joint);
    }
    pub fn joint(&self, kind: usize, person: u8) -> Option<&Joint2D> {
        self.joints.get(&kind)?.get(person.checked_sub(1)? as usize)
    }
    pub fn validate(&self) -> Result<()> {
        Camera::new(self.camera.k, self.camera.r, self.camera.t)?;
        let mut uids = std::collections::BTreeSet::new();
        for joints in self.joints.values() {
            if joints.len() > 255 {
                return Err("more than 255 candidates per view".into());
            }
            for j in joints {
                if !j.conf.is_finite()
                    || j.conf < 0.0
                    || !j.point.iter().all(|v| v.is_finite())
                    || !uids.insert(j.uid)
                {
                    return Err("invalid confidence, pixel or duplicate UID".into());
                }
            }
        }
        Ok(())
    }
}
pub type MultiView = Vec<View>;
pub type Pose3D = BTreeMap<usize, Vec3>;
pub type MultiPose3D = BTreeMap<usize, Pose3D>;
/// One byte per camera, with zero denoting no candidate. Independent of host endianness.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct PersonIds(pub [u8; 16]);
impl PersonIds {
    pub fn conflicts(self, b: Self) -> bool {
        self.0
            .iter()
            .zip(b.0)
            .any(|(&a, b)| a != 0 && b != 0 && a != b)
    }
    pub fn contains(self, b: Self) -> bool {
        self.0.iter().zip(b.0).all(|(&a, b)| b == 0 || a == b)
    }
    pub fn union(self, b: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] | b.0[i]))
    }
}
pub const BODY25: [&[usize]; 25] = [
    &[15, 16],
    &[0, 2, 5],
    &[3],
    &[4],
    &[],
    &[6],
    &[7],
    &[],
    &[1, 9, 12],
    &[10],
    &[11],
    &[22, 24],
    &[13],
    &[14],
    &[19, 21],
    &[17],
    &[18],
    &[],
    &[],
    &[20],
    &[],
    &[],
    &[23],
    &[],
    &[],
];
pub const ORDERED_JOINTS: [usize; 15] = [0, 4, 3, 2, 7, 6, 5, 1, 11, 10, 9, 14, 13, 12, 8];
pub const MIN_BONES: [f32; 15] = [
    0., 0.4, 0.05, 0.1, 0.05, 0.05, 0.1, 0.05, 0., 0., 0.2, 0.2, 0., 0.2, 0.2,
];
pub const MAX_BONES: [f32; 15] = [
    0.5, 0.75, 0.3, 0.5, 0.5, 0.3, 0.5, 0.5, 0., 0.25, 0.55, 0.6, 0.25, 0.55, 0.6,
];

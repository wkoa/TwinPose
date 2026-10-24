// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use crate::Result;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign, Mul, Neg, Sub};
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(transparent)]
pub struct Vec3(pub [f32; 3]);
impl Vec3 {
    pub fn dot(self, b: Self) -> f32 {
        self.0[0] * b.0[0] + self.0[1] * b.0[1] + self.0[2] * b.0[2]
    }
    pub fn cross(self, b: Self) -> Self {
        let [x, y, z] = self.0;
        let [a, b, c] = b.0;
        Self([y * c - z * b, z * a - x * c, x * b - y * a])
    }
    pub fn norm(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn distance(self, b: Self) -> f32 {
        (self - b).norm()
    }
    pub fn normalized(self) -> Result<Self> {
        let n = self.norm();
        if !n.is_finite() || n <= f32::MIN_POSITIVE {
            return Err("zero/non-finite direction".into());
        }
        Ok(Self(self.0.map(|v| v / n)))
    }
    pub fn finite(self) -> bool {
        self.0.iter().all(|x| x.is_finite())
    }
}
impl Add for Vec3 {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + b.0[i]))
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] - b.0[i]))
    }
}
impl Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, b: f32) -> Self {
        Self(self.0.map(|x| x * b))
    }
}
impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self {
        self * -1.0
    }
}
impl AddAssign for Vec3 {
    fn add_assign(&mut self, b: Self) {
        *self = *self + b;
    }
}
pub type Mat3 = [[f32; 3]; 3];
pub const IDENTITY: Mat3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
pub fn transpose(a: Mat3) -> Mat3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}
pub fn mat_vec(a: Mat3, b: Vec3) -> Vec3 {
    Vec3(a.map(|r| Vec3(r).dot(b)))
}
pub fn mat_mul(a: Mat3, b: Mat3) -> Mat3 {
    let bt = transpose(b);
    a.map(|r| bt.map(|c| Vec3(r).dot(Vec3(c))))
}
pub fn inverse(m: Mat3) -> Result<Mat3> {
    let [[a, b, c], [d, e, f], [g, h, i]] = m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    // Relative determinant rejects parallel rays without imposing a world scale.
    let scale = m.iter().flatten().fold(0.0_f32, |v, x| v.max(x.abs()));
    if !det.is_finite() || det.abs() <= 1e-8 * scale.powi(3) {
        return Err("singular geometry".into());
    }
    Ok([
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    .map(|r| r.map(|x| x * (1.0 / det))))
}
#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}
impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Result<Self> {
        if !origin.finite() {
            return Err("invalid ray origin".into());
        }
        Ok(Self {
            origin,
            direction: direction.normalized()?,
        })
    }
    pub fn distance_to(self, point: Vec3) -> f32 {
        (self.origin + self.direction * (point - self.origin).dot(self.direction)).distance(point)
    }
    pub fn distance(self, b: Self) -> f32 {
        let c = self.direction.cross(b.direction);
        let delta = self.origin - b.origin;
        if c.norm() < 1e-6 {
            delta.cross(self.direction).norm()
        } else {
            delta.dot(Vec3(c.0.map(|v| v / c.norm()))).abs()
        }
    }
}
pub fn triangulate(rays: &[Ray], weights: Option<&[f32]>) -> Result<(Vec3, f32)> {
    if rays.len() < 2 || weights.is_some_and(|w| w.len() != rays.len()) {
        return Err("triangulation requires >=2 rays and matching weights".into());
    }
    let mut a = [[0.; 3]; 3];
    let mut b = Vec3::default();
    let mut positive = 0;
    for (n, ray) in rays.iter().enumerate() {
        if !ray.origin.finite()
            || !ray.direction.finite()
            || (ray.direction.norm() - 1.).abs() > 1e-4
        {
            return Err("invalid ray".into());
        }
        let w = weights.map_or(1., |v| v[n]);
        if !w.is_finite() || w < 0. {
            return Err("invalid triangulation weight".into());
        }
        if w > 0. {
            positive += 1
        }
        let d = ray.direction.0;
        let normal: Mat3 = std::array::from_fn(|i| {
            std::array::from_fn(|j| d[i] * d[j] - if i == j { 1. } else { 0. })
        });
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += normal[i][j] * w;
            }
        }
        b += mat_vec(normal, ray.origin) * w;
    }
    if positive < 2 {
        return Err("not enough positive weights".into());
    }
    let center = mat_vec(inverse(a)?, b);
    if !center.finite() {
        return Err("non-finite triangulation".into());
    }
    Ok((
        center,
        rays.iter()
            .fold(0.0_f32, |r, ray| r.max(ray.distance_to(center))),
    ))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Camera {
    pub k: Mat3,
    pub r: Mat3,
    pub t: Vec3,
}
impl Camera {
    pub fn new(k: Mat3, r: Mat3, t: Vec3) -> Result<Self> {
        inverse(k)?;
        inverse(r)?;
        if !t.finite() {
            return Err("invalid camera translation".into());
        }
        Ok(Self { k, r, t })
    }
    pub fn position(&self) -> Vec3 {
        -mat_vec(transpose(self.r), self.t)
    }
    pub fn ray(&self, uv: [f32; 2]) -> Result<Ray> {
        let kri = mat_mul(transpose(self.r), inverse(self.k)?);
        Ray::new(self.position(), -mat_vec(kri, Vec3([uv[0], uv[1], 1.])))
    }
    pub fn project(&self, p: Vec3) -> Result<[f32; 2]> {
        let v = mat_vec(self.k, mat_vec(self.r, p) + self.t).0;
        if !v.iter().all(|x| x.is_finite()) || v[2].abs() < 1e-8 {
            return Err("projection at infinity".into());
        }
        Ok([v[0] / v[2], v[1] / v[2]])
    }
}
pub fn rodrigues(v: Vec3) -> Mat3 {
    let angle = v.norm();
    if angle < 1e-10 {
        return IDENTITY;
    }
    let [x, y, z] = (v * (1. / angle)).0;
    let c = angle.cos();
    let s = angle.sin();
    let q = 1. - c;
    [
        [c + x * x * q, x * y * q - z * s, x * z * q + y * s],
        [y * x * q + z * s, c + y * y * q, y * z * q - x * s],
        [z * x * q - y * s, z * y * q + x * s, c + z * z * q],
    ]
}

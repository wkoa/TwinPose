// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use crate::{
    Result,
    geometry::{Ray, Vec3, triangulate},
    model::*,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Params {
    pub joint_types: Vec<usize>,
    pub max_person_num: usize,
    pub use_bone_length_score: bool,
    pub use_edge_num_in_association: bool,
    pub conf_discard_threshold: f32,
    pub conf_exponent: f32,
    pub max_joint_radius: f32,
    pub max_joint_cluster_radius: f32,
    pub association_score_threshold: f32,
    pub pose_discard_threshold: f32,
    pub use_temporal_information: bool,
    pub tracking_discard_threshold: f32,
    pub tracking_distance: f32,
    pub lost_track_penalty: f32,
}
impl Default for Params {
    fn default() -> Self {
        Self {
            joint_types: ORDERED_JOINTS.to_vec(),
            max_person_num: 0,
            use_bone_length_score: true,
            use_edge_num_in_association: false,
            conf_discard_threshold: 0.,
            conf_exponent: 1.,
            max_joint_radius: 0.05,
            max_joint_cluster_radius: 0.1,
            association_score_threshold: 5.,
            pose_discard_threshold: 10.,
            use_temporal_information: false,
            tracking_discard_threshold: 10.,
            tracking_distance: 0.1,
            lost_track_penalty: 0.5,
        }
    }
}
impl Params {
    pub fn validate(&self) -> Result<()> {
        let set: BTreeSet<_> = self.joint_types.iter().copied().collect();
        if set.is_empty()
            || set.len() != self.joint_types.len()
            || set.iter().any(|&v| v >= 25)
            || self.max_person_num > 255
        {
            return Err("invalid joint types/person limit".into());
        }
        for (i, &p) in self.joint_types.iter().enumerate() {
            for &c in BODY25[p] {
                if let Some(j) = self.joint_types.iter().position(|&v| v == c)
                    && j >= i
                {
                    return Err("joint_types must be ordered children before parent".into());
                }
            }
        }
        let finite = [
            self.conf_discard_threshold,
            self.conf_exponent,
            self.max_joint_radius,
            self.max_joint_cluster_radius,
            self.association_score_threshold,
            self.pose_discard_threshold,
            self.tracking_discard_threshold,
            self.tracking_distance,
            self.lost_track_penalty,
        ]
        .iter()
        .all(|v| v.is_finite());
        if !finite
            || !(0.0..1.0).contains(&self.conf_discard_threshold)
            || self.conf_exponent <= 0.
            || self.max_joint_radius <= 0.
            || self.max_joint_cluster_radius <= 0.
            || self.association_score_threshold < 0.
            || self.pose_discard_threshold < 0.
            || self.tracking_discard_threshold < 0.
            || self.tracking_distance <= 0.
            || !(0.0..=1.0).contains(&self.lost_track_penalty)
        {
            return Err("invalid algorithm thresholds".into());
        }
        Ok(())
    }
    pub fn bones(&self) -> Vec<(usize, usize)> {
        self.joint_types
            .iter()
            .flat_map(|&p| {
                BODY25[p]
                    .iter()
                    .filter(|c| self.joint_types.contains(c))
                    .map(move |&c| (p, c))
            })
            .collect()
    }
}
#[derive(Clone, Debug)]
pub struct AnomalyDetector {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub deviation: f32,
    pub min_z: f32,
    pub max_z: f32,
    pub min_score: f32,
}
impl AnomalyDetector {
    pub fn bone(kind: usize, samples: &[f32]) -> Self {
        let mut a = Self {
            min: MIN_BONES.get(kind).copied().unwrap_or(0.),
            max: MAX_BONES.get(kind).copied().unwrap_or(f32::MAX),
            mean: 0.,
            deviation: 0.,
            min_z: 2.,
            max_z: 3.,
            min_score: 0.5,
        };
        if !samples.is_empty() {
            let count = samples.len() as f64;
            let mean = samples.iter().map(|&v| v as f64).sum::<f64>() / count;
            let variance =
                samples.iter().map(|&v| (v * v) as f64).sum::<f64>() / count - mean * mean;
            a.mean = mean as f32;
            a.deviation = variance.max(0.).sqrt() as f32;
        }
        a
    }
    pub fn evaluate(&self, v: f32) -> f32 {
        if v < self.min || v > self.max {
            return 0.;
        }
        let z = if self.deviation == 0. {
            0.
        } else {
            (v - self.mean).abs() / self.deviation
        };
        (z / (self.min_z - self.max_z) + self.max_z / (self.max_z - self.min_z))
            .clamp(self.min_score, 1.)
    }
}
#[derive(Clone, Debug)]
pub struct Joint {
    pub position: Vec3,
    pub radius: f32,
    pub score: f32,
    pub views: (usize, usize),
    pub ids: PersonIds,
}
#[derive(Clone, Debug)]
pub struct Cluster {
    pub position: Vec3,
    pub score: f32,
    pub ids: PersonIds,
    pub members: Vec<Joint>,
}
#[derive(Clone, Debug, Serialize)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub weight: f32,
}
#[derive(Default, Clone, Debug)]
pub struct DebugFrame {
    pub candidates: BTreeMap<usize, Vec<Joint>>,
    pub clusters: BTreeMap<usize, Vec<Cluster>>,
    pub associations: Vec<(PersonIds, f32)>,
    pub graphs: Vec<Vec<Vec<Edge>>>,
    pub rejected_geometry: usize,
}
struct Disjoint {
    parent: Vec<usize>,
    ids: Vec<PersonIds>,
    weights: Vec<f32>,
}
impl Disjoint {
    fn new(n: usize, people: usize) -> Self {
        let ids = (0..n)
            .map(|i| {
                let mut p = PersonIds::default();
                p.0[i / people] = (i % people + 1) as u8;
                p
            })
            .collect();
        Self {
            parent: (0..n).collect(),
            ids,
            weights: vec![0.; n],
        }
    }
    fn find(&mut self, n: usize) -> usize {
        if self.parent[n] != n {
            self.parent[n] = self.find(self.parent[n]);
        }
        self.parent[n]
    }
    fn merge(&mut self, a: usize, b: usize, w: f32) {
        let a = self.find(a);
        let b = self.find(b);
        if a != b && !self.ids[a].conflicts(self.ids[b]) {
            self.parent[b] = a;
            self.weights[a] += self.weights[b] + w;
            self.ids[a] = self.ids[a].union(self.ids[b]);
        }
    }
}
#[derive(Clone)]
struct Branch {
    last: Option<(usize, usize)>,
    last_score: f32,
    score: f32,
    clusters: Vec<Option<usize>>,
    bone_scores: Vec<f32>,
}
pub fn build(
    views: &MultiView,
    p: &Params,
    detectors: &[AnomalyDetector],
    history: Option<&MultiPose3D>,
) -> Result<(MultiPose3D, DebugFrame)> {
    p.validate()?;
    if views.len() < 2 || views.len() > 16 {
        return Err("requires 2..=16 views".into());
    }
    if detectors.len() < 25 {
        return Err("requires 25 bone detector slots".into());
    }
    for v in views {
        v.validate()?;
    }
    let max_people = if p.max_person_num == 0 {
        views
            .iter()
            .flat_map(|v| v.joints.values().map(Vec::len))
            .max()
            .unwrap_or(0)
    } else {
        p.max_person_num
    };
    if views
        .iter()
        .flat_map(|v| v.joints.values())
        .any(|j| j.len() > max_people)
    {
        return Err("person limit below detection candidate count".into());
    }
    for d in detectors {
        if ![
            d.min,
            d.max,
            d.mean,
            d.deviation,
            d.min_z,
            d.max_z,
            d.min_score,
        ]
        .iter()
        .all(|v| v.is_finite())
            || d.min > d.max
            || d.deviation < 0.
            || d.max_z <= d.min_z
            || !(0.0..=1.0).contains(&d.min_score)
        {
            return Err("invalid bone detector".into());
        }
    }
    let mut out = DebugFrame::default();
    let bones = p.bones();
    let mut rays: Vec<BTreeMap<u32, Ray>> = Vec::new();
    for view in views {
        let mut rs = BTreeMap::new();
        for js in view.joints.values() {
            for j in js {
                rs.insert(j.uid, view.camera.ray(j.point)?);
            }
        }
        rays.push(rs);
    }
    for &kind in &p.joint_types {
        let mut list = Vec::new();
        for a in 0..views.len() {
            for b in 0..a {
                for (ia, ja) in views[a].joints.get(&kind).into_iter().flatten().enumerate() {
                    for (ib, jb) in views[b].joints.get(&kind).into_iter().flatten().enumerate() {
                        let ra = rays[a][&ja.uid];
                        let rb = rays[b][&jb.uid];
                        let radius = ra.distance(rb) * 0.5;
                        let conf = (ja.conf * jb.conf).sqrt();
                        if radius >= p.max_joint_radius || conf <= p.conf_discard_threshold {
                            continue;
                        }
                        let Ok((position, _)) = triangulate(&[ra, rb], None) else {
                            out.rejected_geometry += 1;
                            continue;
                        };
                        let score = ((1. - radius / p.max_joint_radius)
                            * ((conf - p.conf_discard_threshold)
                                / (1. - p.conf_discard_threshold))
                                .powf(p.conf_exponent))
                        .sqrt();
                        if !score.is_finite() {
                            return Err("confidence score overflow".into());
                        }
                        let mut ids = PersonIds::default();
                        ids.0[a] = (ia + 1) as u8;
                        ids.0[b] = (ib + 1) as u8;
                        list.push(Joint {
                            position,
                            radius,
                            score,
                            views: (a, b),
                            ids,
                        });
                    }
                }
            }
        }
        list.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.ids.cmp(&b.ids)));
        let inv = 1. / (2. * p.max_joint_cluster_radius);
        if !inv.is_finite() || inv == 0. {
            return Err("unrepresentable cluster scale".into());
        }
        let mut grids: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
        for (i, j) in list.iter().enumerate() {
            let c = j.position.0.map(|v| (v * inv).round() as i64);
            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        grids
                            .entry([
                                c[0].saturating_sub(x),
                                c[1].saturating_sub(y),
                                c[2].saturating_sub(z),
                            ])
                            .or_default()
                            .push(i);
                    }
                }
            }
        }
        let mut clusters: BTreeMap<PersonIds, Cluster> = BTreeMap::new();
        for j in &list {
            let mut cluster = Cluster {
                position: Vec3::default(),
                score: 0.,
                ids: j.ids,
                members: Vec::new(),
            };
            let grid = j.position.0.map(|v| (v * inv).floor() as i64);
            for &idx in grids.get(&grid).into_iter().flatten() {
                let near = &list[idx];
                if cluster.ids.conflicts(near.ids) {
                    continue;
                }
                let d = j.position.distance(near.position);
                if d >= p.max_joint_cluster_radius - near.radius {
                    continue;
                }
                cluster.score += near.score * (1. - d / p.max_joint_cluster_radius);
                cluster.members.push(near.clone());
                cluster.ids = cluster.ids.union(near.ids);
            }
            if clusters
                .get(&cluster.ids)
                .is_some_and(|c| cluster.score < c.score)
            {
                continue;
            }
            let mut rs = Vec::new();
            let mut ws = Vec::new();
            for (v, &id) in cluster.ids.0[..views.len()].iter().enumerate() {
                if id != 0 {
                    let joint = views[v].joint(kind, id).ok_or("missing cluster joint")?;
                    rs.push(rays[v][&joint.uid]);
                    ws.push(joint.conf);
                }
            }
            match triangulate(&rs, Some(&ws)) {
                Ok((point, _)) => {
                    cluster.position = point;
                    clusters.insert(cluster.ids, cluster);
                }
                Err(_) => out.rejected_geometry += 1,
            }
        }
        out.candidates.insert(kind, list);
        out.clusters.insert(kind, clusters.into_values().collect());
    }
    if max_people == 0 {
        return Ok((MultiPose3D::new(), out));
    }
    let nodes = views.len() * max_people;
    let mut weights: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    for &(a, b) in &bones {
        for ca in &out.clusters[&a] {
            for cb in &out.clusters[&b] {
                let score = detectors[b].evaluate(ca.position.distance(cb.position));
                if score <= 0. {
                    continue;
                }
                for ja in &ca.members {
                    for jb in &cb.members {
                        if ja.ids != jb.ids {
                            continue;
                        }
                        let (v, w) = ja.views;
                        let mut n = v * max_people + ja.ids.0[v] as usize - 1;
                        let mut m = w * max_people + ja.ids.0[w] as usize - 1;
                        if n > m {
                            std::mem::swap(&mut n, &mut m)
                        }
                        *weights.entry((n, m)).or_default() += if p.use_edge_num_in_association {
                            1.
                        } else {
                            (ja.score * jb.score).sqrt()
                                * if p.use_bone_length_score { score } else { 1. }
                        };
                    }
                }
            }
        }
    }
    // At threshold zero the original complete graph also merges zero-weight edges.
    if p.association_score_threshold == 0. {
        for a in 0..nodes {
            for b in a + 1..nodes {
                weights.entry((a, b)).or_default();
            }
        }
    }
    let mut edges: Vec<_> = weights.into_iter().collect();
    edges.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut ds = Disjoint::new(nodes, max_people);
    for ((a, b), w) in edges {
        if w < p.association_score_threshold {
            break;
        }
        ds.merge(a, b, w);
    }
    let mut seen = BTreeSet::new();
    for n in 0..nodes {
        let root = ds.find(n);
        if seen.insert(ds.ids[root]) {
            out.associations.push((ds.ids[root], ds.weights[root]));
        }
    }
    out.associations
        .sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out.associations.truncate(max_people);
    for (ids, _) in &out.associations {
        let mut graphs = Vec::new();
        for &(a, b) in &bones {
            let mut graph = Vec::new();
            for (i, ca) in out.clusters[&a].iter().enumerate() {
                for (j, cb) in out.clusters[&b].iter().enumerate() {
                    let score = detectors[b].evaluate(ca.position.distance(cb.position));
                    if score <= 0. {
                        continue;
                    }
                    let mut sum = 0.;
                    for ja in &ca.members {
                        if !ids.contains(ja.ids) {
                            continue;
                        }
                        for jb in &cb.members {
                            if ja.ids == jb.ids {
                                sum += (ja.score * jb.score).sqrt();
                            }
                        }
                    }
                    if sum > 0. {
                        graph.push(Edge {
                            a: i,
                            b: j,
                            weight: sum * if p.use_bone_length_score { score } else { 1. },
                        })
                    }
                }
            }
            graphs.push(graph);
        }
        out.graphs.push(graphs);
    }
    if let Some(history) = history {
        for graphs in &mut out.graphs {
            let mut best = p.tracking_discard_threshold;
            let mut tracked = None;
            for pose in history.values() {
                let mut score = 0.;
                for (k, &(a, b)) in bones.iter().enumerate() {
                    if let (Some(ha), Some(hb)) = (pose.get(&a), pose.get(&b)) {
                        for edge in &graphs[k] {
                            let d = ha.distance(out.clusters[&a][edge.a].position)
                                + hb.distance(out.clusters[&b][edge.b].position);
                            if d <= 2. * p.tracking_distance {
                                score += edge.weight;
                            }
                        }
                    }
                }
                if score > best {
                    best = score;
                    tracked = Some(pose);
                }
            }
            if let Some(pose) = tracked {
                for (k, &(a, b)) in bones.iter().enumerate() {
                    if let (Some(ha), Some(hb)) = (pose.get(&a), pose.get(&b)) {
                        for edge in &mut graphs[k] {
                            let d = ha.distance(out.clusters[&a][edge.a].position)
                                + hb.distance(out.clusters[&b][edge.b].position);
                            if d > 2. * p.tracking_distance {
                                edge.weight *= p.lost_track_penalty;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut result = MultiPose3D::new();
    for (person, graphs) in out.graphs.iter().enumerate() {
        let mut branches: BTreeMap<usize, Vec<Branch>> = BTreeMap::new();
        for &kind in &p.joint_types {
            branches.insert(
                kind,
                (0..out.clusters[&kind].len())
                    .map(|i| {
                        let mut slots = vec![None; 25];
                        slots[kind] = Some(i);
                        Branch {
                            last: None,
                            last_score: 0.,
                            score: 0.,
                            clusters: slots,
                            bone_scores: vec![0.; bones.len()],
                        }
                    })
                    .collect(),
            );
        }
        for (k, &(a, b)) in bones.iter().enumerate() {
            for e in &graphs[k] {
                let score = branches[&a][e.a].last_score + branches[&b][e.b].score + e.weight;
                if score > branches[&a][e.a].score {
                    let branch = &mut branches.get_mut(&a).unwrap()[e.a];
                    branch.score = score;
                    branch.last = Some((b, e.b));
                }
            }
            for i in 0..branches[&a].len() {
                if branches[&a][i].last.is_none() {
                    let mut fallback: Option<usize> = None;
                    for (j, cb) in out.clusters[&b].iter().enumerate() {
                        if detectors[b].evaluate(out.clusters[&a][i].position.distance(cb.position))
                            <= 0.
                        {
                            continue;
                        }
                        if fallback
                            .is_none_or(|old| branches[&b][j].score > branches[&b][old].score)
                        {
                            fallback = Some(j);
                        }
                    }
                    branches.get_mut(&a).unwrap()[i].last = fallback.map(|j| (b, j));
                }
                let Some((kind, index)) = branches[&a][i].last else {
                    continue;
                };
                let child = branches[&kind][index].clone();
                let parent = &mut branches.get_mut(&a).unwrap()[i];
                parent.bone_scores[k] = parent.score - parent.last_score - child.score;
                parent.last_score = parent.score;
                for &joint in &p.joint_types {
                    if parent.clusters[joint].is_none() {
                        parent.clusters[joint] = child.clusters[joint];
                    }
                }
                for (x, y) in parent.bone_scores.iter_mut().zip(&child.bone_scores) {
                    if *x == 0. {
                        *x = *y;
                    }
                }
            }
        }
        let root = *p.joint_types.last().unwrap();
        let mut best = None;
        for branch in &branches[&root] {
            if best.is_none_or(|b: &Branch| branch.score > b.score) {
                best = Some(branch);
            }
        }
        if let Some(branch) = best
            && branch.score >= p.pose_discard_threshold
        {
            let mut pose = Pose3D::new();
            for &kind in &p.joint_types {
                if let Some(idx) = branch.clusters[kind] {
                    pose.insert(kind, out.clusters[&kind][idx].position);
                }
            }
            result.insert(person, pose);
        }
    }
    Ok((result, out))
}
/// A sequence owns its temporal/bone history. An error never advances this state.
pub struct Tracker {
    pub params: Params,
    history: VecDeque<Vec<Vec<f32>>>,
    previous: Option<MultiPose3D>,
}
impl Tracker {
    pub fn new(params: Params) -> Result<Self> {
        params.validate()?;
        Ok(Self {
            params,
            history: VecDeque::new(),
            previous: None,
        })
    }
    pub fn reset(&mut self) {
        self.history.clear();
        self.previous = None;
    }
    pub fn process(&mut self, views: &MultiView) -> Result<(MultiPose3D, DebugFrame)> {
        self.process_corrected(views, "None")
    }
    pub fn process_corrected(
        &mut self,
        views: &MultiView,
        correction: &str,
    ) -> Result<(MultiPose3D, DebugFrame)> {
        let detectors: Vec<_> = (0..25)
            .map(|k| {
                let samples: Vec<f32> = self
                    .history
                    .iter()
                    .flat_map(|f| f[k].iter().copied())
                    .collect();
                AnomalyDetector::bone(k, &samples)
            })
            .collect();
        let (poses, debug) = build(
            views,
            &self.params,
            &detectors,
            if self.params.use_temporal_information {
                self.previous.as_ref()
            } else {
                None
            },
        )?;
        let mut samples = vec![Vec::new(); 25];
        for pose in poses.values() {
            for (a, b) in self.params.bones() {
                if let (Some(a), Some(v)) = (pose.get(&a), pose.get(&b)) {
                    samples[b].push(a.distance(*v));
                }
            }
        }
        let mut corrected = poses.clone();
        crate::dataset::correct(&mut corrected, correction)?;
        self.history.push_back(samples);
        if self.history.len() > 30 {
            self.history.pop_front();
        }
        self.previous = Some(corrected.clone());
        Ok((corrected, debug))
    }
}

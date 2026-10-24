// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
use std::{collections::BTreeMap, path::PathBuf};
use twinpose::{dataset::*, evaluation::*, geometry::*, model::*, pose::*};
fn near(a: Vec3, b: Vec3) {
    assert!(a.distance(b) < 1e-4, "{a:?} != {b:?}");
}
fn scene() -> (MultiView, Vec<Pose3D>) {
    let points = [
        [0., 0.8, 4.],
        [0., 0.6, 4.],
        [0.2, 0.6, 4.],
        [0.4, 0.5, 4.],
        [0.6, 0.4, 4.],
        [-0.2, 0.6, 4.],
        [-0.4, 0.5, 4.],
        [-0.6, 0.4, 4.],
        [0., 0., 4.],
        [0.1, 0., 4.],
        [0.1, -0.4, 4.],
        [0.1, -0.8, 4.],
        [-0.1, 0., 4.],
        [-0.1, -0.4, 4.],
        [-0.1, -0.8, 4.],
    ];
    let poses: Vec<Pose3D> = [-1., 1.]
        .iter()
        .map(|&offset| {
            points
                .iter()
                .enumerate()
                .map(|(k, &p)| (k, Vec3(p) + Vec3([offset, 0., 0.])))
                .collect()
        })
        .collect();
    let views = [-2., 0., 2.]
        .iter()
        .map(|&x| {
            let cam = Camera::new(
                [[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]],
                IDENTITY,
                Vec3([-x, 0., 0.]),
            )
            .unwrap();
            let mut view = View {
                camera: cam,
                joints: BTreeMap::new(),
                bone_weights: BTreeMap::new(),
            };
            for (i, p) in poses.iter().enumerate() {
                for (&kind, &point) in p {
                    view.add(
                        kind,
                        Joint2D {
                            point: view.camera.project(point).unwrap(),
                            conf: 1.,
                            uid: (i * 25 + kind) as u32,
                        },
                    );
                }
            }
            view
        })
        .collect();
    (views, poses)
}
#[test]
fn weighted_triangulation_and_camera_convention() {
    let (views, poses) = scene();
    let expected = poses[0][&4];
    let rays: Vec<_> = views
        .iter()
        .map(|v| v.camera.ray(v.joint(4, 1).unwrap().point).unwrap())
        .collect();
    let (center, radius) = triangulate(&rays, Some(&[1., 0.5, 0.8])).unwrap();
    near(center, expected);
    assert!(radius < 1e-4);
    near(views[0].camera.position(), Vec3([-2., 0., 0.]));
}
#[test]
fn reject_parallel_and_invalid_weight() {
    let a = Ray::new(Vec3::default(), Vec3([0., 0., 1.])).unwrap();
    let b = Ray::new(Vec3([1., 0., 0.]), Vec3([0., 0., 1.])).unwrap();
    assert!(triangulate(&[a, b], None).is_err());
    assert!(triangulate(&[a, b], Some(&[1., -1.])).is_err());
    assert!(triangulate(&[a], None).is_err());
    assert!(triangulate(&[a, b], Some(&[1.])).is_err());
}
#[test]
fn ray_distance_parallel_and_skew() {
    let a = Ray::new(Vec3::default(), Vec3([1., 0., 0.])).unwrap();
    let b = Ray::new(Vec3([0., 2., 0.]), Vec3([1., 0., 0.])).unwrap();
    assert!((a.distance(b) - 2.).abs() < 1e-6);
    let c = Ray::new(Vec3([0., 0., 3.]), Vec3([0., 1., 0.])).unwrap();
    assert!((a.distance(c) - 3.).abs() < 1e-6);
}
#[test]
fn rebuild_two_people_without_identity_mix() {
    let (views, expected) = scene();
    let mut tracker = Tracker::new(Params::default()).unwrap();
    let (poses, debug) = tracker.process(&views).unwrap();
    assert_eq!(poses.len(), 2);
    assert_eq!(debug.rejected_geometry, 0);
    for pose in poses.values() {
        assert_eq!(pose.len(), 15);
        assert!(
            expected
                .iter()
                .any(|truth| pose.iter().all(|(k, v)| v.distance(truth[k]) < 1e-4))
        );
    }
}
#[test]
fn empty_observations_are_empty_not_previous_pose() {
    let (mut views, _) = scene();
    let p = Params {
        use_temporal_information: true,
        ..Default::default()
    };
    let mut tracker = Tracker::new(p).unwrap();
    assert!(!tracker.process(&views).unwrap().0.is_empty());
    for view in &mut views {
        view.joints.clear();
    }
    assert!(tracker.process(&views).unwrap().0.is_empty());
}
#[test]
fn rejected_frame_does_not_advance_temporal_state() {
    let (views, _) = scene();
    let p = Params {
        use_temporal_information: true,
        ..Default::default()
    };
    let mut a = Tracker::new(p.clone()).unwrap();
    let mut b = Tracker::new(p).unwrap();
    a.process(&views).unwrap();
    b.process(&views).unwrap();
    let mut bad = views.clone();
    bad[0].joints.get_mut(&0).unwrap()[0].conf = f32::NAN;
    assert!(a.process(&bad).is_err());
    assert_eq!(a.process(&views).unwrap().0, b.process(&views).unwrap().0);
    a.reset();
    assert_eq!(
        a.process(&views).unwrap().0,
        Tracker::new(Params::default())
            .unwrap()
            .process(&views)
            .unwrap()
            .0
    );
}
#[test]
fn maximum_view_and_person_capacity_are_checked() {
    let (views, _) = scene();
    let mut tracker = Tracker::new(Params::default()).unwrap();
    let many = vec![views[0].clone(); 17];
    assert!(tracker.process(&many).is_err());
    let p = Params {
        max_person_num: 1,
        ..Default::default()
    };
    assert!(Tracker::new(p).unwrap().process(&views).is_err());
}
#[test]
fn sparse_joint_subset_is_supported() {
    let (views, _) = scene();
    let p = Params {
        joint_types: vec![4, 3, 2],
        association_score_threshold: 0.,
        pose_discard_threshold: 0.,
        ..Default::default()
    };
    let poses = Tracker::new(p).unwrap().process(&views).unwrap().0;
    assert!(!poses.is_empty());
    assert!(
        poses
            .values()
            .all(|p| p.keys().all(|k| [2, 3, 4].contains(k)))
    );
}
#[test]
fn confidence_scores_above_one_are_preserved() {
    let (mut views, _) = scene();
    views[0].joints.get_mut(&0).unwrap()[0].conf = 1.01;
    views[0].validate().unwrap();
    assert_eq!(views[0].joints[&0][0].conf, 1.01);
}
#[test]
fn validate_params_order_and_thresholds() {
    let mut p = Params::default();
    p.joint_types.reverse();
    assert!(p.validate().is_err());
    p = Params::default();
    p.conf_discard_threshold = 1.;
    assert!(p.validate().is_err());
    p = Params::default();
    p.max_joint_radius = 0.;
    assert!(p.validate().is_err());
    p = Params::default();
    p.tracking_distance = f32::NAN;
    assert!(p.validate().is_err());
}
#[test]
fn packed_ids_do_not_depend_on_endianness() {
    let mut a = PersonIds::default();
    a.0[15] = 255;
    let mut b = PersonIds::default();
    b.0[0] = 2;
    assert!(!a.conflicts(b));
    assert!(a.union(b).contains(a));
    b.0[15] = 1;
    assert!(a.conflicts(b));
    assert!(!a.contains(b));
}
#[test]
fn anomaly_detector_hard_cutoffs_and_distribution() {
    let a = AnomalyDetector::bone(3, &[0.2, 0.25, 0.3]);
    assert_eq!(a.evaluate(0.01), 0.);
    assert_eq!(a.evaluate(0.6), 0.);
    assert_eq!(a.evaluate(0.25), 1.);
    assert!(a.evaluate(0.4) >= 0.5);
}
#[test]
fn convert_coco_and_missing_gt_joints() {
    let source = MultiPose3D::from([(
        7,
        Pose3D::from([
            (5, Vec3([0., 0., 1.])),
            (6, Vec3([2., 0., 1.])),
            (11, Vec3([0., 2., 1.])),
            (12, Vec3([2., 2., 1.])),
        ]),
    )]);
    let result = convert_pose(&source, "Coco17").unwrap();
    near(result[&7][&1], Vec3([1., 0., 1.]));
    near(result[&7][&8], Vec3([1., 2., 1.]));
    assert!(!result[&7].contains_key(&0));
    assert!(convert_pose(&source, "unknown").is_err());
}
#[test]
fn zero_distortion_and_rodrigues() {
    let k = [[1000., 0., 500.], [0., 1000., 400.], [0., 0., 1.]];
    let out = optimal_camera(k, [0.; 5], [1000., 800.], 0.).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((k[i][j] - out[i][j]).abs() < 1e-3);
        }
    }
    let r = rodrigues(Vec3([0., 0., std::f32::consts::FRAC_PI_2]));
    near(mat_vec(r, Vec3([1., 0., 0.])), Vec3([0., 1., 0.]));
}
#[test]
fn sync_uses_explicit_correspondences_and_rejects_bad_ranges() {
    let source: Vec<_> = (0..10)
        .map(|i| MultiPose3D::from([(i, Pose3D::new())]))
        .collect();
    let synced = synchronize(&source, &[(1, 2), (5, 6)]).unwrap();
    assert!(synced[0].is_empty());
    assert!(synced[1].contains_key(&2));
    assert!(synced[4].contains_key(&5));
    assert!(synchronize(&source, &[(1, 2), (1, 6)]).is_err());
    assert!(synchronize(&source, &[(1, 20), (5, 30)]).is_err());
}
#[test]
fn correction_is_explicit_and_does_not_remap_axes() {
    let mut poses = MultiPose3D::from([(0, Pose3D::from([(0, Vec3([1., 2., 3.]))]))]);
    correct(&mut poses, "None").unwrap();
    near(poses[&0][&0], Vec3([1., 2., 3.]));
    correct(&mut poses, "Shelf").unwrap();
    near(poses[&0][&0], Vec3([1., 2., 3.05]));
    assert!(correct(&mut poses, "bogus").is_err());
}
#[test]
fn missing_mpjpe_is_null_not_zero() {
    let gt = Pose3D::from([(0, Vec3([0., 0., 1.])), (2, Vec3([1., 0., 1.]))]);
    let empty = evaluate(&gt, &MultiPose3D::new());
    assert_eq!(empty.mpjpe_mm, None);
    let missing = evaluate(
        &gt,
        &MultiPose3D::from([(0, Pose3D::from([(0, Vec3([0., 0., 1.]))]))]),
    );
    assert_eq!(missing.mpjpe_mm, None);
    assert_eq!(missing.pck, Some(0.5));
    assert_eq!(missing.predicted_gt_joints, 1);
    let same = evaluate(&gt, &MultiPose3D::from([(0, gt.clone())]));
    assert_eq!(same.mpjpe_mm, Some(0.));
    assert_eq!(same.pck, Some(1.));
}
#[test]
fn actual_bundled_shelf_data_is_read_only_and_reconstructs() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = root.join("dataset/shelf/dataset.json");
    let config: Config = serde_json::from_value(json(&path).unwrap()).unwrap();
    let data = load(&path, &config).unwrap();
    assert_eq!(data.frames.len(), 301);
    assert_eq!(data.frames[0].0, 300);
    assert_eq!(data.frames[0].1.len(), 5);
    let params: Params =
        serde_json::from_value(json(&root.join("dataset/shelf/algorithm.json")).unwrap()).unwrap();
    let mut tracker = Tracker::new(params).unwrap();
    let (result, _) = tracker
        .process_corrected(&data.frames[0].1, "Shelf")
        .unwrap();
    assert!(!result.is_empty());
    assert!(result.values().all(|p| p.values().all(|v| v.finite())));
}
#[test]
fn confidence_overflow_is_reported_not_serialized_as_null() {
    let (mut views, _) = scene();
    for view in &mut views {
        for js in view.joints.values_mut() {
            for j in js {
                j.conf = f32::MAX;
            }
        }
    }
    assert!(
        Tracker::new(Params::default())
            .unwrap()
            .process(&views)
            .is_err()
    );
}
#[test]
fn independent_sequences_and_runs_are_deterministic() {
    let (views, _) = scene();
    let mut a = Tracker::new(Params::default()).unwrap();
    let mut b = Tracker::new(Params::default()).unwrap();
    for _ in 0..35 {
        assert_eq!(a.process(&views).unwrap().0, b.process(&views).unwrap().0);
    }
}

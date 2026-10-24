#!/usr/bin/env python3
"""Differential check of the original C++ headers and the Rust library/CLI.

Requires clang++ or CXX and Cargo. Optional --opencv-check requires numpy/cv2
in this validation interpreter only; the Rust product never links OpenCV.
"""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOINTS = [0,4,3,2,7,6,5,1,11,10,9,14,13,12,8]

def run(args, **kw):
    return subprocess.run([str(x) for x in args], check=True, text=True, **kw)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'dataset/shelf/dataset.json')
    parser.add_argument('--algorithm', type=Path, default=ROOT/'dataset/shelf/algorithm.json')
    parser.add_argument('--limit', type=int, default=30)
    parser.add_argument('--tolerance', type=float, default=0.0001)
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--canonical-order', action='store_true', help='Normalize only C++ hash-map/tied-score traversal; report distinctly from unmodified reference')
    parser.add_argument('--opencv-check', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if args.limit <= 0 or not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error('limit/tolerance must be positive')
    config=json.loads(args.dataset.read_text())
    params=json.loads(args.algorithm.read_text())
    assert params.get('joint_types',DEFAULT_JOINTS)==DEFAULT_JOINTS, 'reference harness currently supports the upstream default 15-joint order'
    params['use_temporal_information']=args.temporal
    people=params.get('max_person_num') or config['detection_max_person_num']
    run(['cargo','build','--manifest-path',ROOT/'Cargo.toml','--release','--bins','--examples','--locked','--offline'])
    frames=[json.loads(l) for l in run([ROOT/'target/release/examples/dump_dataset',args.dataset],capture_output=True).stdout.splitlines()][:args.limit]
    assert frames, 'no input frames'
    calibration_error=None
    if args.opencv_check:
        import cv2
        import numpy as np
        cameras=json.loads((args.dataset.parent/config['camera_load_path']).read_text())
        calibration_error=0.0
        for view, (_, cam) in zip(frames[0]['views'],sorted(cameras.items()),strict=True):
            k=np.array(cam['K'],dtype=np.float32).reshape(3,3)
            d=np.array(cam.get('distCoeff',[0]*5),dtype=np.float32)
            expected,_=cv2.getOptimalNewCameraMatrix(k,d,tuple(cam['imgSize']),cam.get('rectifyAlpha',0))
            delta=float(np.max(np.abs(expected-np.array(view['camera']['k']))))
            calibration_error=max(calibration_error,delta)
            assert delta<0.001, (delta,expected,view['camera']['k'])
    p=lambda key,default:params.get(key,default)
    fields=[len(frames),config['detection_view_num'],people,int(args.temporal),{'None':0,'Shelf':1,'4DA':2}[config.get('result_correction','None')],int(p('use_bone_length_score',True)),int(p('use_edge_num_in_association',False)),p('conf_discard_threshold',0),p('conf_exponent',1),p('max_joint_radius',0.05),p('max_joint_cluster_radius',0.1),p('association_score_threshold',5),p('pose_discard_threshold',10),p('tracking_discard_threshold',10),p('tracking_distance',0.1),p('lost_track_penalty',0.5)]
    for frame in frames:
        for view in frame['views']:
            cam=view['camera']
            fields.extend(x for key in ('k','r') for row in cam[key] for x in row)
            fields.extend(cam['t'])
            fields.append(len(view['joints']))
            for kind,joints in sorted(view['joints'].items(),key=lambda p:int(p[0])):
                fields.extend([kind,len(joints)])
                for joint in joints: fields.extend([*joint['point'],joint['conf'],joint['uid']])
    with tempfile.TemporaryDirectory(prefix='twinpose-reference-') as temp:
        temp=Path(temp)
        exe=temp/'reference'
        run([os.environ.get('CXX','clang++'),'-std=c++20','-O2',*(['-DCANONICAL_ORDER'] if args.canonical_order else []),'-I',ROOT/'src','-I',ROOT/'libraries',ROOT/'rust/tools/reference.cpp','-o',exe])
        native=run([exe],input=' '.join(map(str,fields)),capture_output=True).stdout.splitlines()
        algorithm=temp/'algorithm.json'; algorithm.write_text(json.dumps(params))
        # Stream is deliberately uncorrected; apply the exact dataset correction below.
        output=run([ROOT/'target/release/twinpose','stream','--algorithm',algorithm],input=''.join(json.dumps(f)+'\n' for f in frames),capture_output=True).stdout.splitlines()
        # Dataset runner separately validates corrected temporal-history semantics.
        path=temp/'result.jsonl'
        run([ROOT/'target/release/twinpose','run','--dataset',args.dataset,'--algorithm',algorithm,'--output',path,'--limit',args.limit],capture_output=True)
        output=path.read_text().splitlines()
    assert len(native)==len(output)==len(frames)
    maximum=0.0; compared=0; id_permutations=0; mismatches=[]
    for index,(cpp,rust) in enumerate(zip(native,output,strict=True)):
        it=iter(cpp.split()); assert int(next(it))==index
        expected={}
        for _ in range(int(next(it))):
            pid=int(next(it)); pose={}
            for _ in range(int(next(it))):
                kind=next(it); pose[kind]=[float(next(it)) for _ in range(3)]
            expected[pid]=pose
        actual=json.loads(rust)['poses']

        if len(expected)!=len(actual):
            mismatches.append({'frame_index':index,'frame_id':frames[index]['frame_id'],'max_distance':None,'expected':list(expected.values()),'actual':list(actual.values())})
            continue
        exp=list(expected.values()); act=list(actual.values())
        def error(order):
            worst=0.0
            for a,j in zip(exp,order):
                b=act[j]
                if a.keys()!=b.keys(): return math.inf
                for kind in a: worst=max(worst,math.dist(a[kind],b[kind]))
            return worst
        order=min(itertools.permutations(range(len(act))),key=error)
        delta=error(order)

        if not math.isfinite(delta) or delta>args.tolerance: mismatches.append({'frame_index':index,'frame_id':frames[index]['frame_id'],'max_distance':delta if math.isfinite(delta) else None,'expected':exp,'actual':act})
        if math.isfinite(delta): maximum=max(maximum,delta)
        compared+=sum(len(p) for p in exp)
        if any(str(eid)!=list(actual)[j] for eid,j in zip(expected,order)): id_permutations+=1
    report={'dataset':str(args.dataset),'upstream':'fc684f1e2f487fac1e4005b8520b02a79b6458a2','frames':len(frames),'compared_joints':compared,'temporal':args.temporal,'cpp_canonical_order':args.canonical_order,'max_joint_distance':maximum,'tolerance':args.tolerance,'frames_with_permuted_person_ids':id_permutations,'max_camera_matrix_element_error':calibration_error,'mismatch_frames':[{'frame_index':v['frame_index'],'frame_id':v['frame_id'],'max_distance':v['max_distance']} for v in mismatches],'note':'Person IDs are frame-local association ranks; equal-score order differs between C++ unordered_map and Rust BTreeMap. This is a port consistency test, not physical accuracy acceptance.'}
    text=json.dumps(report,indent=2,allow_nan=False)
    print(text)
    if args.report:
        args.report.write_text(text+'\n')
        args.report.with_suffix('.mismatches.json').write_text(json.dumps(mismatches,indent=2,allow_nan=False)+'\n')
    if mismatches: raise SystemExit(1)

if __name__=='__main__':main()

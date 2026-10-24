// Copyright (c) 2026 HYPERTHEORY. MIT; see LICENSE.
// Validation only: compile original algorithm headers, never linked by Rust.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <deque>
#include <iostream>
#include <iomanip>
#include <unordered_set>
#include "algorithm/PoseBuilder.h"
#include "algorithm/OpenPoseBody25.h"
#include "dataset/Correction.h"
// This is the original Ink3D constructor (ink.cpp:295). All needed vector/matrix
// methods are header-only; no renderer, OpenCV, SDL or GL library is linked.
ink::Ray::Ray(const ink::Vec3& o,const ink::Vec3& d):origin(o),direction(d){}
#ifdef CANONICAL_ORDER
// Only remove platform-dependent unordered-map/tied-score traversal. Arithmetic,
// scores, thresholds, clustering and DP remain the original C++ implementation.
struct CanonicalBuilder : v3::PoseBuilder {
    using PoseBuilder::PoseBuilder;
    static bool less_ids(v3::PackedIntArray a,v3::PackedIntArray b){return std::lexicographical_compare(v3::as_array(a),v3::as_array(a)+16,v3::as_array(b),v3::as_array(b)+16);}
    void build(){
        init();build_joint_lists();
        for(auto& list:joint_list_per_type)std::sort(list.begin(),list.end(),[](const Joint& a,const Joint& b){return a.score!=b.score?a.score>b.score:less_ids(a.person_id_per_view,b.person_id_per_view);});
        cluster_joints();
        for(auto& list:cluster_list_per_type)std::sort(list.begin(),list.end(),[](const JointCluster& a,const JointCluster& b){return less_ids(a.person_id_per_view,b.person_id_per_view);});
        associate_views();
        std::sort(association_infos.begin(),association_infos.end(),[](auto& a,auto& b){return a.second!=b.second?a.second>b.second:less_ids(a.first,b.first);});
        build_pose_graphs();if(t_params.history_multi_pose)track_poses();resolve_pose_graphs();
    }
};
#endif
int main(){using namespace v3;int frame_count,view_count,temporal,correction;PoseBuilderParams p;PoseTrackingParams tp;
    std::cin>>frame_count>>view_count>>p.max_person_num>>temporal>>correction;
    std::cin>>p.use_bone_length_score>>p.use_edge_num_in_association>>p.conf_discard_threshold>>p.conf_exponent>>p.max_joint_radius>>p.max_joint_cluster_radius>>p.association_score_threshold>>p.pose_discard_threshold>>tp.tracking_discard_threshold>>tp.tracking_distance>>tp.lost_track_penalty;
    p.skeleton_tree=OpenPoseBody25;p.ordered_joint_types={0,4,3,2,7,6,5,1,11,10,9,14,13,12,8};
    float mins[]={0.,.4,.05,.1,.05,.05,.1,.05,0.,0.,.2,.2,0.,.2,.2};float maxs[]={.5,.75,.3,.5,.5,.3,.5,.5,0.,.25,.55,.6,.25,.55,.6};
    std::deque<std::vector<std::vector<float>>> history;MultiPose3D previous;
    std::cout<<std::setprecision(9);
    for(int frame=0;frame<frame_count;++frame){std::vector<Camera> cameras; cameras.reserve(view_count);MultiView views;
        for(int v=0;v<view_count;++v){ink::Mat3 k,r;ink::Vec3 t;for(int i=0;i<3;++i)for(int j=0;j<3;++j)std::cin>>k[i][j];for(int i=0;i<3;++i)for(int j=0;j<3;++j)std::cin>>r[i][j];std::cin>>t.x>>t.y>>t.z;cameras.emplace_back(k,r,t);View view(&cameras.back());int kinds;std::cin>>kinds;for(int i=0;i<kinds;++i){int type,num;std::cin>>type>>num;for(int j=0;j<num;++j){Joint2D joint;std::cin>>joint.point.x>>joint.point.y>>joint.conf>>joint.uid;view.add_joint(type,joint);}}view.precompute_rays();views.add_view(std::move(view));}
        if(!std::cin){std::cerr<<"truncated fixture\n";return 2;}
        std::vector<AnomalyDetector> detectors(15);p.bone_lengths.clear();for(int k=0;k<15;++k){auto& d=detectors[k];d.init_cutoff_hard_thresholds(mins[k],maxs[k]);d.init_evaluation_params(2,3,.5);std::vector<float> samples;for(auto& h:history)samples.insert(samples.end(),h[k].begin(),h[k].end());if(!samples.empty())d.init_normal_distribution(samples);p.bone_lengths.push_back(&d);}
        tp.history_multi_pose=temporal&&frame>0?&previous:nullptr;
#ifdef CANONICAL_ORDER
        CanonicalBuilder builder(&views,p,tp);
#else
        PoseBuilder builder(&views,p,tp);
#endif
        builder.build();std::vector<std::vector<float>> samples(15);for(auto& [id,pose]:builder.result)for(int a=0;a<15;++a)for(int b:OpenPoseBody25[a])if(b<15&&pose.has_joint(a)&&pose.has_joint(b))samples[b].push_back(pose.get_joint(a).distance(pose.get_joint(b)));history.push_back(samples);if(history.size()>30)history.pop_front();
        if(correction==1)dataset::correct_shelf(builder.result);if(correction==2)dataset::correct_4da(builder.result);previous=builder.result;
        std::cout<<frame<<" "<<std::distance(builder.result.begin(),builder.result.end());
        for(auto& [id,pose]:builder.result){std::cout<<" "<<id<<" "<<std::distance(pose.begin(),pose.end());for(auto& [kind,point]:pose)std::cout<<" "<<kind<<" "<<point.x<<" "<<point.y<<" "<<point.z;}std::cout<<"\n";
    }
}

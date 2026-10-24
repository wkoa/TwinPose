# Notices and numerical references

The Rust port is based on TwinPose's MIT-licensed source, copyright (c) 2026
HYPERTHEORY. The validation-only Ray constructor and reference math retain
Ink3D's copyright (c) 2021–2023 HYPERTHEORY (MIT; see libraries/ink/ink.h).
The upstream LICENSE and per-file notices are retained. Cargo dependencies
retain their own licenses and are locked in Cargo.lock.

The independently written Brown-Conrady inverse and rectified-camera viewport
calculation follow the numerical conventions of OpenCV's `undistortPoints` and
`getOptimalNewCameraMatrix` (five inverse iterations and nine samples per edge):
https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/calibration_base.cpp
https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/undistort.dispatch.cpp

For provenance, retain the source's BSD notice:

Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
Copyright (C) 2009, Willow Garage Inc., all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* The name of the copyright holders may not be used to endorse or promote
  products derived from this software without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall the Intel Corporation or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages (including,
but not limited to, procurement of substitute goods or services; loss of use,
data, or profits; or business interruption) however caused and on any theory of
liability, whether in contract, strict liability, or tort (including negligence
or otherwise) arising in any way out of the use of this software, even if advised
of the possibility of such damage.

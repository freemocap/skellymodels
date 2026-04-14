# SkellyModels Heirarchy and definitions

## Tracked points
- Produced by a tracker (i.e. from skellytracker)
- Presumed 3d triangulation of 2d points from tracker

## Mapping
- Maps Tracked Points (from a tracker) to Keypoints (i.e. anatomical keypoints)
- Defines direct mappings or weighted sums of Tracked points to define all required keypoints of a skeleton

## Keypoints
- As in anatomical keypoints
- Defined anatomical points on a Rigid body (i.e. greater trochanter)

## Rigid Bodies
- a set of two or more Keypoints whose distance never changes
- One keypoint is defined as the 'parent' and represents the origin of the RB
- A DIFFERENT keypoint defines the primary axis (usually +X)
- An optional 3rd keypoint defins a secondary axis (usually +Y, most not be colinear with parent and primary keypoints)
- Types: 
  - **standard**: 3 or more non-colinear points. Defines origin, and full XYZ orientation. basis vectors calcualted by Gram-Schmidt method  
  - **simple**: 2 or more colinear points. Define origin and primary axis, but cannot define secondary axis or 'roll' around primary axis. Re-orients with SLERP to minimize roll around primary axis

## Linkage
- Connection of two or more rigid bodies that share exactly ONE keypoint
- One RB is the 'parent' and the others are children. Parent RB defines the reference frame of the linkage, for joint angle calcuation. Parent keypoint of parent RB defines origin of the linkage, primary axis of RB defines 0-angle of linkage
- Types: 
  - 1-1: connection between two RBs (one parent, one child)
  - 1-many: connection between 3 or more RB's (one parent, many children)

# Chain
- A set of two or more linkages that share a rigid body.
- One RB is defined as the root/parent of the chain, and must be at the END of the chain
- All linkages in a chain must be 1:1

# Skeleton
A set of two or more linkages that shre a rigid body, at least one of which is is a 1:many type of linkage

内部：

visual_branch.extract_vision_bev(batch)
 -> vision_bev [B,80,H,W]

lidar_branch.extract_lidar_bev(batch)
 -> lidar_bev [B,128,H,W]

BEVDeformableFusion(vision_bev, lidar_bev)
 -> fused_bev [B,128,H,W]

fusion 内部关键过程：

vision_bev [B,80,H,W] -> visual_adapter -> Fv0 [B,128,H,W]
lidar_bev  [B,128,H,W] -> lidar_adapter -> Fl0 [B,128,H,W]

Fv0 -> visual_pyramid:  H, H/2, H/4, H/8
Fl0 -> lidar_pyramid:   H, H/2, H/4, H/8

Fv0 query + lidar_pyramid value -> lidar_to_visual Ov
Fl0 query + visual_pyramid value -> visual_to_lidar Ol

concat[Fv0, Fl0, Ov, Ol] -> conv fusion -> fused_bev [B,128,H,W]

然后：

fused_bev
 -> run_ring_sharp_downstream(...)

downstream：

fused_bev [B,128,H,W]
 -> Radon Transform
 -> sinogram
 -> VL occ_conv_yaw
 -> row FFT
 -> spec
 -> SpecGlobalDescriptorHead(spec)
 -> global descriptor [B,256]
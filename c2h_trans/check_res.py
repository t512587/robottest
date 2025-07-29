import pyrealsense2 as rs

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()

depth_intrinsics = depth_stream.get_intrinsics()
color_intrinsics = color_stream.get_intrinsics()

print("Depth Camera Intrinsics:")
print(f"  Width: {depth_intrinsics.width}")
print(f"  Height: {depth_intrinsics.height}")
print(f"  PPX (Principal Point X): {depth_intrinsics.ppx}")
print(f"  PPY (Principal Point Y): {depth_intrinsics.ppy}")
print(f"  Fx (Focal Length X): {depth_intrinsics.fx}")
print(f"  Fy (Focal Length Y): {depth_intrinsics.fy}")
print(f"  Distortion Model: {depth_intrinsics.model}")
print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")

print("\nColor Camera Intrinsics:")
print(f"  Width: {color_intrinsics.width}")
print(f"  Height: {color_intrinsics.height}")
print(f"  PPX (Principal Point X): {color_intrinsics.ppx}")
print(f"  PPY (Principal Point Y): {color_intrinsics.ppy}")
print(f"  Fx (Focal Length X): {color_intrinsics.fx}")
print(f"  Fy (Focal Length Y): {color_intrinsics.fy}")
print(f"  Distortion Model: {color_intrinsics.model}")
print(f"  Distortion Coefficients: {color_intrinsics.coeffs}")

pipeline.stop()

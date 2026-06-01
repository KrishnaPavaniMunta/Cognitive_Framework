from openni import openni2
import math

# Use your working Redist path
REDIST_PATH = r"C:\Orbbec\OpenNI-Windows-x64-2.3.0.86\Redist"
openni2.initialize(REDIST_PATH)

dev = openni2.Device.open_any()
# Ensure alignment is ON so we pull the registered parameters
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

depth_stream = dev.create_depth_stream()
depth_stream.start()

# 1. Check Resolution
video_mode = depth_stream.get_video_mode()
w, h = video_mode.resolutionX, video_mode.resolutionY

# 2. Check Raw Field of View (in Radians)
hfov = depth_stream.get_horizontal_fov()
vfov = depth_stream.get_vertical_fov()

# 3. Calculate fx, fy, cx, cy based on OpenNI profile
fx = w / (2.0 * math.tan(hfov / 2.0))
fy = h / (2.0 * math.tan(vfov / 2.0))
cx = w / 2.0
cy = h / 2.0

print("--- ORBBEC OPENNI INTRINSICS CHECK ---")
print(f"Active Profile Resolution: {w}x{h}")
print(f"Calculated Focal Length fx: {fx:.2f}")
print(f"Calculated Focal Length fy: {fy:.2f}")
print(f"Principal Point Center cx: {cx:.1f}")
print(f"Principal Point Center cy: {cy:.1f}")

depth_stream.stop()
openni2.unload()

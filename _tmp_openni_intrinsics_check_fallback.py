from openni import openni2
import math

REDIST_PATH = r"C:\Users\Krishna.Munta\Downloads\Orbbec_OpenNI_v2.3.0.86-beta6_windows_release\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs"
openni2.initialize(REDIST_PATH)

dev = openni2.Device.open_any()

# Registration setting can fail on some devices/drivers; continue without it.
try:
    dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)
except Exception as e:
    print(f"[WARN] set_image_registration_mode failed: {e}")

depth_stream = dev.create_depth_stream()
depth_stream.start()

video_mode = depth_stream.get_video_mode()
w, h = video_mode.resolutionX, video_mode.resolutionY

hfov = depth_stream.get_horizontal_fov()
vfov = depth_stream.get_vertical_fov()

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

from picamera2 import Picamera2, Preview
import libcamera


picam2 = Picamera2(0)
main_stream = {}
lores_stream = {"size": (800, 600)}
picam2.start_preview(Preview.QTGL)
my_config = picam2.create_video_configuration(main_stream, lores_stream, transform=libcamera.Transform(vflip=1, hflip=1), display="lores")
picam2.configure(my_config)


picam2.start_and_record_video("video_took/test_video.mp4", duration=1)
	

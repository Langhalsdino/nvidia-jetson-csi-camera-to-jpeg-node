import os
import time
import cv2
import logging

import make87 as m87
from make87_messages.core.header_pb2 import Header
from make87_messages.image.compressed.image_jpeg import ImageJPEG

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
):
    """
    Constructs a GStreamer pipeline string for the CSI Jetson camera using nvargus.
    Make sure that nvargus-daemon is running in the background.
    """
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

class CameraPublisher:
    def __init__(self, source, use_gstreamer=False, topic_name="IMAGE_DATA", message_type=ImageJPEG):
        """
        Initializes the camera publisher.
        
        Parameters:
            source: If using V4L2, this is the device path. It is ignored if use_gstreamer is True.
            use_gstreamer (bool): If True, the CSI camera is opened using the GStreamer pipeline.
            topic_name (str): The topic name for the publisher.
            message_type: The message type to publish (e.g. ImageJPEG).
        """
        self.use_gstreamer = use_gstreamer
        if self.use_gstreamer:
            # Use the GStreamer pipeline for the CSI camera on Jetson
            pipeline = gstreamer_pipeline()
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open camera device with GStreamer pipeline")
            logging.info("Camera opened using GStreamer pipeline.")
        else:
            # Use a traditional V4L2 device
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera device: {source}")
            logging.info("Camera opened using device path: %s", source)
        
        # Create a publisher topic using make87 with the ImageJPEG message type
        self.publisher = m87.get_publisher(name=topic_name, message_type=message_type)

    def publish_frames(self):
        """
        Captures frames from the camera, encodes them as JPEG,
        and publishes them using the ImageJPEG message.
        """
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.error("Failed to capture frame. Exiting capture loop.")
                break

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.error("Failed to encode frame as JPEG.")
                continue

            jpeg_buffer = buffer.tobytes()
            # Create a header; you can add more metadata if required.
            header = m87.create_header(Header, entity_path="/")
            # Create the ImageJPEG message with the header and JPEG data
            message = ImageJPEG(header=header, data=jpeg_buffer)

            # Publish the message using make87
            self.publisher.publish(message)
            logging.info("Published a JPEG frame.")

            # Delay to control the frame rate (approximately 30 FPS)
            time.sleep(0.033)

    def release(self):
        """Releases the camera device."""
        self.cap.release()

def get_v4l_devices(device_folder="/dev/v4l/by-path/"):
    """
    Returns a sorted list of full device paths found in the given folder.
    """
    if not os.path.isdir(device_folder):
        logging.error("Device folder %s does not exist.", device_folder)
        return []
    devices = os.listdir(device_folder)
    devices.sort()
    return [os.path.join(device_folder, device) for device in devices]

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    m87.initialize()

    # Set this flag True to use the CSI Jetson camera via the NVARGUS GStreamer pipeline.
    use_gstreamer = True

    if use_gstreamer:
        logging.info("Using GStreamer pipeline for CSI Jetson camera via nvargus.")
        try:
            # The source parameter is not used in gstreamer mode.
            publisher = CameraPublisher(source=None, use_gstreamer=True)
            publisher.publish_frames()
        except Exception as e:
            logging.error("Error with GStreamer camera: %s", e, exc_info=True)
        finally:
            try:
                publisher.release()
            except Exception as e:
                logging.error("Error releasing camera: %s", e, exc_info=True)
    else:
        # Fallback: Loop over V4L2 devices found in the specified folder.
        device_folder = "/dev/v4l/by-path/"
        while True:
            devices = get_v4l_devices(device_folder)
            if not devices:
                logging.warning("No V4L2 devices found in %s. Retrying in 1 second...", device_folder)
                time.sleep(1)
                continue

            for device_path in devices:
                logging.info("Attempting to use device: %s", device_path)
                try:
                    publisher = CameraPublisher(device_path)
                    publisher.publish_frames()
                except Exception as e:
                    logging.error("Error with device %s: %s", device_path, e, exc_info=True)
                finally:
                    try:
                        publisher.release()
                    except Exception as e:
                        logging.error("Error releasing device %s: %s", device_path, e, exc_info=True)
                logging.info("Waiting 1 second before trying the next device...")
                time.sleep(1)

if __name__ == '__main__':
    main()

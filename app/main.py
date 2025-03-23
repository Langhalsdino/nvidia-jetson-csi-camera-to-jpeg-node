import os
import time
import cv2
import logging
from typing import List, Optional

import numpy as np
import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import make87 as m87
from make87_messages.core.header_pb2 import Header
from make87_messages.image.compressed.image_jpeg import ImageJPEG


def gstreamer_pipeline(
    capture_width: int = 1280,
    capture_height: int = 720,
    display_width: int = 1280,
    display_height: int = 720,
    framerate: int = 60,
    flip_method: int = 0,
) -> str:
    """
    Constructs and returns a GStreamer pipeline string for the CSI camera using nvargus.

    Note:
        Ensure that the nvargus-daemon is running in the background.

    Returns:
        A GStreamer pipeline string.
    """
    pipeline_str = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink name=sink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
    return pipeline_str


class CameraPublisher:
    def __init__(
        self,
        source: Optional[str],
        use_gstreamer: bool = False,
        topic_name: str = "IMAGE_DATA",
        message_type=ImageJPEG,
        capture_width: int = 1280,
        capture_height: int = 720,
        display_width: int = 1280,
        display_height: int = 720,
        framerate: int = 60,
        flip_method: int = 0,
    ) -> None:
        """
        Initializes the CameraPublisher.

        Parameters:
            source: If not using GStreamer, this is the device path (e.g., /dev/v4l/by-path/xyz).
                    Ignored when use_gstreamer is True.
            use_gstreamer: If True, the CSI camera is opened using the GStreamer pipeline.
            topic_name: The topic name for the publisher.
            message_type: The message type to publish (e.g. ImageJPEG).
            capture_width: The capture width for the CSI camera.
            capture_height: The capture height for the CSI camera.
            display_width: The width of the displayed image.
            display_height: The height of the displayed image.
            framerate: The framerate to capture the video.
            flip_method: The flip method (0 means no rotation).
        """
        self.use_gstreamer: bool = use_gstreamer

        if self.use_gstreamer:
            Gst.init(None)
            pipeline_str: str = gstreamer_pipeline(
                capture_width, capture_height, display_width, display_height, framerate, flip_method
            )
            self.pipeline: Gst.Pipeline = Gst.parse_launch(pipeline_str)
            self.appsink = self.pipeline.get_by_name("sink")
            if not self.appsink:
                raise RuntimeError("Cannot get appsink from the GStreamer pipeline.")
            # Start the pipeline.
            self.pipeline.set_state(Gst.State.PLAYING)
            self.display_width: int = display_width
            self.display_height: int = display_height
            logging.info("Camera opened using GStreamer pipeline.")
        else:
            # Fallback to using cv2.VideoCapture with a V4L2 device.
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera device: {source}")
            logging.info("Camera opened using device path: %s", source)

        # Create a publisher topic using make87 with the specified message type.
        self.publisher = m87.get_publisher(name=topic_name, message_type=message_type)

    def publish_frames(self) -> None:
        """
        Captures frames from the camera, encodes them as JPEG,
        and publishes them using the ImageJPEG message.
        """
        if self.use_gstreamer:
            while True:
                # Pull a sample from the appsink element.
                sample = self.appsink.emit("pull-sample")
                if sample is None:
                    logging.error("Failed to retrieve sample from GStreamer appsink. Exiting capture loop.")
                    break

                buf: Gst.Buffer = sample.get_buffer()
                success, map_info = buf.map(Gst.MapFlags.READ)
                if not success:
                    logging.error("Failed to map Gst buffer for reading.")
                    continue

                try:
                    # Convert the Gst buffer data to a NumPy array.
                    frame = np.frombuffer(map_info.data, dtype=np.uint8)
                    frame = frame.reshape((self.display_height, self.display_width, 3))
                except Exception as e:
                    logging.error("Error converting Gst buffer to numpy array: %s", e)
                    buf.unmap(map_info)
                    continue
                buf.unmap(map_info)

                # Encode the frame as JPEG.
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logging.error("Failed to encode frame as JPEG.")
                    continue
                jpeg_buffer = buffer.tobytes()

                # Create a header and ImageJPEG message.
                header = m87.create_header(Header, entity_path="/")
                message = ImageJPEG(header=header, data=jpeg_buffer)

                # Publish the message.
                self.publisher.publish(message)
                logging.info("Published a JPEG frame from GStreamer pipeline.")

                # Delay to control the frame rate (approximately 30 FPS).
                time.sleep(0.033)
        else:
            # Fallback: use cv2.VideoCapture to read frames.
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to capture frame from device. Exiting capture loop.")
                    break

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    logging.error("Failed to encode frame as JPEG.")
                    continue

                jpeg_buffer = buffer.tobytes()
                header = m87.create_header(Header, entity_path="/")
                message = ImageJPEG(header=header, data=jpeg_buffer)
                self.publisher.publish(message)
                logging.info("Published a JPEG frame from device.")

                time.sleep(0.033)

    def release(self) -> None:
        """
        Releases the camera resource.
        """
        if self.use_gstreamer:
            self.pipeline.set_state(Gst.State.NULL)
        else:
            self.cap.release()


def get_v4l_devices(device_folder: str = "/dev/v4l/by-path/") -> List[str]:
    """
    Returns a sorted list of full device paths found in the given folder.

    Parameters:
        device_folder: The folder to search for V4L2 devices.

    Returns:
        A list of device paths.
    """
    if not os.path.isdir(device_folder):
        logging.error("Device folder %s does not exist.", device_folder)
        return []
    devices: List[str] = os.listdir(device_folder)
    devices.sort()
    return [os.path.join(device_folder, device) for device in devices]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    m87.initialize()

    # Set use_gstreamer to True to use the CSI camera via the NVARGUS GStreamer pipeline.
    use_gstreamer: bool = True

    if use_gstreamer:
        logging.info("Using GStreamer pipeline for CSI Jetson camera via nvargus.")
        try:
            # The 'source' parameter is ignored when use_gstreamer is True.
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
        device_folder: str = "/dev/v4l/by-path/"
        while True:
            devices: List[str] = get_v4l_devices(device_folder)
            if not devices:
                logging.warning("No V4L2 devices found in %s. Retrying in 1 second...", device_folder)
                time.sleep(1)
                continue

            for device_path in devices:
                logging.info("Attempting to use device: %s", device_path)
                try:
                    publisher = CameraPublisher(source=device_path)
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

import pyrealsense2 as rs
import numpy as np
import cv2

#----------------------------------------------------------------
# Intel RealSense D405 - Depth Stream Heatmap
#----------------------------------------------------------------
# This script captures the depth stream from the D405 camera,
# applies a color heatmap to it, and displays the result in real-time.
#----------------------------------------------------------------

#----------------------------------------------------------------
# Intel RealSense D405 - Capture Depth Matrix at Controlled FPS
#----------------------------------------------------------------
# This script visualizes the depth stream in real-time while
# capturing and saving the raw depth matrix variable at a
# specified, slower frame rate.
#----------------------------------------------------------------

import pyrealsense2 as rs
import numpy as np
import cv2
import time # Import the time module

def main():
    # --- Configuration ---
    # The rate at which to capture the depth matrix (in frames per second)
    CAPTURE_FPS = 5
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    colorizer = rs.colorizer()

    print("Starting pipeline...")
    pipeline.start(config)
    print("Pipeline started.")

    # --- Timing variables for controlling capture rate ---
    interval = 1.0 / CAPTURE_FPS
    last_capture_time = 0

    try:
        while True:
            # This part runs at the full camera speed
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                continue

            # --- Real-time Visualization (runs every frame) ---
            # We colorize the depth frame for viewing purposes only
            colorized_depth_frame = colorizer.colorize(depth_frame)
            color_image = np.asanyarray(colorized_depth_frame.get_data())
            # Display the resulting frame
            cv2.imshow('RealSense Depth Heatmap', color_image)

            # --- Controlled FPS Data Capture ---
            current_time = time.time()
            if current_time - last_capture_time >= interval:
                last_capture_time = current_time
                
                # Get the raw depth data and save it as a numpy array (matrix)
                # This is the variable you want to use for your processing
                depth_matrix = np.asanyarray(depth_frame.get_data())

                #
                # --- YOUR PROCESSING CODE GOES HERE ---
                #
                # For now, we just print a confirmation.
                # You can save the 'depth_matrix' to a file, run calculations, etc.
                # The values in this matrix are the distances in millimeters.
                #
                print(f"Captured depth matrix at {time.strftime('%H:%M:%S')}. "
                      f"Shape: {depth_matrix.shape}, "
                      f"Data Type: {depth_matrix.dtype}")
                #
                # Example: To get the distance of the center pixel:
                # h, w = depth_matrix.shape
                # center_x, center_y = w // 2, h // 2
                # center_distance = depth_matrix[center_y, center_x]
                # print(f"Distance at center: {center_distance} mm")
                #

            # Check for the 'q' key to exit the loop
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                print("'q' pressed, exiting...")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        print("Stopping pipeline...")
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Pipeline stopped and all windows closed.")

if __name__ == "__main__":
    main()
    
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################


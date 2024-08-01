"""
MIT BWSI Autonomous RACECAR
MIT License
racecar-neo-prereq-labs

File Name: template.py << [Modify with your own file name!]

Title: [PLACEHOLDER] << [Modify with your own title]

Author: [PLACEHOLDER] << [Write your name or team name here]

Purpose: [PLACEHOLDER] << [Write the purpose of the script here]

Expected Outcome: [PLACEHOLDER] << [Write what you expect will happen when you run
the script.]
"""

########################################################################################
# Imports
########################################################################################

import sys
from classification_model import load_model, infer
from sign_image import crop
import numpy as np
import skimage.transform


# If this file is nested inside a folder in the labs folder, the relative path should
# be [1, ../../library] instead.
sys.path.insert(1, '../../library')
import racecar_core


########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# Declare any global variables here


########################################################################################
# Functions
########################################################################################

# [FUNCTION] The start function is run once every time the start button is pressed
def start():
    load_model("traffic_sign_identification_edgetpu.tflite")
    # Remove 'pass' and write your source code for the start() function here


# [FUNCTION] After start() is run, this function is run once every frame (ideally at
# 60 frames per second or slower depending on processing speed) until the back button
# is pressed  
def update():

    image = rc.camera.get_color_image()
    image = crop(image, 100)
    if image is None:
        pass
    else:
        rc.display.show_color_image(image)

        image = skimage.transform.resize(image, (32, 32, 3), mode='constant')

        image = np.float32(image)
    
        image = np.array(image)
        
        image = np.expand_dims(image, axis=0)
        
        predicted_class, confidence = infer(image)
    
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")


# [FUNCTION] update_slow() is similar to update() but is called once per second by
# default. It is especially useful for printing debug messages, since printing a 
# message every frame in update is computationally expensive and creates clutter
def update_slow():
    pass  # Remove 'pass and write your source code for the update_slow() function here


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()

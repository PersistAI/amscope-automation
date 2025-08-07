import os
from datetime import datetime
from flask import Flask, request, render_template, redirect, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from skimage import io, color, exposure, feature, filters
import matplotlib.pyplot as plt
import pyautogui
import time
from opcua import Server, ua
from pywinauto import application
from pywinauto.application import Application
import psutil
from pywinauto.findwindows import ElementNotFoundError
import json
from tucam_api import Tucam, TUCAM_IDPROP, TUCAM_Capa_SetValue, TUCAM_Prop_SetValue, TUCAMRET
from ctypes import c_void_p, c_double, c_int32, c_int
import logging
import zipfile
from misumi_xy_wrapper import MisumiXYWrapper, AxisName, DriveMode
import cv2
from brightfield_analysis import process_image, upload_and_extract_to_folder, CLAHE_images, check_and_upload_model, load_images_from_folder, calculate_and_update_statistics_in_txt_files, create_statistics_df
import torch


RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

app = Flask(__name__)
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image):
    global color_var, edge_density

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image / 255.0 if image.max() > 1 else image

    mask = gray > 0.05
    background = filters.gaussian(gray, sigma=25)
    gray_norm = np.clip(gray - background, 0, 1)
    gray_eq = exposure.equalize_hist(gray_norm)

    color_var = np.var(image[mask]) if image.ndim == 3 else 0
    edges = feature.canny(gray_eq, sigma=1.0)
    edge_density = np.sum(edges & mask) / np.sum(mask)
    score = 0
    if color_var > 500:
        score += 1
    if edge_density < 0.1:
        score += 1

    label = "crystalline" if score >= 1 else "amorphous"
    return label, edges

def generate_plot(image, edges, label, timestamp_prefix, plot_filename):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Classification: {label} | Color Var: {color_var:.2f}, Edge Density: {edge_density:.4f}",
        fontsize=14
    )

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    edge_overlay = image.copy()
    if edge_overlay.ndim == 2:
        edge_overlay = np.stack([edge_overlay]*3, axis=2)
    edge_overlay[edges] = [255, 0, 0]
    axes[1].imshow(edge_overlay)
    axes[1].set_title("Edges (Red Overlay)")
    axes[1].axis('off')

    axes[2].set_title("Color Histogram")
    if image.ndim == 3:
        for i, color_name in enumerate(['r', 'g', 'b']):
            hist, bins = np.histogram(image[:, :, i].ravel(), bins=256, range=(0, 255))
            hist = hist / hist.sum()
            axes[2].plot(bins[:-1], hist, color=color_name)
    else:
        hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 255))
        hist = hist / hist.sum()
        axes[2].plot(bins[:-1], hist, color='black')

    plt.tight_layout()
    plot_path = os.path.join(app.config['RESULTS_FOLDER'], plot_filename)
    plt.savefig(plot_path)
    plt.close()

#making analysis part of upload into a reusable function (for both auto upload and manual upload)
def analyze_image(filename, timestamp, file, well_name, sample_num): 
        ext = filename.rsplit('.', 1)[1].lower()
        original_filename = f"{timestamp}_original_{well_name}_S{sample_num}.{ext}"
        original_path = os.path.join(app.config['RESULTS_FOLDER'], original_filename)

        image_bytes = file.read()
        image_np = io.imread(image_bytes, plugin='imageio')
        io.imsave(original_path, image_np)

        label, edges = extract_features(image_np)
        plot_filename = f"{timestamp}_plot_{well_name}_S{sample_num}.png"

        generate_plot(image_np, edges, label, timestamp, plot_filename)

        return label, plot_filename


def serve_results(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

'''

need 2 functions: 
    - one for each type of analysis script

'''

def capture_amorphous_crystalline(data):
    results = []

    stage = None
    TUCAM_Prop_SetValue.argtypes = [c_void_p, c_int32, c_double, c_int32]
    TUCAM_Capa_SetValue.restype  = TUCAMRET

    #INITIALIZE THE XY STAGE, TRY cONNECTING TO COM3 AND COM4
    try:
        stage = MisumiXYWrapper(port='COM3')

    except Exception as e:
        print(f"COM4 failed: {e}")

        try: 
            stage = MisumiXYWrapper(port='COM4')
            
        except Exception as e:
            print(f"COM3 also failed: {e}")

    for well_data in data: 
        well_name = well_data.get("well")
        positions = well_data.get("sample-positions", [])

        for idx, coords in enumerate(positions):
            x = coords["x"]
            y = coords["y"]
            idx+=1
            print(f"Processing {well_name} sample #{idx} at ({x}, {y})")

            amscope = Tucam()

            try:
                stage.move_to_position({AxisName.X: x, AxisName.Y: y})
            except Exception as e: 
                print("Failed to move ", e)

            amscope.OpenCamera(0)

            #try to give camera a delay between opening the camera and taking the picture
            time.sleep(10)

            #try to change camera settings (on hold)
            try: 
                handle = c_void_p(amscope.TUCAMOPEN.hIdxTUCam)
                result = TUCAM_Prop_SetValue(handle, TUCAM_IDPROP.TUIDP_EXPOSURETM.value, c_double(200.0), 0)

                if result == 0: 
                    print("Successfully set values!")

            except Exception as e: 
                print("Cannot set the value: ", e)

            if amscope.TUCAMOPEN.hIdxTUCam != 0:
                amscope.SaveImageData() #this takes the picture
                print("Image captured!")
                amscope.CloseCamera()
                
            amscope.UnInitApi()

            # This needs to change based on what device is being used to run the script
            #This is where the captured images get stored
            image_path = r"C:\Users\ruyek\OneDrive\Desktop\Image"


            if not os.path.exists(image_path):
                return {"error": "Image folder not found"}

            files = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if os.path.isfile(os.path.join(image_path, f))
            ]

            if not files:
                return {"error": "No images found"}
        

            latest_image = max(files, key=os.path.getctime)
            print("Using latest image:", latest_image)

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Open file and pass to analyzer
            with open(latest_image, 'rb') as file:
                label, plot_filename = analyze_image(
                    filename=latest_image,
                    timestamp=timestamp,
                    file=file,
                    well_name = well_name,
                    sample_num = idx
                )
                results.append((label, plot_filename))

    #once everything is done, home the xy stage
    print("Homing all axes...")
    stage.home_all_axes(timeout=5)
    return results if results else {"error": "No samples processed"}


#This is manual upload script for the old flask app
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']

        print("This is file var: ", file)

        if file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = secure_filename(file.filename)
        print("This is filename var: ", filename)

        ext = filename.rsplit('.', 1)[1].lower()
        original_filename = f"{timestamp}_original.{ext}"
        original_path = os.path.join(app.config['RESULTS_FOLDER'], original_filename)

        image_bytes = file.read()
        image_np = io.imread(image_bytes, plugin='imageio')
        io.imsave(original_path, image_np)

        label, edges = extract_features(image_np)
        plot_filename = f"{timestamp}_plot.png"

        generate_plot(image_np, edges, label, timestamp, plot_filename)


        return render_template('result.html', label=label, plot_filename=plot_filename)

    return render_template('index.html')


#UASERVER LOGIC
#create method node
def add_amscope(server): 
    id = server.register_namespace("Amscope") #creates the Amscope object node name
    root = server.get_objects_node()
    amscope = root.add_object(id, "Amscope") #create new object node called Amscope in root directory


    def capture_amorphous_and_crystalline(parent, input_args): 
        #load json file
        data = [
            {
                "well": "A1",
                "sample-positions": [

                    #distance between center is approx 11000

                    {"x": 14000, "y": 12000},
                    {"x": 14000, "y": 23000},
                    {"x": 3000, "y": 23000},
                    {"x": 3000, "y": 12000}
                    
                ]
            }
        ]

        
        result = capture_amorphous_crystalline(data) #call the capture method with test data

        print("Capture result:", result)
        return [ua.Variant(str(result), ua.VariantType.String)]

    amscope.add_method(id, "capture_amorphous_crystalline", capture_amorphous_and_crystalline, [ua.VariantType.String], [ua.VariantType.String]) #This method takes in a string, and outputs a string

def main(): 
    #create the server object and set the endpoint where clients will connect
    server = Server()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/")
    add_amscope(server)
    server.start()

    try: 
        while True:  #keep the server running until it is manually stopped
            time.sleep(1) #this is needed bc it prevents loop from hogging CPU
    finally: 
        server.stop() #guarantees server stops when you interrupt itl

if __name__=="__main__": 
    main()
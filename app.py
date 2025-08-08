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
from brightfield_analysis import CLAHE_images, check_and_upload_model, load_images_from_folder, process_export_image, calculate_and_update_statistics_in_txt_files, extract_diameters_from_file, plot_diameter_histogram
import torch
from PIL import Image

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


model_path = os.path.join("content", "persist_model", "best.pt")
parent_path = os.path.join("results")
model = torch.hub.load('ultralytics/yolov5', 'custom', model_path)  # custom trained model

def tif_to_png(parent_path):

    for filename in os.listdir(parent_path): 
        if filename.lower().endswith('tif') or filename.lower().endswith('tiff'): 
            #conver to png
            image_path = os.path.join(parent_path, filename)
            tif_image = Image.open(image_path)

            new_filename = os.path.splitext(filename)[0] + ".png"
            tif_image.save(os.path.join(parent_path, new_filename))



def brightfield_analysis(): 
    tif_to_png(parent_path) #convert all tif files to png files

    CLAHE_images(parent_path)

    check_and_upload_model(model_path)    

    CLAHE_folder_path = os.path.join(parent_path, "CLAHE_images")

    image_list, image_filenames = load_images_from_folder(CLAHE_folder_path)

    print(f"Loaded {len(image_list)} images from the folder")

    model.conf = 0.20  # confidence threshold (0-1)
    model.iou = 0.45  # NMS IoU threshold (0-1)

    results = model(image_list, size=320)  # includes NMS

    print(results)

    annotated_imgs = results.render()
        
    model_inf_dir = os.path.join(parent_path, 'brightfield_output/model_inference')
    os.makedirs(model_inf_dir, exist_ok=True)

    for i, img in enumerate(annotated_imgs):
        # Extract original filename and add '_annotated' suffix
        original_filename = image_filenames[i]
        filename_without_extension = os.path.splitext(original_filename)[0]
        annotated_filename = filename_without_extension + '_annotated.jpg'

        # Define the save path
        save_path = os.path.join(model_inf_dir, annotated_filename)

        # Save the image
        cv2.imwrite(save_path, img)

        # Specify model path and folder paths
        # data_folder = '/content/data/dat_folder/CLAHE_images' # I've made it dynamic now, should be good... 2024Jan04 JMM
        output_folder_imgs = os.path.join(model_inf_dir,'imgs')
        postprocessed_imgs_path = os.path.join(parent_path,'postprocessed_imgs')


        os.makedirs(output_folder_imgs, exist_ok=True)
        os.makedirs(postprocessed_imgs_path, exist_ok=True)

    # Process each image
    for i, img_detections in enumerate(results.xyxy):
        process_export_image(os.path.join(CLAHE_folder_path, image_filenames[i]), img_detections.cpu().numpy(), postprocessed_imgs_path)

    # process_stats_path = '/content/data/dat_folder/output/export_images'
    calculate_and_update_statistics_in_txt_files(postprocessed_imgs_path)

    # List to store all diameters from all files for the cumulative plot
    all_diameters = []

    # Loop through each file, plot histogram, and collect diameters
    for file in os.listdir(postprocessed_imgs_path):
        if file.startswith('boundingboxes_') and file.endswith('.txt'):
            file_path = os.path.join(postprocessed_imgs_path, file)
            diameters = extract_diameters_from_file(file_path)
            all_diameters.extend(diameters)  # Add to cumulative list

            # Plot title
            base_name = file.replace('boundingboxes_', '').replace('.txt', '')
            title = f"{base_name} Microsphere Diameter Distribution Plot"
            save_path = os.path.join(postprocessed_imgs_path, f"{base_name}_diameter_distribution.png")
            plot_diameter_histogram(diameters, title, save_path)

    # Plot the cumulative histogram
    cumulative_title = "Cumulative Microsphere Diameter Distribution Plot"
    cumulative_save_path = os.path.join(postprocessed_imgs_path, "cumulative_diameter_distribution.png")
    plot_diameter_histogram(all_diameters, cumulative_title, cumulative_save_path)

    return {
        "cumulative_plot": cumulative_save_path,
        "postprocessed_imgs_path": postprocessed_imgs_path
    }

def zip_and_delete_image(root, zip_filename):  #root is the path to the results folder

    images_to_delete = []

    with zipfile.ZipFile(zip_filename, 'w') as zipf:  #opens a new zip file, zip_filename
        for current_root, dirs, files in os.walk(root): 
            for file in files: 
                if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.txt')): 
                    file_path = os.path.join(current_root, file)
                    zipf.write(file_path, arcname=file)
                    images_to_delete.append(file_path)

        dirs.clear()

    for file in images_to_delete: 
        try: 
            os.remove(file)
            print("Successfully removed original image")
        
        except Exception as e: 
            print("Failed to remove file: ", e)



def capture_brightfield(data): 
    results = []

    stage = None

    try:
        stage = MisumiXYWrapper(port='COM3')

    except Exception as e:
        print(f"COM4 failed: {e}")

        try: 
            stage = MisumiXYWrapper(port='COM4')
            
        except Exception as e:
            print(f"COM3 also failed: {e}")

    #move stage and take image at each well
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

            if amscope.TUCAMOPEN.hIdxTUCam != 0:

                amscope.SaveImageData() #this takes the picture
                print("Image captured!")
                amscope.CloseCamera()

            amscope.UnInitApi()

            image_path = r"C:\Users\ruyek\OneDrive\Desktop\Image"


            if not os.path.exists(image_path):
                return {"error": "Image folder not found"}


            files = [   
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if os.path.isfile(os.path.join(image_path, f))
            ]



            corrected_files = []
            for file in files: 
                if file.endswith(".tif.tif"): 
                    corrected_files.append(file[:-4])
                elif file.endswith(".tiff.tiff"):
                    corrected_files.append(file[:-5])   
                else: 
                    corrected_files.append(file)

            print(corrected_files)
            latest_image = max(files, key=os.path.getctime)
            print("Using latest image:", latest_image)

            

            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            ext = latest_image.rsplit('.', 1)[1].lower()
            original_filename = f"{timestamp}_brightfield.{ext}"
            # Open file and pass to analyzer
            original_path = os.path.join(app.config['RESULTS_FOLDER'], original_filename)

            with open(latest_image, 'rb') as file: 
                image_bytes = file.read()
                image_np = io.imread(image_bytes, plugin='imageio')
                io.imsave(original_path, image_np)

        
            brightfield_analysis() #goes through all the captured images and analyzes them

    print("Homing all axes...")
    stage.home_all_axes(timeout=5)

    zip_filename = f"{timestamp}_brightfield_original.zip"
    zip_and_delete_image(parent_path, zip_filename)
    return {"Finished brightfield analysis on all wells."}



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
    zip_filename = f"{timestamp}_amorphous_crystalline_original.zip"
    zip_and_delete_image(parent_path, zip_filename)
    return results if results else {"error": "No samples processed"}

#UASERVER LOGIC
#create method node
def add_amscope(server): 
    id = server.register_namespace("Amscope") #creates the Amscope object node name
    root = server.get_objects_node()
    amscope = root.add_object(id, "Amscope") #create new object node called Amscope in root directory

    def capture_brightfield_node(parent, input_args): 
        #load json file
        data = [
            {
                "well": "A1",
                "sample-positions": [

                    #distance between center is approx 11000

                    {"x": 14000, "y": 12000},
                    {"x": 14000, "y": 23000},
                    #{"x": 3000, "y": 23000},
                    #{"x": 3000, "y": 12000}
                    
                ]
            }
        ]

        
        result = capture_brightfield(data) #call the capture method with test data

        print("Capture result:", result)
        return [ua.Variant(str(result), ua.VariantType.String)]


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
    amscope.add_method(id, "capture_brightfield", capture_brightfield_node, [ua.VariantType.String], [ua.VariantType.String])


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
        server.stop() #guarantees server stops when you interrupt it

if __name__=="__main__": 
    main()
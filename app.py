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

stage = None


def init_hardware():
    global stage
    if stage is None:
        try:
            stage = MisumiXYWrapper(port='COM3')
        except:
            stage = MisumiXYWrapper(port='COM4')

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

def move_to_KX2(): 
    try:
        stage.move_to_position({AxisName.X: 100000, AxisName.Y: 0})
        result = ["Successfully moved to KX2 position."]
        logging.info("Returning from move_to_KX2_node: %s", result)
        return result
    except Exception as e:
        logging.exception("Move to KX2 failed")
        return [f"Error: {e}"]

def capture_brightfield(data): 
    results = []

    plate_letters = ['A', 'B', 'C', 'D']

    x_offset = 10320   # well-to-well spacing X
    y_offset = 10250   # well-to-well spacing Y
    x_mini_offset = 1500   # how much camera moves within a well
    y_mini_offset = 1500


    starting_well = data[0]
    well_name = starting_well.get("starting-well")
    positions = starting_well.get("starting-position", [])
    total_wells = int(starting_well.get("total-wells"))
    print("Currently on position: ", positions)

    starting_x = positions[0]["x"]
    starting_y = positions[0]["y"]

    print(f"starting x: {starting_x}")
    print(f"starting y: {starting_y}")

    if total_wells: 
        x_limit = 6
        y_limit = 4
        print(f"the x_limit is {x_limit}, y_limit is {y_limit}")

    # Initial stage position
    x_pos, y_pos = starting_x, starting_y

    well_count = 0

    for y_count in range(y_limit):
        for x_count in range(x_limit): 
            # Save well center
            well_x, well_y = x_pos, y_pos

            for idx in range(9): 
                with open("config.json", "r") as f: 
                    config = json.load(f)

                home = os.path.expanduser("~")
                dirty_image_path = os.path.join(home, config["image_path"])

                image_path = os.path.normpath(dirty_image_path)
                print(image_path)

                if not os.path.exists(image_path):
                    print("Creating image folder...")
                    os.makedirs(image_path, exist_ok=True)

                well_number = f"{plate_letters[y_count]}{x_count+1}"
                print(f"Well {well_number}, Sample {idx+1} – image {idx+1}/9 at ({x_pos}, {y_pos})")

                try:
                    stage.move_to_position({AxisName.X: x_pos, AxisName.Y: y_pos})
                except Exception as e: 
                    print("Failed to move ", e)

                amscope = Tucam()
                amscope.OpenCamera(0)


                if amscope.TUCAMOPEN.hIdxTUCam != 0:
                    amscope.SaveImageData()
                    print("Image captured!")
                    amscope.CloseCamera()
                    
                amscope.UnInitApi()

                files = [
                    os.path.join(image_path, f)
                    for f in os.listdir(image_path)
                    if os.path.isfile(os.path.join(image_path, f))
                ]
                
                if not files:
                    return {"error": "No images found"}

                latest_image = max(files, key=os.path.getctime)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ext = latest_image.rsplit('.', 1)[1].lower()
                original_filename = f"{timestamp}_brightfield_{well_number}_S{idx+1}.{ext}"
                original_path = os.path.join(app.config['RESULTS_FOLDER'], original_filename)

                with open(latest_image, 'rb') as file: 
                    image_bytes = file.read()
                    image_np = io.imread(image_bytes, plugin='imageio')
                    io.imsave(original_path, image_np)

                brightfield_analysis()

                
                if idx == 0: #north
                    print(f"idx: {idx}")
                    x_pos -= x_mini_offset

                elif idx == 1: #east
                    print(f"idx: {idx}")
                    x_pos += x_mini_offset
                    y_pos -= y_mini_offset

                elif idx == 2: #south
                    print(f"idx: {idx}")
                    y_pos += y_mini_offset
                    x_pos += x_mini_offset

                elif idx == 3: #west
                    print(f"idx: {idx}")
                    x_pos -= x_mini_offset
                    y_pos += y_mini_offset

                elif idx == 4: #northeast
                    print(f"idx: {idx}")
                    y_pos-=y_mini_offset * 2
                    x_pos -= x_mini_offset

                elif idx == 5: #southeast
                    print(f"idx: {idx}")
                    x_pos += x_mini_offset * 2

                elif idx == 6: #southwest (FIX)
                    print(f"idx: {idx}")
                    y_pos += y_mini_offset * 1.5

                elif idx == 7: #northwest
                    print(f"idx: {idx}") 
                    x_pos -= x_mini_offset * 2

            x_pos, y_pos = well_x, well_y

            x_pos -= x_offset
            well_count+=1

            if well_count >= total_wells: 
                break
        

        if well_count >= total_wells: 
            break


        y_pos -= y_offset
        x_pos = starting_x


    print("Homing all axes...")
    stage.home_all_axes(timeout=5)

    zip_filename = f"{timestamp}_brightfield_original.zip"
    zip_and_delete_image(parent_path, zip_filename)

    stage.disconnect()
    amscope.CloseCamera()
    amscope.UnInitApi()
    return ["Finished brightfield analysis on all wells."]


def capture_amorphous_crystalline(data):
    results = []

    #NOTE: 500steps = 1mm
    
    x_offset = 10320 #how much camera moves between wells
    y_offset = 10250

    x_mini_offset = 1500 #how much camera moves within a well
    y_mini_offset = 1500


    starting_well = data[0]
    well_name = starting_well.get("starting-well")
    positions = starting_well.get("starting-position", [])
    total_wells = int(starting_well.get("total-wells"))
    print("Currently on position: ", positions)

    starting_x = positions[0]["x"]
    starting_y = positions[0]["y"]

    print(f"starting x: {starting_x}")
    print(f"starting y: {starting_y}")

    if total_wells:  #Assuming we use 24 wells for everything

            x_limit = 6
            y_limit = 4
            print(f"the x_limit is {x_limit}, y_limit is {y_limit}")



    plate_letters = ['A', 'B', 'C', 'D']
    x_count = 0 #use these to determine if we hit the limit of the well
    y_count = 0

    x_pos = starting_x
    y_pos = starting_y

    well_count = 0

    #INITIALIZE THE XY STAGE, TRY cONNECTING TO COM3 AND COM4

    for y_count in range(y_limit):
        for x_count in range(x_limit):

            well_x = x_pos
            well_y = y_pos

            for idx in range(9): 
                with open("config.json", "r") as f: 
                    config = json.load(f)

                home = os.path.expanduser("~")
                dirty_image_path = os.path.join(home, config["image_path"])

                image_path = os.path.normpath(dirty_image_path)
                print(image_path)

                if not os.path.exists(image_path):
                    print("Creating image folder...")
                    os.makedirs(image_path, exist_ok=True)

                well_number = f"{plate_letters[y_count]}{x_count+1}"
                print(f"Processing {well_number} sample #{idx} at ({x_pos}, {y_pos})")
                
                
                try:
                    stage.move_to_position({AxisName.X: x_pos, AxisName.Y: y_pos})
                    time.sleep(0.5)
                except Exception as e: 
                    print("Failed to move ", e)
            
                amscope = Tucam()
                amscope.OpenCamera(0)

                if amscope.TUCAMOPEN.hIdxTUCam != 0:
                    amscope.SaveImageData() #this takes the picture
                    print("Image captured!")
                    amscope.CloseCamera()
                    
                amscope.UnInitApi()

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
                        well_name = well_number,
                        sample_num = idx+1
                    )
                    results.append((label, plot_filename))


                
                if idx == 0: #north
                    print(f"idx: {idx}")
                    x_pos -= x_mini_offset

                elif idx == 1: #east
                    print(f"idx: {idx}")
                    x_pos += x_mini_offset
                    y_pos -= y_mini_offset

                elif idx == 2: #south
                    print(f"idx: {idx}")
                    y_pos += y_mini_offset
                    x_pos += x_mini_offset

                elif idx == 3: #west
                    print(f"idx: {idx}")
                    x_pos -= x_mini_offset
                    y_pos += y_mini_offset

                elif idx == 4: #northeast
                    print(f"idx: {idx}")
                    y_pos-=y_mini_offset * 2
                    x_pos -= x_mini_offset

                elif idx == 5: #southeast
                    print(f"idx: {idx}")
                    x_pos += x_mini_offset * 2

                elif idx == 6: #southwest (FIX)
                    print(f"idx: {idx}")
                    y_pos += y_mini_offset * 1.5

                elif idx == 7: #northwest
                    print(f"idx: {idx}") 
                    x_pos -= x_mini_offset * 2



            x_pos = well_x
            y_pos = well_y

            # Move to next well
            x_pos -= x_offset
            well_count+=1

            if well_count >= total_wells: 
                break
        

        if well_count >= total_wells: 
            break

        # finished row, move to next row of wells
        y_pos -= y_offset
        x_pos = starting_x

    #once everything is done, home the xy stage
    print("Homing all axes...")
    stage.home_all_axes(timeout=5)
    zip_filename = f"{timestamp}_amorphous_crystalline_original.zip"
    zip_and_delete_image(parent_path, zip_filename)

    stage.disconnect()
    return ["success!"] if results else ["No samples processed"]

#UASERVER LOGIC
#create method node
def add_amscope(server): 
    id = server.register_namespace("Amscope") #creates the Amscope object node name
    root = server.get_objects_node()
    amscope = root.add_object(id, "Amscope") #create new object node called Amscope in root directory

    def move_to_KX2_node(parent, *args):
        result = move_to_KX2()   
        print(result)
        return [ua.Variant(result, ua.VariantType.String)]  


    def capture_brightfield_node(parent, input_args): 
        #load json file
        with open('well_positions.json', 'r') as file:
            positions = json.load(file)

        key = str(input_args.Value)
        print("Key is: ", key)

        if key == "24": 
            print("Selected 24 well plate")
            data = positions["24"]

        else:
                print(f"Unknown plate key: {key}")
                data = [
                    {
                        "starting-well": "A1",
                        "starting-position": [{"x": 60440, "y": 32500}],
                        "total-wells": key
                    }
                ]
                        
        result = capture_brightfield(data) #call the capture method with test data

        print("Capture result:", result)
        return [ua.Variant(str(result), ua.VariantType.String)]


    def capture_amorphous_and_crystalline(parent, input_args): 
        #load json file
        with open('well_positions.json', 'r') as file:
            positions = json.load(file)

        key = str(input_args.Value)
        print("Key is: ", key)

        if key == "24": 
            print("Selected 24 well plate")
            data = positions["24"]

        else:
                print(f"Unknown plate key: {key}")
                data = [
                    {
                        "starting-well": "A1",
                        "starting-position": [{"x": 60440, "y": 32500}],
                        "total-wells": key
                    }
                ]
        
        result = capture_amorphous_crystalline(data) #call the capture method with test data

        print("Capture result:", result)
        return [ua.Variant(str(result), ua.VariantType.String)]

    amscope.add_method(id, "capture_amorphous_crystalline", capture_amorphous_and_crystalline, [ua.VariantType.String], [ua.VariantType.String]) #This method takes in a string, and outputs a string
    amscope.add_method(id, "capture_brightfield", capture_brightfield_node, [ua.VariantType.String], [ua.VariantType.String])
    amscope.add_method(id, "move_to_kx2", move_to_KX2_node, [], [ua.VariantType.String])


def main(): 
    
    init_hardware()  
    #create the server object and set the endpoint where clients will connect
    server = Server()
    server.set_endpoint("opc.tcp://0.0.0.0:4840/")
    add_amscope(server)
    server.start()

    try: 
        while True:  #keep the server running until it is manually stopped
            time.sleep(1) 
    finally: 
        server.stop() #guarantees server stops when you interrupt it

if __name__=="__main__": 
    main()
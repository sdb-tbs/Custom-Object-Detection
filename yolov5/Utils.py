import os
from PIL import Image, ImageDraw
import glob
import numpy as np
from sklearn.cluster import KMeans
import cv2
def process_images(image_folder, label_folder, output_folder, new_size=(256, 256)):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each image in the image folder
    for image_file in os.listdir(image_folder):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            # Construct paths for the image and corresponding label file
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, os.path.splitext(image_file)[0] + '.txt')

            # Check if the corresponding label file exists
            if os.path.exists(label_path):
                # Open and resize the image
                with Image.open(image_path) as img:
                    img = img.resize((64,64))
                    draw = ImageDraw.Draw(img)
                    img_width, img_height = img.size

                    # Read label file and draw rectangles
                    with open(label_path, 'r') as file:
                        for line in file:
                            line = line.strip()  # Remove any leading/trailing whitespace
                            if line:  # Check if the line is not empty
                                # Parse the line
                                _, cx, cy, w, h = [float(val) for val in line.split()]

                                # Convert normalized coordinates to absolute pixel coordinates
                                x1 = int((cx - w / 2) * img_width)
                                y1 = int((cy - h / 2) * img_height)
                                x2 = int((cx + w / 2) * img_width)
                                y2 = int((cy + h / 2) * img_height)

                                # Draw the rectangle
                                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                    # Save the processed image
                    output_path = os.path.join(output_folder, image_file)
                    img.save(output_path)
#
# # Example usage
# process_images(r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Rifle.v1i.yolov5pytorch\valid\images',
#                r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Rifle.v1i.yolov5pytorch\valid\labels',
#                r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Rifle.v1i.yolov5pytorch\valid\New folder')
def grayscale_images_in_folder(folder_path):
    """
    Grayscale all images in the specified folder and save them with the same name.

    :param folder_path: Path to the folder containing images.
    """
    # List all image files in the folder
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct full image path
            image_path = os.path.join(folder_path, file)

            # Open the image
            with Image.open(image_path) as img:
                # Convert to grayscale
                grayscale_img = img.convert('L')

                # Save the grayscaled image with the same name in the same folder
                grayscale_img.save(os.path.join(folder_path, file))

    print("Grayscaling completed for all images in the folder.")

# Example usage:
# grayscale_images_in_folder(r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Rifle.v1i.yolov5pytorch\New folder (2)')
def delete_every_third_image(directory):
    # Get a list of image files in the directory
    images = glob.glob(os.path.join(directory, '*.txt'))  # Assuming images are in JPG format

    # Loop through the images and delete every third one
    for i, image_path in enumerate(images):
        if (i + 1) % 3 == 0:  # +1 because indexing starts at 0
            os.remove(image_path)
            print(f"Deleted: {image_path}")

# delete_every_third_image(r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Rifle.v1i.yolov5pytorch\train\labels')


def read_yolo_annotations(directory):
    """
    Reads YOLO formatted annotation files (.txt) from a specified directory
    and returns a list of bounding boxes.

    Parameters:
    directory (str): The directory containing YOLO annotation files.

    Returns:
    np.array: An array of bounding boxes [width, height].
    """
    boxes = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r') as file:
                for line in file:
                    parts = line.split()
                    if len(parts) == 5:
                        _, x_center, y_center, width, height = map(float, parts)
                        boxes.append([width, height])
    return np.array(boxes)

def calculate_anchors(directory, num_anchors=9):
    """
    Calculates the optimal anchors for YOLOv5 training using k-means clustering on bounding boxes.

    Parameters:
    directory (str): The directory containing YOLO annotation files.
    num_anchors (int): Number of anchors to calculate.

    Returns:
    np.array: Array of calculated anchors [width, height].
    """
    boxes = read_yolo_annotations(directory)
    kmeans = KMeans(n_clusters=num_anchors, random_state=0).fit(boxes)
    anchors = kmeans.cluster_centers_
    anchors = np.round(anchors, decimals=2)
    return anchors
# normalized_anchors = calculate_anchors(r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5'
#                                        r'\data\images\Rifle.v1i.yolov5pytorch\train\labels', num_anchors=3)
# input_size = 160
# absolute_anchors = (normalized_anchors * input_size).astype(int)
# print(absolute_anchors)
def extract_frames_from_npz(npz_file_path, output_directory,frame_height=80, frame_width=120):
    # Load the .npz file
    with np.load(npz_file_path) as data:
        # Check if there are keys in the npz file
        if data.files:
            first_key = data.files[0]
            print(f"Using key '{first_key}' from the npz file.")

            # Get the frames using the first key
            all_frames = data[first_key]

            # Loop through each flattened frame
            for i, flat_frame in enumerate(all_frames):
                # Reshape the flat frame to its original dimensions
                frame = flat_frame.reshape((frame_height, frame_width)).astype(np.uint8)

                # Define the output file path with zero-padded frame number
                frame_file_path = os.path.join(output_directory, f'frame14_{i}.png')

                # Save the frame as a .png file
                cv2.imwrite(frame_file_path, frame)
                print(f'Saved {frame_file_path}')
        else:
            print("No keys found in the npz file.")


#
#
# extract_frames_from_npz(r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\live_video_10fps00_80x120pixel_8bit_15_26_53.npz',
#                         r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\2')
#
#

def clean_images(text_folder, image_folder, image_extensions=['jpg', 'png', 'jpeg']):
    # Read all .txt files
    txt_files = {os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(text_folder, '*.txt'))}

    # Iterate over each image extension type
    for ext in image_extensions:
        # Read all image files with the current extension
        image_files = glob.glob(os.path.join(image_folder, f'*.{ext}'))

        for image_file in image_files:
            # Get the name of the image file without extension
            image_name = os.path.splitext(os.path.basename(image_file))[0]

            # If the image name is not in the txt_files set, delete the image
            if image_name not in txt_files:
                os.remove(image_file)
                print(f'Deleted {image_file}')

# Example usage
# text_folder_path = r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\2_bbx'
# image_folder_path = r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\2'
# clean_images(text_folder_path, image_folder_path)

def draw_bounding_boxes(text_folder, image_folder, output_folder):
    # Read all .txt files
    txt_files = glob.glob(os.path.join(text_folder, '*.txt'))

    for txt_file in txt_files:
        base_name = os.path.splitext(os.path.basename(txt_file))[0]

        # Corresponding image file path
        image_path = os.path.join(image_folder, base_name + '.png')  # Assuming images are in .jpg format

        # Check if the corresponding image file exists
        if os.path.exists(image_path):
            # Open the image
            with Image.open(image_path) as im:
                draw = ImageDraw.Draw(im)

                # Read the txt file and draw boxes
                print(txt_file)
                with open(txt_file, 'r') as file:
                    for line in file:
                        _, x_center, y_center, width, height = map(float, line.split())
                        # Convert YOLO format to bounding box coordinates
                        x1 = (x_center - width / 2) * im.width
                        y1 = (y_center - height / 2) * im.height
                        x2 = (x_center + width / 2) * im.width
                        y2 = (y_center + height / 2) * im.height
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

                # Save the image
                output_path = os.path.join(output_folder, base_name + '_bbox.png')
                im.save(output_path)


# Example usage
text_folder_path = r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\1_bbx\labels'
image_folder_path = r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\1'
output_folder_path = r'C:\aistorm\Cheetah_face_recognition\yolov5_88p\yolov5\data\images\Nerf_Gun_video\1_bbx\labels'
draw_bounding_boxes(text_folder_path, image_folder_path, output_folder_path)

#!venv/bin/python3.12
# coding: utf-8

# In[2]:


# Standard library imports
from pathlib import Path  # To handle and manipulate filesystem paths
import os  # For interacting with the operating system
import glob  # For finding all file paths matching a specified pattern

# Third-party imports
import numpy as np  # For numerical operations and handling arrays
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
from PIL import Image  # For opening, manipulating, and saving many different image file formats

# PyTorch imports
import torch  # Main PyTorch library for building and training neural networks
from torch.utils.data import Dataset, DataLoader  # For handling datasets and data loaders
import torch.nn.functional as F

# PyTorch-I/O extension
import torchio as tio  # For medical image processing in PyTorch

# pydicom imports
import pydicom  # For reading, modifying, and writing DICOM files
from pydicom.data import get_testdata_file  # For accessing test DICOM files
from pydicom.fileset import FileSet  # For working with DICOM FileSets

# Scikit-learn imports
from sklearn.model_selection import train_test_split  # For splitting datasets into training and testing sets

from collections import defaultdict


# In[3]:


os.chdir('/Users/eleanorbolton/OneDrive - University of Leeds/CCP_MRI_IMAGE_SUBSET/')
print(os.getcwd())


# #### Process the DICOM Image
# This function processes a DICOM image and returns the image as a NumPy array. It optionally resizes the image to reduce its size in memory.
# 

# In[78]:

import torchio as tio
import matplotlib.pyplot as plt

# Function to display images
def display_images(images, title):
    plt.figure(figsize=(12, 6))
    for i in range(images.shape[0]):
        plt.subplot(1, images.shape[0], i + 1)
        plt.imshow(images[i, 0, images.shape[2] // 2].cpu(), cmap="gray")
        plt.title(f"{title} Image {i+1}")
        plt.axis("off")
    plt.show()



def process_dicom_image(path: str, resize=True) -> np.ndarray:
    """ Given a path to a DICOM image, process and return the image. 
        Reduces the size in memory.
    """
    dicom_file = pydicom.dcmread(path)
    image = dicom_file.pixel_array
    image = image - np.min(image)
    image = image.astype(np.uint8)
    
    # resize the image to 256x256 using PIL
    if resize:
        image = Image.fromarray(image)
        image = image.resize((384, 512))
        image = np.array(image)
    
    return image


# #### Get Sequence Image
# This function returns a sorted list of images from a specified MRI sequence subfolder. It excludes images that are entirely black.
# 

# In[79]:


def get_sequence_images(path: str) -> list:
    images = []
    
    # Get a list of all DICOM files in the directory
    image_path_list = glob.glob(os.path.join(path, '*'))
    
    # Read the DICOM files and store them with their instance numbers
    dicom_files = []
    for image_path in image_path_list:
        try:
            dicom_file = pydicom.dcmread(image_path)
            instance_number = dicom_file.InstanceNumber
            dicom_files.append((instance_number, image_path))
        except Exception as e:
            print(f"Error reading {image_path}: {e}")
    
    # Sort the files by instance number
    dicom_files.sort(key=lambda x: x[0])
    
    # Read the pixel data in sorted order
    for _, image_path in dicom_files:
        try:
            dicom_file = pydicom.dcmread(image_path)
            image = dicom_file.pixel_array
            images.append(image)
        except Exception as e:
            print(f"Error reading pixel data from {image_path}: {e}")
    
    return images


# ### Defining the central slice
# The anatomical "middle" of the MR image will be different in each subject. we therefore need to decide the best way to define the central slice

# #### Get the best slice
# 
# This is based on the sum of the pixel tensor and finds the max sum

# In[80]:


def find_best_slice(dicom_files):
    """ Find the slice with the highest sum of pixel intensities. """
    max_sum = -1
    best_slice = None

    for dicom_file, image_path in dicom_files:
        try:
            image = dicom_file.pixel_array
            image_sum = np.sum(image)
            if image_sum > max_sum:
                max_sum = image_sum
                best_slice = (dicom_file, image_path)
        except Exception as e:
            print(f"Error reading {image_path}: {e}")

    return best_slice


# ### Removing Duplicate Images
# Some images are present for the same subjects at the same position but have been processed. This Function removes the least infomrative of the duplicate image based on the number of 0 pixels

# In[81]:


def remove_duplicates(dicom_files):
    """ Remove duplicate instance numbers, keeping only the slice with the highest sum of intensities. """
    instance_dict = defaultdict(list)

    for dicom_file, image_path in dicom_files:
        instance_number = dicom_file.InstanceNumber
        instance_dict[instance_number].append((dicom_file, image_path))

    # Keep only the slice with the highest sum of intensities for each instance number
    unique_dicom_files = []
    for instance_number, files in instance_dict.items():
        if len(files) > 1:
            best_slice = find_best_slice(files)
            unique_dicom_files.append(best_slice)
        else:
            unique_dicom_files.append(files[0])

    return unique_dicom_files


# ### Get best subject Images:
# Selects the best images and surrounding images (based on seq_len) according to the sum of the intensities 

# In[146]:


def get_best_patient_images(base_path):
    """ 
    Process all images in the 't1_vibe_we' subfolder of each subject.
    Sort images by Instance Number and return a sequence of a fixed length.

    Parameters:
        base_path (str): Base path containing all subject folders.

    Returns:
        np.array: Array of images for each subject that meet the criteria.
    """
    seq_len = 20
    all_images = []

    for root, dirs, files in os.walk(base_path):
        if 't1_vibe_we' in dirs:
            t1_vibe_we_path = os.path.join(root, 't1_vibe_we')
            
            # Get the images in the 't1_vibe_we' sequence
            dicom_files = []
            for image_path in glob.glob(os.path.join(t1_vibe_we_path, '*')):
                try:
                    dicom_file = pydicom.dcmread(image_path)
                    dicom_files.append((dicom_file, image_path))
                except Exception as e:
                    print(f"Error reading {image_path}: {e}")

            # Sort the files by Instance Number
            dicom_files.sort(key=lambda x: x[0].InstanceNumber)
            
            # Remove duplicates
            dicom_files = remove_duplicates(dicom_files)

            # Find the best slice
            best_slice = find_best_slice(dicom_files)
            if best_slice:
                best_dicom_file, best_image_path = best_slice
                best_instance_number = best_dicom_file.InstanceNumber
                print(f"Best instance number: {best_instance_number}")

                # Calculate the start and end indices for the selected sequence
                start_index = max(0, best_instance_number - (seq_len // 2))
                end_index = start_index + seq_len

                # Select the slices around the best slice
                selected_slices = dicom_files[start_index:end_index]

                images = []
                for dicom_file, image_path in selected_slices:
                    try:
                        image = process_dicom_image(image_path)
                        # Add channel dimension (1, x, y, z)
                        # image = np.expand_dims(image, axis=0)
                        images.append(image)
                    except Exception as e:
                        print(f"Error processing image {image_path}: {e}")

                # Determine the original image dimensions
                if images:
                    img_shape = images[0].shape

                if len(images) < seq_len:
                    # Pad with zero images of the same shape as the original images
                    diff = seq_len - len(images)
                    images.extend([np.zeros(img_shape, dtype=np.uint8) for _ in range(diff)])

                all_images.extend(images)
                print(np.array(all_images).shape)
    return np.array(all_images)



# #### Read CSV file and set up paths

# In[4]:


# Reading the CSV file
training_data_dir = "/Users/eleanorbolton/Library/CloudStorage/OneDrive-UniversityofLeeds/t1_vibe_we_hand_subset/" 
csv_path = os.path.join(training_data_dir, 'training_labels_subset.csv')
labels_df = pd.read_csv(csv_path)

# Split the data into training and validation sets
train_df, valid_df = train_test_split(labels_df, test_size=0.2, random_state=42, stratify=labels_df['progression'])

# Save the splits for reference
train_df.to_csv(os.path.join(training_data_dir, 'train_split.csv'), index=False)
valid_df.to_csv(os.path.join(training_data_dir, 'valid_split.csv'), index=False)


# ### Creating a custom dataset

# In[180]:


class HandScanDataset(Dataset):
    def __init__(self, labels_df, data_dir, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the patient IDs and labels
            data_dir (str): Path to the data folder
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.data_dir = data_dir
        self.transform = transform

        # Create a list of patient IDs and their corresponding labels
        self.patient_ids = self.labels_df['patient ID'].astype(str).str.zfill(5).tolist()
        self.labels = self.labels_df['progression'].apply(lambda x: 1 if x == 'y' else 0).tolist()

        # Create a dictionary of the labels
        self.dict_labels = dict(zip(self.patient_ids, self.labels))
        print(self.dict_labels)


    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.patient_ids[idx]
        label = self.labels[idx]

        # Process the images for this patient
        patient_dir = os.path.join(self.data_dir, patient_id)
        images = get_best_patient_images(patient_dir)  # Ensure this function only returns images for the given patient
        
        # If no images were returned, handle this case (optional)
        if len(images) == 0:
            raise ValueError(f"No images found for patient {patient_id}")


        images_tensor = torch.tensor(images, dtype=torch.float32)
        images_tensor_channel = torch.unsqueeze(images_tensor, 0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            images_tensor_channel = self.transform(images_tensor_channel)

        return images_tensor_channel, label_tensor



# ### Setting up transformation using torch.io

# #### Custom thresholding 
# Separates the the foregrounds (objects of interest – hand) from the background 
# Pixels with intensity values above this threshold are considered part of the foreground, while those below are treated as background.
# 

# In[5]:


class CustomThresholding(tio.Transform):
    def __init__(self, threshold_percentage=0.1):
        super().__init__()
        self.threshold_percentage = threshold_percentage

    def apply_transform(self, subject):
        for key, image in subject.get_images_dict(intensity_only=True).items():
            max_intensity = torch.max(image.data)
            threshold_value = self.threshold_percentage * max_intensity
            binary_mask = (image.data > threshold_value).float()
            subject.add_image(tio.LabelMap(tensor=binary_mask), f'{key}_mask')
        return subject


# #### Morphological Operations
# improves the mask of the hand to identfiy the ROI

# In[6]:


class MorphologicalOperations(tio.Transform):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.kernel = torch.ones((1, 1, kernel_size, kernel_size, kernel_size), dtype=torch.float32)

    def apply_transform(self, subject):
        for key, image in subject.get_images_dict(intensity_only=False).items():
            if 'mask' in key:
                # Add batch and channel dimensions
                mask_tensor = image.data.unsqueeze(0).unsqueeze(0)
                
                # Morphological opening (erosion followed by dilation)
                eroded = F.conv3d(mask_tensor, self.kernel, padding=1) > (self.kernel_size ** 3 - 1)
                dilated = F.conv3d(eroded.float(), self.kernel, padding=1) > 0
                
                # Morphological closing (dilation followed by erosion)
                dilated_closed = F.conv3d(dilated.float(), self.kernel, padding=1) > 0
                eroded_closed = F.conv3d(dilated_closed.float(), self.kernel, padding=1) > (self.kernel_size ** 3 - 1)
                
                # Remove batch and channel dimensions
                final_mask = eroded_closed.squeeze(0).squeeze(0)
                subject.add_image(tio.LabelMap(tensor=final_mask), f'{key}_processed')
        return subject


# In[12]:


transform = tio.Compose([
    tio.ToCanonical(),                # Reorient images to a standard orientation
    tio.CropOrPad((20, 512, 512)),     # Crop or pad to 2 slices and 384x512 pixels
    CustomThresholding(threshold_percentage=0.1),  # Apply custom thresholding
    MorphologicalOperations(kernel_size=3),        # Apply morphological operations
    tio.RandomElasticDeformation(
        num_control_points=8,
        max_displacement=(1, 1, 1),
        locked_borders=True
    ),
    tio.RandomFlip(axes=(2,)),        # Randomly flip along the vertical axis only
    #tio.RandomNoise(std=(0, 0.02)),   # Add subtle Gaussian noise 
    tio.RandomBlur(std=(0.5, 1.0))    # Apply subtle blur
])



# In[13]:


validation_transform = tio.Compose([
    tio.ToCanonical(),                # Reorient images to a standard orientation
    tio.CropOrPad((20, 512, 512))   # Crop or pad images to the desired shape
])


# In[14]:


class HandScanDataset2(Dataset):
    def __init__(self, labels_df, data_dir, transform=None):
        """
        Args:
            labels_df (DataFrame): DataFrame containing the patient IDs and labels
            data_dir (str): Path to the data folder
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels_df = labels_df
        self.data_dir = data_dir
        self.transform = transform

        # Create a list of patient IDs and their corresponding labels
        self.patient_ids = self.labels_df['patient ID'].astype(str).str.zfill(5).tolist()
        self.labels = self.labels_df['progression'].apply(lambda x: 1 if x == 'y' else 0).tolist()

        # Create a dictionary of the labels
        self.dict_labels = dict(zip(self.patient_ids, self.labels))
        

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patient_id = self.patient_ids[idx]
        label = self.labels[idx]

        # Process the images for this patient
        patient_dir = os.path.join(self.data_dir, patient_id)
        images = self.get_best_patient_images(patient_dir)  # Call the internal method
        
        # If no images were returned, handle this case (optional)
        if len(images) == 0:
            raise ValueError(f"No images found for patient {patient_id}")

            # Display the raw images before any transformation
        print(f"Displaying raw images for patient: {patient_id}")
        for i, img in enumerate(images):
            plt.figure(figsize=(6, 6))
            plt.imshow(img, cmap='gray')
            plt.title(f"Raw Image {i+1}")
            plt.axis('off')
            plt.show()


        images_tensor = torch.tensor(images, dtype=torch.float32)
        images_tensor_channel = torch.unsqueeze(images_tensor, 0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.transform:
            images_tensor_channel = self.transform(images_tensor_channel)

        return images_tensor_channel, label_tensor

    def get_best_patient_images(self, base_path):
        """ 
        Process all images in the 't1_vibe_we' subfolder of each subject.
        Sort images by Instance Number and return a sequence of a fixed length.
        """
        seq_len = 2
        all_images = []
        img_shape = (512, 384)  # Set a default image shape

        for root, dirs, files in os.walk(base_path):
            if 't1_vibe_we' in dirs:
                t1_vibe_we_path = os.path.join(root, 't1_vibe_we')
                
                # Get the images in the 't1_vibe_we' sequence
                dicom_files = []
                for image_path in glob.glob(os.path.join(t1_vibe_we_path, '*')):
                    try:
                        dicom_file = pydicom.dcmread(image_path)
                        dicom_files.append((dicom_file, image_path))
                    except Exception as e:
                        print(f"Error reading {image_path}: {e}")

                # Sort the files by Instance Number
                dicom_files.sort(key=lambda x: x[0].InstanceNumber)
                
                # Remove duplicates
                dicom_files = self.remove_duplicates(dicom_files)

                # Find the best slice
                best_slice = self.find_best_slice(dicom_files)
                if best_slice:
                    best_dicom_file, best_image_path = best_slice
                    best_instance_number = best_dicom_file.InstanceNumber

                    # Calculate the start and end indices for the selected sequence
                    start_index = max(0, best_instance_number - (seq_len // 2))
                    end_index = start_index + seq_len

                    # Select the slices around the best slice
                    selected_slices = dicom_files[start_index:end_index]

                    images = []
                    for dicom_file, image_path in selected_slices:
                        try:
                            image = self.process_dicom_image(image_path)
                            images.append(image)
                        except Exception as e:
                            print(f"Error processing image {image_path}: {e}")

                    # Determine the original image dimensions
                    if images:
                        img_shape = images[0].shape  # Set img_shape based on the first image

                    if len(images) < seq_len:
                        # Pad with zero images of the same shape as the original images
                        diff = seq_len - len(images)
                        images.extend([np.zeros(img_shape, dtype=np.uint8) for _ in range(diff)])

                    all_images.extend(images)
        return np.array(all_images)


    def remove_duplicates(self, dicom_files):
        """ Remove duplicate instance numbers, keeping only the slice with the highest sum of intensities. """
        instance_dict = defaultdict(list)

        for dicom_file, image_path in dicom_files:
            instance_number = dicom_file.InstanceNumber
            instance_dict[instance_number].append((dicom_file, image_path))

        # Keep only the slice with the highest sum of intensities for each instance number
        unique_dicom_files = []
        for instance_number, files in instance_dict.items():
            if len(files) > 1:
                best_slice = self.find_best_slice(files)
                unique_dicom_files.append(best_slice)
            else:
                unique_dicom_files.append(files[0])

        return unique_dicom_files

    def find_best_slice(self, dicom_files):
        """ Find the slice with the highest sum of pixel intensities. """
        max_sum = -1
        best_slice = None

        for dicom_file, image_path in dicom_files:
            try:
                image = dicom_file.pixel_array
                image_sum = np.sum(image)
                if image_sum > max_sum:
                    max_sum = image_sum
                    best_slice = (dicom_file, image_path)
            except Exception as e:
                print(f"Error reading {image_path}: {e}")

        return best_slice

    def process_dicom_image(self, path: str, resize=True) -> np.ndarray:
            dicom_file = pydicom.dcmread(path)
            image = dicom_file.pixel_array.astype(np.float32)
            
            # Normalize the image: Zero mean and unit variance
            #mean = np.mean(image)
            #std = np.std(image)
            #image = (image - mean) / (std + 1e-7)  # Add a small epsilon to prevent division by zero

            # Apply 95% clipping
            #lower_bound = np.percentile(image, 2.5)
            #upper_bound = np.percentile(image, 97.5)
            #image = np.clip(image, lower_bound, upper_bound)

            # Normalize again after clipping
            #mean = np.mean(image)
            #std = np.std(image)
            #image = (image - mean) / (std + 1e-7)

            # Convert back to uint8 for further processing
            image = (image * 255).astype(np.uint8)

            if resize:
                image = Image.fromarray(image)
                image = image.resize((512, 384))  # Resize the image to 512x384
                image = np.array(image)

            return image


    def get_sequence_images(self, path: str) -> list:
            images = []
            
            # Get a list of all DICOM files in the directory
            image_path_list = glob.glob(os.path.join(path, '*'))
            
            # Read the DICOM files and store them with their instance numbers
            dicom_files = []
            for image_path in image_path_list:
                try:
                    dicom_file = pydicom.dcmread(image_path)
                    instance_number = dicom_file.InstanceNumber
                    dicom_files.append((instance_number, image_path))
                except Exception as e:
                    print(f"Error reading {image_path}: {e}")
            
            # Sort the files by instance number
            dicom_files.sort(key=lambda x: x[0])
            
            # Read the pixel data in sorted order
            for _, image_path in dicom_files:
                try:
                    dicom_file = pydicom.dcmread(image_path)
                    image = dicom_file.pixel_array
                    images.append(image)
                except Exception as e:
                    print(f"Error reading pixel data from {image_path}: {e}")
            
            return images


# #### Dataloader

# In[15]:


# Creating datasets
train_dataset = HandScanDataset2(labels_df=train_df, data_dir=training_data_dir, transform=transform)
valid_dataset = HandScanDataset2(labels_df=valid_df, data_dir=training_data_dir, transform=validation_transform)

# Creating data loaders
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# In[16]:


# Check a few samples directly from the dataset
for i in range(len(train_dataset)):
    images, labels = train_dataset[i]
    print(f"Sample {i}: Image shape: {images.shape}, Label: {labels}")
    if i == 2:  # Check only the first 3 samples
        break


# In[17]:


# Iterate through the train_loader and print the shape of the batches
for images, labels in train_loader:
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    break  # Remove this break to see all batches, or keep to see just the first batch


# In[ ]:






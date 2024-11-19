import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from PIL import Image, UnidentifiedImageError
import os

root = tk.Tk()
root.title('Image Files Lister')
root.geometry('700x450')

# Text widget to display data and results
text = tk.Text(root, height=30, width=90)
text.grid(column=0, row=0)

def open_folder():
    # Open a folder dialog
    folder_path = fd.askdirectory()
    if folder_path:
        image_files = []
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            # Ensure that it's a file
            if os.path.isfile(filepath):
                try:
                    # Attempt to open the file as an image
                    with Image.open(filepath) as img:
                        image_files.append(filename)
                except (UnidentifiedImageError, IOError):
                    # If PIL cannot identify the file, it's not an image; skip it
                    continue
        # Display the list of image file names
        text.delete('1.0', tk.END)  # Clear previous content
        if image_files:
            text.insert(tk.END, "Image files in the folder:\n")
            for image_file in image_files:
                text.insert(tk.END, image_file + '\n')
        else:
            text.insert(tk.END, "No image files found in the selected folder.\n")

# Button to open folder and trigger the listing
open_button = ttk.Button(
    root,
    text='Open a Folder',
    command=open_folder
)

open_button.grid(column=0, row=1, padx=10, pady=10)

root.mainloop()
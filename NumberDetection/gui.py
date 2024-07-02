import tkinter
from tkinter import filedialog, messagebox
import customtkinter as ctk
from pytube import YouTube

def uploadPhoto():
    file_path = filedialog.askopenfilename(
        title="Select a Photo",
        filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("JPEG Files", "*.jpeg"), ("GIF Files", "*.gif"), ("BMP Files", "*.bmp")]
    )
    if file_path:
        messagebox.showinfo("Information", f"Selected file: {file_path}")
    else:
        messagebox.showwarning("Warning", "No file selected")

ctk.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("720x480")
app.title("Hieroglyphic Scanner")

title = ctk.CTkLabel(app, text="Welcome to the Hieroglyphic Scanner", text_color="white")
title.pack(padx=10, pady=10)

pictureButton = ctk.CTkButton(app, text="Upload a photo", command=uploadPhoto)
pictureButton.pack(padx=10, pady=10)

app.mainloop()

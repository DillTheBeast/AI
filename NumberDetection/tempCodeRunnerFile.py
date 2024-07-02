import tkinter as tk

def basic_tkinter_test():
    root = tk.Tk()
    root.geometry("720x480")
    root.title("Basic Tkinter Test")

    label = tk.Label(root, text="This is a basic Tkinter label")
    label.pack(padx=10, pady=10)

    root.mainloop()

basic_tkinter_test()

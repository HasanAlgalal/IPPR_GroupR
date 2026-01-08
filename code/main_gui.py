from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog

from pipeline.plate_detect import detect_plate_info


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("IPPR LPR + State + Vehicle (KL Motorcycle / Penang Van + Car/Truck)")
        self.root.geometry("980x560")

        top = tk.Frame(root)
        top.pack(fill="x", pady=10)

        btn = tk.Button(top, text="Select Image", width=25, height=2, command=self.select_image)
        btn.pack()

        self.text = tk.Text(root, font=("Consolas", 10))
        self.text.pack(fill="both", expand=True, padx=10, pady=10)

    def log(self, s: str):
        self.text.insert("end", s + "\n")
        self.text.see("end")

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.JPG *.JPEG *.PNG")],
        )
        if not path:
            return

        info = detect_plate_info(path)

        self.log(f"Image: {os.path.basename(path)}")
        if not info.plate_text:
            self.log("[FAIL] Plate not detected")
            self.log("-" * 90)
            return

        self.log(f"Plate: {info.plate_text}")
        self.log(f"Prefix: {info.prefix}")
        self.log(f"State: {info.state}")
        self.log(f"Vehicle Type: {info.vehicle_type}")
        self.log(f"Mode: {info.mode}  quality={info.quality}")
        self.log(f"Debug: {info.debug}")
        self.log("-" * 90)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()

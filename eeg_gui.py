import customtkinter as ctk
from tkinter import filedialog
from tkinter import messagebox
import os, preprocessing, hyperscan

class EEGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("500x600")
        self.title("EEG Analysis")

        #Use file dialog Add label that shows the which file selected later
        self.label = ctk.CTkLabel(self, text="No Folder Selected", font=("Arial", 16)) #Initial state (no folder selected)
        self.label.pack()

        #Button for picking directories
        self.button_input = ctk.CTkButton(self, text="Select Input Directory", command=lambda: self.directory("input"))
        self.button_input.pack(pady=20)

        #For output dir
        self.button_input = ctk.CTkButton(self, text="Select Output Directory", command=lambda: self.directory("output"))
        self.button_input.pack(pady=20)

        """This part commented out"""
        # self.button_output = ctk.CTkButton(self, text="Select Output Directory", command=lambda: self.directory("output"))
        # self.button_output.pack(pady=20)

        #Button for running the program (main.py)
        self.button_run = ctk.CTkButton(self, text="Run Script",state="disabled" ,command=self.run_script)
        self.button_run.pack(pady=20)

        self.input_path = ""
        self.output_path = ""

        ##For inputing following params numPoints, numAscans, numBscans
        self.param_select = ctk.CTkFrame(self)
        self.param_select.pack(pady=20)

        ctk.CTkLabel(self.param_select, text="sampling rate").grid(row=0, column=0, padx=10, pady=5)
        ctk.CTkLabel(self.param_select, text="downsample freq").grid(row=1, column=0, padx=10, pady=5)


        self.sample_freq = ctk.CTkEntry(self.param_select)
        self.downsamp = ctk.CTkEntry(self.param_select)

        self.sample_freq.grid(row=0,column=1,padx=10)
        self.downsamp.grid(row=1,column=1,padx=10)
    
        #By default use these values
        self.sample_freq.insert(0,"500")
        self.downsamp.insert(0,"250")
        
        self.montage_type = ctk.StringVar(value="standard_1020")

        self.reg_mode = ctk.IntVar(value=0)

        frame = ctk.CTkFrame(self)
        frame.pack(padx=20, pady=20)

        
        frame.grid_columnconfigure(0, weight=1)  # Left spacer
        frame.grid_columnconfigure(3, weight=1)  # Right spacer

        #Montage 
        ctk.CTkLabel(frame, text="Montage").grid(row=0, column=1, sticky="w", padx=(20, 15), pady=(10, 15))
        ctk.CTkRadioButton(frame, text="standard_1020", variable=self.montage_type, value="standard_1020").grid(row=1, column=1, sticky="w", padx=(20, 15), pady=8)
        # ctk.CTkRadioButton(frame, text="standard_1010", variable=self.montage_type, value="standard_1010").grid(row=2, column=1, sticky="w", padx=(20, 15), pady=8)
        ctk.CTkRadioButton(frame, text="standard_1005", variable=self.montage_type, value="standard_1005").grid(row=2, column=1, sticky="w", padx=(20, 15), pady=8)

        #Mode select
        ctk.CTkLabel(frame, text="Mode Selection").grid(row=0, column=2, sticky="w", padx=(15, 20), pady=(10, 15))
        ctk.CTkRadioButton(frame, text="EEG Preprocessing", variable=self.reg_mode, value=0).grid(row=1, column=2, sticky="w", padx=(15, 20), pady=8)
        ctk.CTkRadioButton(frame, text="Hyperscanning", variable=self.reg_mode, value=1).grid(row=2, column=2, sticky="w", padx=(15, 20), pady=8)

        # #Which frequency band? (This is maybe!!)
        # ctk.CTkLabel(frame, text="Frequency bands").grid(row=0, column=3, sticky="w", padx=(15, 20), pady=(10, 15))
        


    def directory(self, folder_type):
        path = filedialog.askdirectory()
        if(path):
            if(folder_type == "input"):
                self.input_path = path
                self.label.configure(text=f"Input Directory:\n{path}")
            else:
                self.output_path = path
                self.label.configure(text=f"Output Directory:\n{path}")
            if self.input_path:
                self.button_run.configure(state="normal") #enable to run button

    # def directory(self, folder_type):
    #     path = filedialog.askdirectory()
    #     if(path):
    #         if(folder_type == "input"):
    #             self.input_path = path
    #             self.label.configure(text=f"Input Directory:\n{path}")
    #             #Define output path as well
    #         if(folder_type == "output"):
    #             self.output_path = path
    #             self.label.configure(text=f"Input Directory:\n{path}")                
    #         if self.input_path:
    #             self.button_run.configure(state="normal") #enable to run button

    def run_script(self):
        try:
            samp_get = int(self.sample_freq.get())
            downsamp_get = int(self.downsamp.get())
        except ValueError:
            messagebox.showwarning("Invalid Input", "Parameters must be integers")
            return
        montage_type = self.montage_type.get()
        mode = self.reg_mode.get()
        print("selected: ",montage_type)
        print("selected: mode", mode)
        #Pick the correct mode script
        if mode == 0:
            preprocessing.main(input_dir=self.input_path,
                                  output_dir=self.output_path,
                                  sampling_freq=samp_get,
                                  downsample_freq=downsamp_get,
                                  montage=montage_type
                                )
        if mode == 1:
            hyperscan.main(

            )
            pass

if __name__ == "__main__":
    app = EEGApp()
    app.mainloop()
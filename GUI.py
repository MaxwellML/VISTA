from seegull import run_program
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import rasterio
from rasterio.plot import show
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from pyproj import Transformer

tif_path = None #file to be loaded from.

def start_gui(run_program): #entry point for the program.

    def set_coordinate_entries(lon, lat):
        lon_entry.delete(0, tk.END) #delete text from longitude box.
        lon_entry.insert(0, f"{lon:.6f}") #insert given longitude with 6 decimal digits of precision.
    
        lat_entry.delete(0, tk.END) #delete text from latitude box.
        lat_entry.insert(0, f"{lat:.6f}") #insert given latitude with 6 decimal digits of precision.

    class RightSideBar(ttk.Frame): #right side bar showing the DEM preview and overlay result, ttk.Frame specifies it as a widget container.
        def __init__(self, parent):
            super().__init__(parent, padding=8) #initialise the object as a ttk.Frame and add padding.
    
            self.dem = None #store DEM data.
            self.dem_transform = None #store raster transformer.
            self.dem_crs = None #store raster CRS.
            self.dem_path = None #store file path to DEM.
            self.overlay = None #store result from LoS calculation to be overlayed.
            self.observer_xy = None #store observer cooridnates.
    
            self.point_selected_callback = None #reference to helper function for auto filling boxes on click.
            self.clicked_points = [] #store the coordinates the user has clicked on.
            self.tip = None
    
            self.rowconfigure(1, weight=1) #only allow preview to grow if needed.
            self.columnconfigure(0, weight=1) #allow preview section to be streched sideways.
    
            self.title_label = ttk.Label(self, text="DEM PREVIEW", font=("Segoe UI", 12, "bold")) #add Title.
            self.title_label.grid(row=0, column=0, sticky="w", pady=(0, 6)) #position Title.
    
            self.fig = Figure(figsize=(7, 6), dpi=100) #create a Matplotlib Figure object.
            self.ax = self.fig.add_subplot(111) #add axes.
            self.ax.set_title("No DEM loaded") #if no DEM loaded, inform user.
            self.ax.set_xticks([])
            self.ax.set_yticks([]) #do not show ticks around empty canvas message.
    
            self.canvas = FigureCanvasTkAgg(self.fig, master=self) #create a Tkinter compatible canvas wrapper for Matplotlib.
            self.canvas_widget = self.canvas.get_tk_widget() #render it with equivalent widget corresponding to a Matplotlib figure.
            self.canvas_widget.grid(row=1, column=0, sticky="nsew") #force DEM canvas to fill entire grid cell.
    
            self.toolbar_frame = ttk.Frame(self) #add container for Matplotlib toolbar.
            self.toolbar_frame.grid(row=2, column=0, sticky="ew") #position Matplotlib toolbar.
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False) #create the Matplotlib toolbar to add interactivity to the canvas.
            self.toolbar.update() #standard practice: refresh toolbar before displaying.
            self.toolbar.pack(side="left") #move toolbar to the left.
            
            self.show_overlay = tk.BooleanVar(value=True) #create a Boolean state for whether overlay should be visible, default to show.
            self.overlay_im = None #default to no overlay image.
            self.overlay_cbar = None #default to no colour bar.

            self.view_xlim = None
            self.view_ylim = None #store information about current zoom so it isn't lost on refresh.
    
            menubar = tk.Menu(parent) #create menu for the window in the top bar...
            parent.config(menu=menubar) #...and attach it.
    
            file_menu = tk.Menu(menubar, tearoff=0) #create file menu inside the top bar...
            menubar.add_cascade(label="File", menu=file_menu) #...and add it.
    
            view_menu = tk.Menu(menubar, tearoff=0) #create view menu inside the top bar...
            menubar.add_cascade(label="View", menu=view_menu) #...and add it.
    
            view_menu.add_checkbutton(
                label="Show overlay",
                variable=self.show_overlay,
                command=self.toggle_overlay
            ) #create a button to toggle overlay.
    
            self.canvas.mpl_connect("button_press_event", self.on_click) #create a click handler to detect what coordinates a user may click on when drawing a polygon.
    
        def on_click(self, event): #when the user clicks mouse.
            if self.dem is None:
                return
            if event.inaxes != self.ax: 
                return #if click happens outside of canvas, do nothing.
            if event.xdata is None or event.ydata is None:
                return #if the graph coordinates cannot be worked out, do nothing.
            if self.toolbar.mode != "":
                return
    
            x = event.xdata #obtain x coordinate user clicked on.
            y = event.ydata #obtain y coordinate user clicked on.
            self.clicked_points.append((x, y)) #return coordinates user clicked on.
            self.observer_xy = (x, y)
    
            if self.dem_crs is not None and self.point_selected_callback is not None: #if a coordinate system can be found, and the helper function is ready.
                transformer = Transformer.from_crs(self.dem_crs, "EPSG:4326", always_xy=True) #create DEM transformer.
                lon, lat = transformer.transform(x, y) #convert clicked position into longitude and latitude.
                self.point_selected_callback(lon, lat) #send coordinates to helper function.
            if self.ax.has_data():
                self.view_xlim = self.ax.get_xlim()
                self.view_ylim = self.ax.get_ylim()

            self._redraw()
    
        def load_dem(self, dem_path):
    
            with rasterio.open(dem_path) as src:
                self.dem = src.read(1, masked=True) #read in the elevation grid.
                self.dem_transform = src.transform #read in transformer.
                self.dem_crs = src.crs #read in CRS.
                self.dem_path = dem_path #read in file path.
    
            self.overlay = None
            self.observer_xy = None
            self._redraw()
    
        def set_overlay(self, overlay_mask, observer_xy=None):
            if self.dem is None:
                raise ValueError("Load a DEM before setting an overlay.") #check a DEM file is present.
    
            if overlay_mask.shape != self.dem.shape:
                raise ValueError("Overlay mask must have the same shape as the DEM.")
    
            self.overlay = overlay_mask #store DEM LoS render.
            self.observer_xy = observer_xy #store observer coordinates.
            self._redraw() #render.
    
        def toggle_overlay(self): #toggle overlay on/off.
            show = self.show_overlay.get() #retrieve current toggle state.
    
            if self.overlay_im is not None: #if overlay exists.
                self.overlay_im.set_visible(show) #toggle on/off.
    
            if self.overlay_cbar is not None: #if colourbar exists.
                self.overlay_cbar.ax.set_visible(show) #toggle on/off.
    
            self.canvas.draw_idle() #update display.
    
        def clear_overlay(self):
            self.overlay = None #remove any previous overlay.
            self.observer_xy = None #remove any previous observer point.
            self._redraw() #render preview area again.
    
        def draw_external_plot(self, draw_function, *args, **kwargs):
            self.ax.clear()
            draw_function(*args, ax=self.ax, **kwargs)
            self.canvas.draw_idle()
    
        def _redraw(self):
            self.ax.clear() #clean axes to prevent buildup.
    
            if self.dem is None: #if no DEM given, show blank DEM message.
                self.ax.set_title("No DEM loaded")
                self.ax.set_xticks([])
                self.ax.set_yticks([])
                self.canvas.draw_idle()
                return
    
            show(
                self.dem,
                transform=self.dem_transform,
                ax=self.ax,
                cmap="terrain"
            )  #render DEM base image.
    
    
            if self.observer_xy is not None:
                x, y = self.observer_xy
                self.ax.scatter([x], [y], marker="x", s=100, linewidths=2) #
        
            self.ax.set_title("DEM Preview")

            if self.view_xlim is not None and self.view_ylim is not None:
                self.ax.set_xlim(self.view_xlim)
                self.ax.set_ylim(self.view_ylim) #redraw zoom based on saved information. 

            
            self.canvas.draw_idle()
    
        def hide_tip(self, event=None):
            if self.tip is not None:
                self.tip.destroy() #remove window.
                self.tip = None #...and the reference to it.
            
    class LeftSideBar: #for each helper button
            

  
            def __init__(self, widget, text):
                self.widget = widget #widget the popup belongs to.
                self.text = text #text that should appear in popup.
                self.tip = None #start with no text showing.
                
                self.widget.bind("<Enter>", self.show_tip) #show text when mouse enters widget.
                self.widget.bind("<Leave>", self.hide_tip) #hide text when mouse leaves widget.
        
            def show_tip(self, event=None): #when mouse enters help widget.
                x = self.widget.winfo_rootx() + 20
                y = self.widget.winfo_rooty() + 20 #find coordinates of "?" icon, then move 20 right and up.

                self.tip = tw = tk.Toplevel(self.widget) #create a top level window for the popup.
                tw.wm_overrideredirect(True) #specifies this is a blank window.
                tw.wm_geometry(f"+{x}+{y}") #shift window to avoid overlapping with "?".

                label = tk.Label(
                    tw,
                    text=self.text,
                    justify="left",
                    background="#ffffe0",
                    relief="solid",
                    borderwidth=1,
                    padx=6,
                    pady=4
                ) #customise appearance of label inside window.
                label.pack() #place label inside tooltip window.

            def hide_tip(self, event=None):
                    if self.tip is not None:
                        self.tip.destroy() #remove window.
                        self.tip = None #...and the reference to it.

    def validate_inputs():
            max_observer_height = 10000
            lon_text = lon_entry.get().strip() #retrieve user input for longitude.
            lat_text = lat_entry.get().strip() #retrieve user input for latitude.
            observer_height_text = height_entry.get().strip() #retrieve user input for observer height.

            if not lon_text or not lat_text or not observer_height_text:
                raise ValueError("Please fill in all three fields.") #if any field is left blank, raise an error.

            try:
                lon = float(lon_text) 
                lat = float(lat_text)
                observer_height = float(observer_height_text)
            except ValueError:
                raise ValueError("Longitude, latitude, and observer height must all be numbers.")
            #attempt to convert inputs to floats, if they cannot be converted, raise an error.

            if not (-180 <= lon <= 180):
                raise ValueError("Longitude must be between -180 and 180.") #valid range for ESPG:4326 latitude is -180 to 180.

            if not (-90 <= lat <= 90):
                raise ValueError("Latitude must be between -90 and 90.") #valid range for ESPG:4326 latitude is -90 to 90.

            if observer_height <= 0:
                raise ValueError("Observer height must be greater than 0 metres.") #if user inputs negative observer height, raise an error.

            if observer_height > max_observer_height:
                raise ValueError(f"Observer height must not exceed {max_observer_height} metres.") #if user inputs an observer height above the maximum, raise an error.

            return lon, lat, observer_height #return inputs once validated so run_program can be run.
        
    def show_error(message):
            error_label.config(text=message) #update the error label with the error text.

    def file_hanlder():
        
            selected_file_var = tk.StringVar(value=tif_path if tif_path else "No file selected") #default display when no file selected.
            def validate_file(file_path):
                ext = Path(file_path).suffix.lower() #obtain ending of file name.

                if ext not in [".tif", ".tiff"]: #if ending isn't ".tif" or ".tiff"...
                    raise ValueError(f"Unsupported file type: {ext}") #...raise an error.
                
            def load_file():
                global tif_path
                file_path = filedialog.askopenfilename(
                    parent=root, #pop up must be shown in the app.
                    title="Open file for SeeGULL", #text at the top of window.
                    initialdir=".", #begin browsing in the directory of the program.
                    filetypes=[
                        ("GeoTIFF files", "*.tif *.tiff"),
                    ] #files that are shown as acceptable, in this case TIFF.
                ) #use file explorer to allow the user to select a file.

                if not file_path:
                    return  #if user aborts, stop function.

                try:
                    validate_file(file_path) #validate file function.
                    tif_path = file_path #store chosen file path to be passed into run_program later.
                    selected_file_var.set(file_path)
                    right_sidebar.load_dem(tif_path) #load initial DEM preview.
                    error_label.config(text="")
                except Exception as e:
                    messagebox.showerror("Load error", f"Could not load file:\n{e}") #if there is an error with loading the file, alert the user.

            top_bar = ttk.Frame(root, padding=8) #create a container for the file button and text.
            top_bar.grid(row=0, column=0, columnspan=2, sticky="ew") #stretch horizontally.

            file_button = ttk.Button(top_bar, text="File", command=load_file) #create button that runs "load_file" when clicked.
            file_button.pack(side="left") #move to the leftmost side.

            file_label = ttk.Label(top_bar, textvariable=selected_file_var) #label for currently selected file, or default text if no file loaded.
            file_label.pack(side="left", padx=10) #move to the leftmost side, next to the button.

        
    def submit():
            global tif_path
            error_label.config(text="") # clear any previous error message.
            try:
                if not tif_path:
                    raise ValueError("Please select a GeoTIFF file first.")

                lon, lat, observer_height = validate_inputs() #validate the user's inputs.

                right_sidebar.load_dem(tif_path) #load DEM into preview.
                right_sidebar.fig.clear()
                right_sidebar.ax = right_sidebar.fig.add_subplot(111)

                result = run_program(
                    lon,
                    lat,
                    observer_height,
                    tif_path,
                    ax=right_sidebar.ax,
                    show_reference=False
                ) #run the main program with the three values on the embedded axes.

                right_sidebar.observer_xy = result["observer_xy"] #retrieve information about zoom state.
                right_sidebar.overlay = result["overlay"] #store information on current overlay, so future redraws may use it.
                right_sidebar.view_xlim = right_sidebar.ax.get_xlim() #store current x zoom limit on graph.
                right_sidebar.view_ylim = right_sidebar.ax.get_ylim() #store current y zoom limit on graph.

                right_sidebar.overlay_im = right_sidebar.ax.images[-1] #update pointer to overlay.
                right_sidebar.overlay_cbar = getattr(right_sidebar.fig, "_seegull_cbar", None) #update pointer to colourbar.
                

                right_sidebar.toggle_overlay() #update preview with current toggle state.
        
                right_sidebar.canvas.draw_idle() #refresh the embedded preview.

        

            except ValueError as e:
                show_error(str(e)) #handle invalid number input.
            except Exception as e:
                show_error(str(e)) #handle generic bad user input.
   
    global tif_path 
    root = tk.Tk() #create the GUI window.
    root.title("SeeGULL") #title the window.
    root.geometry("1000x650") #default window size.
    root.resizable(True, True) #allow user to resize window.

    root.rowconfigure(1, weight=1)
    root.columnconfigure(1, weight=1) #define region for file bar.

    left_panel = ttk.Frame(root, padding=12)
    left_panel.grid(row=1, column=0, sticky="ns") #define region for left panel.

    right_sidebar = RightSideBar(root)
    right_sidebar.grid(row=1, column=1, sticky="nsew") #define region for right panel.

    tk.Label(left_panel, text="Longitude (EPSG:4326):").grid(row=0, column=0, padx=(12, 4), pady=(15, 8), sticky="w") #add text for longitude input box, push it to the left and add padding.
    lon_entry = tk.Entry(left_panel, width=22) #create the input box.
    lon_entry.grid(row=0, column=1, pady=(15, 8), sticky="w") #place input box into grid.
    lon_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    lon_help.grid(row=0, column=2, padx=6, pady=(15, 8), sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Latitude (EPSG:4326):").grid(row=1, column=0, padx=(12, 4), pady=8, sticky="w") #add text for latitude input box, push it to the left and add padding.
    lat_entry = tk.Entry(left_panel, width=22) #create the input box.
    lat_entry.grid(row=1, column=1, pady=8, sticky="w") #place input box into grid.
    lat_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    lat_help.grid(row=1, column=2, padx=6, pady=8, sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Observer height (m):").grid(row=2, column=0, padx=(12, 4), pady=8, sticky="w") #add text for observer height input box, push it to the left and add padding.
    height_entry = tk.Entry(left_panel, width=22) #create the input box.
    height_entry.grid(row=2, column=1, pady=8, sticky="w") #place helper widget into grid.
    height_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    height_help.grid(row=2, column=2, padx=6, pady=8, sticky="w") #place helper widget into grid.

    right_sidebar.point_selected_callback = set_coordinate_entries # assign function to update coordinates.

    submit_button = tk.Button(left_panel, text="Submit", command=submit) #create a button that when clicked runs the submit function.
    submit_button.grid(row=3, column=0, columnspan=3, pady=(18, 10)) #place button into grid.

    error_label = tk.Label(left_panel, text="", fg="red")
    error_label.grid(row=4, column=0, columnspan=3, pady=(0, 10))
    #attach a widget to display information regarding errors.

    file_hanlder()

    #attach a tooltip to the latitude help widget.
    LeftSideBar(
        lon_help,
        "Enter the coordinate's longitude in EPSG:4326.\nExample: -1.3276"
    )

    #attach a tooltip to the latitude help widget.
    LeftSideBar(
        lat_help,
        "Enter the coordinate's latitude in EPSG:4326.\nExample: 50.730251"
    )

    #attach a tooltip to the observer height help widget.
    LeftSideBar(
        height_help,
        "Enter observer height above the ground in metres.\nExample: 1.5"
    )

    if tif_path:
        right_sidebar.load_dem(tif_path) #load initial DEM preview.

    #start Tkinter event loop so it "listens" for user input.
    root.mainloop()

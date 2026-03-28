from vista import run_program
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
import csv

tif_path = None #file to be loaded from.
metadata_csv_path = None #file containing sample metadata.
loaded_sample_metadata = [] #store metadata loaded from CSV.

def start_gui(run_program): #entry point for the program.

    def set_coordinate_entries(lon, lat):
        row_index = selected_row_var.get() - 1 #work out which sample row should be auto-filled.

        if not (0 <= row_index < len(sample_entries)):
            return

        lon_entry, lat_entry, _ = sample_entries[row_index]

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
            self.raw_overlay = None #store raw sector overlay result.
            self.count_overlay = None #store frequency count result.
            self.observer_points_xy = [] #store observer cooridnates.

            self.point_selected_callback = None #reference to helper function for auto filling boxes on click.
            self.clicked_points = [] #store the coordinates the user has clicked on.
            self.tip = None

            self.rowconfigure(1, weight=1) #only allow preview to grow if needed.
            self.columnconfigure(0, weight=1) #allow preview section to be streched sideways.

            self.title_label = ttk.Label(self, text="DEM PREVIEW", font=("Segoe UI", 12, "bold")) #add Title.
            self.title_label.grid(row=0, column=0, sticky="w", pady=(0, 6)) #position Title.

            self.fig = Figure(figsize=(11, 5.8), dpi=100) #create a Matplotlib Figure object.
            self.ax_raw = self.fig.add_subplot(121) #add axes.
            self.ax_count = self.fig.add_subplot(122, sharex=self.ax_raw, sharey=self.ax_raw) #add axes.
            self.ax_raw.set_title("No DEM loaded") #if no DEM loaded, inform user.
            self.ax_count.set_title("No DEM loaded") #if no DEM loaded, inform user.
            self.ax_raw.set_xticks([])
            self.ax_raw.set_yticks([]) #do not show ticks around empty canvas message.
            self.ax_count.set_xticks([])
            self.ax_count.set_yticks([]) #do not show ticks around empty canvas message.

            self.canvas = FigureCanvasTkAgg(self.fig, master=self) #create a Tkinter compatible canvas wrapper for Matplotlib.
            self.canvas_widget = self.canvas.get_tk_widget() #render it with equivalent widget corresponding to a Matplotlib figure.
            self.canvas_widget.grid(row=1, column=0, sticky="nsew") #force DEM canvas to fill entire grid cell.

            self.toolbar_frame = ttk.Frame(self) #add container for Matplotlib toolbar.
            self.toolbar_frame.grid(row=2, column=0, sticky="ew") #position Matplotlib toolbar.
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame, pack_toolbar=False) #create the Matplotlib toolbar to add interactivity to the canvas.
            self.toolbar.update() #standard practice: refresh toolbar before displaying.
            self.toolbar.pack(side="left") #move toolbar to the left.
            
            self.show_overlay = tk.BooleanVar(value=True) #create a Boolean state for whether overlay should be visible, default to show.
            self.raw_overlay_im = None #default to no overlay image.
            self.count_overlay_im = None #default to no overlay image.
            self.raw_overlay_cbar = None #default to no colour bar.
            self.count_overlay_cbar = None #default to no colour bar.

            self.view_xlim = None
            self.view_ylim = None #store information about current zoom so it isn't lost on refresh.

            menubar = tk.Menu(parent) #create menu for the window in the top bar...
            parent.config(menu=menubar) #...and attach it.

            file_menu = tk.Menu(menubar, tearoff=0) #create file menu inside the top bar...
            menubar.add_cascade(label="File", menu=file_menu) #...and add it.
            file_menu.add_command(label="Load metadata CSV", command=load_metadata_file) #create a button to load metadata from CSV.

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
            if event.inaxes not in (self.ax_raw, self.ax_count):
                return #if click happens outside of canvas, do nothing.
            if event.xdata is None or event.ydata is None:
                return #if the graph coordinates cannot be worked out, do nothing.
            if self.toolbar.mode != "":
                return

            x = event.xdata #obtain x coordinate user clicked on.
            y = event.ydata #obtain y coordinate user clicked on.
            self.clicked_points.append((x, y)) #return coordinates user clicked on.

            if self.dem_crs is not None and self.point_selected_callback is not None: #if a coordinate system can be found, and the helper function is ready.
                transformer = Transformer.from_crs(self.dem_crs, "EPSG:4326", always_xy=True) #create DEM transformer.
                lon, lat = transformer.transform(x, y) #convert clicked position into longitude and latitude.
                self.point_selected_callback(lon, lat) #send coordinates to helper function.
            if self.ax_raw.has_data():
                self.view_xlim = self.ax_raw.get_xlim()
                self.view_ylim = self.ax_raw.get_ylim()

            self._redraw()

        def load_dem(self, dem_path):

            with rasterio.open(dem_path) as src:
                self.dem = src.read(1, masked=True) #read in the elevation grid.
                self.dem_transform = src.transform #read in transformer.
                self.dem_crs = src.crs #read in CRS.
                self.dem_path = dem_path #read in file path.

            self.raw_overlay = None
            self.count_overlay = None
            self.observer_points_xy = []
            self._redraw()

        def set_results(self, raw_overlay, count_overlay, observer_points=None):
            if self.dem is None:
                raise ValueError("Load a DEM before setting an overlay.") #check a DEM file is present.

            if raw_overlay.shape != self.dem.shape:
                raise ValueError("Overlay mask must have the same shape as the DEM.")

            if count_overlay.shape != self.dem.shape:
                raise ValueError("Overlay mask must have the same shape as the DEM.")

            self.raw_overlay = raw_overlay #store DEM LoS render.
            self.count_overlay = count_overlay #store DEM LoS render.
            self.observer_points_xy = observer_points if observer_points is not None else [] #store observer coordinates.
            self._redraw() #render.

        def toggle_overlay(self): #toggle overlay on/off.
            show = self.show_overlay.get() #retrieve current toggle state.

            if self.raw_overlay_im is not None: #if overlay exists.
                self.raw_overlay_im.set_visible(show) #toggle on/off.

            if self.count_overlay_im is not None: #if overlay exists.
                self.count_overlay_im.set_visible(show) #toggle on/off.

            if self.raw_overlay_cbar is not None: #if colourbar exists.
                self.raw_overlay_cbar.ax.set_visible(show) #toggle on/off.

            if self.count_overlay_cbar is not None: #if colourbar exists.
                self.count_overlay_cbar.ax.set_visible(show) #toggle on/off.

            self.canvas.draw_idle() #update display.

        def clear_overlay(self):
            self.raw_overlay = None #remove any previous overlay.
            self.count_overlay = None #remove any previous overlay.
            self.observer_points_xy = [] #remove any previous observer point.
            self._redraw() #render preview area again.

        def _remove_colourbars(self):
            if self.raw_overlay_cbar is not None:
                self.raw_overlay_cbar.remove()
                self.raw_overlay_cbar = None

            if self.count_overlay_cbar is not None:
                self.count_overlay_cbar.remove()
                self.count_overlay_cbar = None

        def _redraw(self):
            self.ax_raw.clear() #clean axes to prevent buildup.
            self.ax_count.clear() #clean axes to prevent buildup.
            self._remove_colourbars()

            if self.dem is None: #if no DEM given, show blank DEM message.
                self.ax_raw.set_title("No DEM loaded")
                self.ax_count.set_title("No DEM loaded")
                self.ax_raw.set_xticks([])
                self.ax_raw.set_yticks([])
                self.ax_count.set_xticks([])
                self.ax_count.set_yticks([])
                self.canvas.draw_idle()
                return

            show(
                self.dem,
                transform=self.dem_transform,
                ax=self.ax_raw,
                cmap="terrain"
            )  #render DEM base image.

            show(
                self.dem,
                transform=self.dem_transform,
                ax=self.ax_count,
                cmap="terrain"
            )  #render DEM base image.

            self.ax_raw.set_title("Raw sector overlays")
            self.ax_count.set_title("Frequency count heatmap")

            if self.raw_overlay is not None and self.show_overlay.get():
                raw_overlay = np.ma.masked_where(self.raw_overlay == 0, self.raw_overlay)

                raw_vmax = float(np.max(self.raw_overlay)) if np.any(self.raw_overlay > 0) else 1.0

                show(
                    raw_overlay,
                    transform=self.dem_transform,
                    ax=self.ax_raw,
                    cmap="bwr",
                    alpha=0.70,
                    zorder=20,
                    vmin=1,
                    vmax=raw_vmax
                )  #render DEM base image.

                self.raw_overlay_im = self.ax_raw.images[-1]
                self.raw_overlay_cbar = self.fig.colorbar(
                    self.raw_overlay_im,
                    ax=self.ax_raw,
                    label="Sector overlap"
                )

            else:
                self.raw_overlay_im = None

            if self.count_overlay is not None and self.show_overlay.get():
                count_overlay = np.ma.masked_where(self.count_overlay == 0, self.count_overlay)

                count_vmax = float(np.max(self.count_overlay)) if np.any(self.count_overlay > 0) else 1.0

                show(
                    count_overlay,
                    transform=self.dem_transform,
                    ax=self.ax_count,
                    cmap="bwr",
                    alpha=0.70,
                    zorder=20,
                    vmin=1,
                    vmax=count_vmax
                )  #render DEM base image.

                self.count_overlay_im = self.ax_count.images[-1]
                self.count_overlay_cbar = self.fig.colorbar(
                    self.count_overlay_im,
                    ax=self.ax_count,
                    label="Number of sightings"
                )

            else:
                self.count_overlay_im = None

            if self.observer_points_xy:
                xs = [x for x, _ in self.observer_points_xy]
                ys = [y for _, y in self.observer_points_xy]

                self.ax_raw.scatter(xs, ys, marker="x", s=80, linewidths=2, color="white", zorder=30) #
                self.ax_count.scatter(xs, ys, marker="x", s=80, linewidths=2, color="white", zorder=30) #

            self.ax_raw.set_xlabel("Easting (m)")
            self.ax_raw.set_ylabel("Northing (m)")
            self.ax_count.set_xlabel("Easting (m)")
            self.ax_count.set_ylabel("Northing (m)")

            if self.view_xlim is not None and self.view_ylim is not None:
                self.ax_raw.set_xlim(self.view_xlim)
                self.ax_raw.set_ylim(self.view_ylim) #redraw zoom based on saved information. 
                self.ax_count.set_xlim(self.view_xlim)
                self.ax_count.set_ylim(self.view_ylim) #redraw zoom based on saved information. 

            self.fig.tight_layout()
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
    def populate_sample_entries(samples):
        for lon_entry, lat_entry, height_entry, heading_entry in sample_entries: #for each entry row in the GUI.
            lon_entry.delete(0, tk.END)
            lat_entry.delete(0, tk.END)
            height_entry.delete(0, tk.END)
            heading_entry.delete(0, tk.END) #remove any old data from the boxes.

        for i, sample in enumerate(samples[:10]): #for each sample from the metadata CSV.
            lon_entry, lat_entry, height_entry, heading_entry = sample_entries[i] #retrieve longitude, latitude, observer height and heading.

            lon_entry.insert(0, str(sample["lon"])) #insert longitude loaded from CSV.
            lat_entry.insert(0, str(sample["lat"])) #insert latitude loaded from CSV.
            height_entry.insert(0, str(sample["observer_height"])) #insert observer height loaded from CSV.
            heading_entry.insert(0, str(sample["heading_deg"])) #insert heading loaded from CSV.

    def load_metadata_csv(file_path):
        samples = [] #will store sample data.

        with open(file_path, newline="", encoding="utf-8-sig") as csvfile: #open CSV.
            reader = csv.DictReader(csvfile) #read in data as a dictionary.

            required_columns = {"lon", "lat", "observer_height", "heading_deg"} #assert which columns are needed.
            if reader.fieldnames is None:
                raise ValueError("The CSV file has no header row.") #if there is no header row, raise an error.

            missing = required_columns - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV file is missing required columns: {', '.join(sorted(missing))}") #if there are any missing columns, raise an error.

            for row_index, row in enumerate(reader, start=2): #start at 2 because row 1 is the header.
                try:
                    lon = float(row["lon"])
                    lat = float(row["lat"])
                    observer_height = float(row["observer_height"]) #extract longitude, latitutde and observer height data.
                    heading_deg = float(row["heading_deg"]) #extract heading data.
                except ValueError:
                    raise ValueError(f"Invalid numeric value in CSV row {row_index}.") #if any data is not a number, raise an error.

                if not (-180 <= lon <= 180):
                    raise ValueError(f"Longitude out of range in CSV row {row_index}.") #if any longitude is out of range, raise an error.

                if not (-90 <= lat <= 90):
                    raise ValueError(f"Latitude out of range in CSV row {row_index}.") #if any latitude is out of range, raise an error.

                if observer_height <= 0:
                    raise ValueError(f"Observer height must be greater than 0 in CSV row {row_index}.") #if height is 0 or less, raise an error.

                if not (0 <= heading_deg < 360):
                    raise ValueError(f"Heading must be between 0 and 360 degrees in CSV row {row_index}.") #if heading is out of range, raise an error.

                samples.append({
                    "lon": lon,
                    "lat": lat,
                    "observer_height": observer_height,
                    "heading_deg": heading_deg
                }) #add data to samples.

        if len(samples) != 10:
            raise ValueError(f"CSV must contain exactly 10 samples for this proof of concept. Found {len(samples)}.") #if there were not exactly 10 samples, raise an error.

        return samples
    
    def metadata_handler(): 
        selected_metadata_var = tk.StringVar(value=metadata_csv_path if metadata_csv_path else "No metadata CSV selected") #Tkinter text displaying the current CSV path or "No metadata CSV selected".
        def load_metadata_file(): 
            global metadata_csv_path 
            global loaded_sample_metadata 

            file_path = filedialog.askopenfilename( #open a file picker window and store the chosen file path in file_path.
                parent=root, #pop up must be shown in the app.
                title="Open metadata CSV for VISTA", #text at the top of window.
                initialdir=".", #begin browsing in the directory of the program.
                filetypes=[
                    ("CSV files", "*.csv"), #only show files ending in .csv as the type we want the user to pick.
                ] #files that are shown as acceptable, in this case CSV.
            ) #use file explorer to allow the user to select a file.

            if not file_path: #check whether the user cancelled the file picker instead of selecting a file.
                return  #if user aborts, stop function.

            try:
                samples = load_metadata_csv(file_path) #read the CSV file and turn it into a validated list of sample dictionaries.
                metadata_csv_path = file_path #store the chosen CSV path.
                loaded_sample_metadata = samples #store the loaded sample list.
                selected_metadata_var.set(file_path) #update the Tkinter text variable to display the chosen CSV file path.
                populate_sample_entries(samples) #fill the visible entry boxes in the GUI with the values loaded from the CSV file.
                error_label.config(text="") #clear any old error message because loading succeeded.
            except Exception as e: 
                messagebox.showerror("Load error", f"Could not load metadata CSV:\n{e}") #if there is an error with loading the file, alert the user.

        return selected_metadata_var, load_metadata_file 
   
   
   
    def validate_inputs():
        if loaded_sample_metadata:
            return loaded_sample_metadata #if metadata has been loaded from CSV, use it directly.

        max_observer_height = 10000
        samples = []

        for idx, (lon_entry, lat_entry, height_entry, heading_entry) in enumerate(sample_entries, start=1):
            lon_text = lon_entry.get().strip() #retrieve user input for longitude.
            lat_text = lat_entry.get().strip() #retrieve user input for latitude.
            observer_height_text = height_entry.get().strip() #retrieve user input for observer height.
            heading_text = heading_entry.get().strip() #retrieve user input for heading.

            if not lon_text or not lat_text or not observer_height_text or not heading_text:
                raise ValueError(f"Please fill in longitude, latitude, observer height, and heading for sample {idx}.") #if any field is left blank, raise an error.

            try:
                lon = float(lon_text) 
                lat = float(lat_text)
                observer_height = float(observer_height_text)
                heading_deg = float(heading_text)
            except ValueError:
                raise ValueError(f"Longitude, latitude, observer height, and heading must all be numbers for sample {idx}.")
            #attempt to convert inputs to floats, if they cannot be converted, raise an error.

            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude must be between -180 and 180 for sample {idx}.") #valid range for ESPG:4326 latitude is -180 to 180.

            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude must be between -90 and 90 for sample {idx}.") #valid range for ESPG:4326 latitude is -90 to 90.

            if observer_height <= 0:
                raise ValueError(f"Observer height must be greater than 0 metres for sample {idx}.") #if user inputs negative observer height, raise an error.

            if observer_height > max_observer_height:
                raise ValueError(f"Observer height must not exceed {max_observer_height} metres for sample {idx}.") #if user inputs an observer height above the maximum, raise an error.

            if not (0 <= heading_deg < 360):
                raise ValueError(f"Heading must be between 0 and 360 degrees for sample {idx}.") #if heading is out of range, raise an error.

            samples.append({
                "lon": lon,
                "lat": lat,
                "observer_height": observer_height,
                "heading_deg": heading_deg
            })

        return samples #return inputs once validated so run_program can be run.
        
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

            metadata_title_label = ttk.Label(top_bar, text="Metadata CSV:") #label for metadata CSV path.
            metadata_title_label.pack(side="left", padx=(20, 4)) #move to the leftmost side, next to the button.

            metadata_label = ttk.Label(top_bar, textvariable=selected_metadata_var) #label for currently selected metadata CSV, or default text if no file loaded.
            metadata_label.pack(side="left", padx=10) #move to the leftmost side, next to the button.
            
    def submit():
            global tif_path
            error_label.config(text="") # clear any previous error message.
            try:
                if not tif_path:
                    raise ValueError("Please select a GeoTIFF file first.")

                sample_metadata = validate_inputs() #validate the user's inputs.

                right_sidebar.load_dem(tif_path) #load DEM into preview.

                result = run_program(
                    sample_metadata,
                    tif_path,
                    show_reference=False
                ) #run the main program with the three values on the embedded axes.

                right_sidebar.set_results(
                    result["raw_overlay"],
                    result["count_overlay"],
                    observer_points=result["observer_points_xy"]
                )

                right_sidebar.view_xlim = right_sidebar.ax_raw.get_xlim() #store current x zoom limit on graph.
                right_sidebar.view_ylim = right_sidebar.ax_raw.get_ylim() #store current y zoom limit on graph.

                right_sidebar.toggle_overlay() #update preview with current toggle state.

                right_sidebar.canvas.draw_idle() #refresh the embedded preview.

            except ValueError as e:
                show_error(str(e)) #handle invalid number input.
            except Exception as e:
                show_error(str(e)) #handle generic bad user input.
   
    global tif_path 
    root = tk.Tk() #create the GUI window.
    root.title("SeeGULL") #title the window.
    root.geometry("1350x760") #default window size.
    root.resizable(True, True) #allow user to resize window.

    root.rowconfigure(1, weight=1)
    root.columnconfigure(1, weight=1) #define region for file bar.

    left_panel = ttk.Frame(root, padding=12)
    left_panel.grid(row=1, column=0, sticky="ns") #define region for left panel.

    selected_metadata_var, load_metadata_file = metadata_handler() #create button to load in a CSV by defining variable and function with does so.
    right_sidebar = RightSideBar(root) #create right hand pannel.
    right_sidebar.grid(row=1, column=1, sticky="nsew") #define region for right panel.

    selected_row_var = tk.IntVar(value=1)
    sample_entries = []

    tk.Label(left_panel, text="Sample").grid(row=1, column=0, padx=(12, 4), pady=8, sticky="w") #add table heading.
    tk.Label(left_panel, text="Longitude (EPSG:4326):").grid(row=1, column=1, padx=(12, 4), pady=8, sticky="w") #add text for longitude input box, push it to the left and add padding.
    lon_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    lon_help.grid(row=1, column=2, padx=6, pady=8, sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Latitude (EPSG:4326):").grid(row=1, column=3, padx=(12, 4), pady=8, sticky="w") #add text for latitude input box, push it to the left and add padding.
    lat_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    lat_help.grid(row=1, column=4, padx=6, pady=8, sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Observer height (m):").grid(row=1, column=5, padx=(12, 4), pady=8, sticky="w") #add text for observer height input box, push it to the left and add padding.
    height_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    height_help.grid(row=1, column=6, padx=6, pady=8, sticky="w") #place helper widget into grid.

    tk.Label(left_panel, text="Heading (deg):").grid(row=1, column=7, padx=(12, 4), pady=8, sticky="w") #add text for heading input box, push it to the left and add padding.
    heading_help = tk.Label(left_panel, text="?", fg="blue", cursor="question_arrow") #create the helper widget and make the foreground blue.
    heading_help.grid(row=1, column=8, padx=6, pady=8, sticky="w") #place helper widget into grid.

    for i in range(10):
        tk.Label(left_panel, text=f"{i + 1}.").grid(row=i + 2, column=0, padx=(12, 4), pady=4, sticky="w") #add row label.

        lon_entry = tk.Entry(left_panel, width=16) #create the input box.
        lon_entry.grid(row=i + 2, column=1, pady=4, sticky="w") #place input box into grid.

        lat_entry = tk.Entry(left_panel, width=16) #create the input box.
        lat_entry.grid(row=i + 2, column=3, pady=4, sticky="w") #place input box into grid.

        height_entry = tk.Entry(left_panel, width=12) #create the input box.
        height_entry.grid(row=i + 2, column=5, pady=4, sticky="w") #place helper widget into grid.

        heading_entry = tk.Entry(left_panel, width=12) #create the input box.
        heading_entry.grid(row=i + 2, column=7, pady=4, sticky="w") #place helper widget into grid.

        sample_entries.append((lon_entry, lat_entry, height_entry, heading_entry))

    right_sidebar.point_selected_callback = set_coordinate_entries # assign function to update coordinates.

    submit_button = tk.Button(left_panel, text="Submit", command=submit) #create a button that when clicked runs the submit function.
    submit_button.grid(row=12, column=0, columnspan=9, pady=(18, 10)) #place button into grid.

    error_label = tk.Label(left_panel, text="", fg="red")
    error_label.grid(row=13, column=0, columnspan=9, pady=(0, 10))
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

    #attach a tooltip to the heading help widget.
    LeftSideBar(
        heading_help,
        "Enter the viewing direction as a compass bearing in degrees.\n0 = north, 90 = east, 180 = south, 270 = west"
    )

    if tif_path:
        right_sidebar.load_dem(tif_path) #load initial DEM preview.

    #start Tkinter event loop so it "listens" for user input.
    root.mainloop()

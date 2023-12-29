from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from dataclasses import dataclass

import sys
import os

from qt_yeti.qt_yeti_general import *
#from qt_yeti.qt_yeti_functions import *
#from qt_yeti.qt_yeti_hardware_settings_tab import *

import numpy as np
import scipy
from scipy.signal import find_peaks, find_peaks_cwt, savgol_filter
from scipy.integrate import quad
from scipy.spatial.transform import Rotation as R

import csv
import configparser # https://docs.python.org/3/library/configparser.html
from datetime import datetime
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as AX
matplotlib.use("Qt5Agg")
plt.rcParams['axes.linewidth'] = 0.5 # Alternative: plt.setp(self.axes_flat_field.spines.values(), linewidth=0.5)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure

from astropy.io import fits
#%% Meta stuff
###############################################################################################################################
	# Use property()
	# https://www.programiz.com/python-programming/property
	# https://realpython.com/python-property/

in_rad = 180/np.pi
in_deg = np.pi/180

def col(x, spectrogram_xsize) -> np.uint32:
	"""
	Translate x to a column index (trivial)\r\n
	This function is self-inverting.
	"""
	x_res = np.rint(x)
	col_res = np.rint(spectrogram_xsize)-1
	return np.int32(np.clip(x_res, 0, col_res))

def row(y, spectrogram_ysize) -> np.uint32:
	"""
	Translate y to a row index row = (max row index) - y.\r\n
	This function is self-inverting.
	"""
	y_res = np.rint(y)
	row_res = np.rint(spectrogram_ysize)-1
	return np.int32(row_res - np.clip(y_res, 0, row_res))

@dataclass(init=True, repr=True)
class Spot:
	x: float
	y: float
	order: int
	wavelength: float
	intensity: float = 1.0

@dataclass
class Order:
	"""
	Echelle order
	=============

	Parameters:
		number_m (float): Absolute order number
		x_span (ndarray): Range starting from x1 to x2 in discrete steps of 1
		fit_parameters (ndarray): Array of fit parameters starting with the (x_offset) followed by the coefficient of the highest order when using polynomials
		calibration_parameters (ndarray): Array of calibration parameters to translate from pixel and order_number_m to wavelength
	"""
	number_m: float = None
	x_range: np.ndarray = None
	fit_parameters: np.ndarray = None
	
	calibration_lines = []

	#Where is the function?
	calibration_parameters: np.ndarray = None
	summation_slice: slice = None
	summation_weights: np.ndarray = None
	order_number_calibtrated = False
	order_dispersion_calibrated = False

class Spectrogram:
	"""Specrogram Class

	Check detailed code.

	Returns:
		Spectrogram: Class of type Spectrogram
	"""
	order_centers_list = []
	order_list = []
	order_fit_coefficients = []
	dispersion_fit_coefficients = []
	fit_function_string = None

	def __init__(self, filename):
		self.header = 0
		self.data = None
		self.filename = filename
		self.shape = 0
		self.xsize = 0
		self.ysize = 0
		self.xrange = 0
		self.yrange = 0
		self.intmax = 0
		self.intmin = 0
		self.update_spectrogram(self.filename)

	def __repr__(self) -> str:
		return f"Spectrogram(filename={bcolors.OKBLUE}{self.filename}{bcolors.ENDC}, xsize = {self.xsize}, ysize = {self.ysize})"
	
	def _set_spectrogram_properties(self,spectrogram_data=None):
		self.header = 0
		self.data = np.asarray(spectrogram_data)
		self.shape = self.data.shape
		self.ysize, self.xsize = self.shape
		self.xrange = np.arange(self.xsize)
		self.yrange = np.arange(self.ysize)
		self.intmax = self.data.max().max()
		self.intmin = self.data.min().min()

	def update_spectrogram(self, filename):
		self.filename = filename
		if(filename == "QtYeti.Sample"):
			self._set_spectrogram_properties( np.abs(np.random.randn(QT_YETI.DETECTOR_Y_PIXELS, QT_YETI.DETECTOR_X_PIXELS)) )
			return self.intmin, self.intmax

		try:
			# Get primary header
			self.header = fits.getheader(self.filename)
			# Check if spectrogram is loaded
			if( self.header['NAXIS'] == 2):
				self._set_spectrogram_properties(np.asarray(fits.getdata(self.filename), dtype=np.int64))
				self.intmin, self.intmax = self.find_mean_minmax()
			else:
				QtYetiLogger(QT_YETI.ERROR,f"Wrong fits type in file: {self.filename}. Please enter valid spectrogram. Loading Sample data now.")
				self.update_spectrogram("QtYeti.Sample")
		except Exception as error:
			QtYetiLogger(QT_YETI.ERROR,f"{error}")

		return self.intmin, self.intmax

	def find_mean_minmax(self):
		row_max, col_max = np.unravel_index(np.argmax(self.data), self.shape)
		max_row_max_index = np.clip(row_max+1,0,self.shape[0])
		max_col_max_index = np.clip(col_max+1,0,self.shape[1])
		min_row_max_index = np.clip(row_max-1,0,self.shape[0])
		min_col_max_index = np.clip(col_max-1,0,self.shape[1])
		mean_max = np.mean(self.data[min_row_max_index:max_row_max_index+1,min_col_max_index:max_col_max_index+1])

		row_min, col_min = np.unravel_index(np.argmin(self.data), self.shape)
		max_row_min_index = np.clip(row_min+1,0,self.shape[0])
		max_col_min_index = np.clip(col_min+1,0,self.shape[1])
		min_row_min_index = np.clip(row_min-1,0,self.shape[0])
		min_col_min_index = np.clip(col_min-1,0,self.shape[1])
		mean_min = np.mean(self.data[min_row_min_index:max_row_min_index+1,min_col_min_index:max_col_min_index+1])

		return np.int64(np.rint(mean_min)), np.int64(np.rint(mean_max))

	def save_order_fit_coefficients(self, loaded_filename):

		if(loaded_filename==""):
			QtYetiLogger(QT_YETI.WARNING,"Empty filename.", True)
			return
		
		if( loaded_filename != self.filename):
			QtYetiLogger(QT_YETI.ERROR,f"Unexpected filename. Requested and loaded file name is different. Requested: \"{loaded_filename}\" vs. Loaded: \"{self.filename}\".")
			return
		
		if(self.order_fit_coefficients): # Not empty
			os_path,os_file = os.path.split(loaded_filename)
			os_filename,os_extension = os.path.splitext(os_file)
			#trace_fit_filename = os_path + "/" + "Order_Fit_Coefficients_" + os_filename + ".txt"
			trace_fit_filename = f"{os_path}/Order_Fit_Coefficients_{os_filename}.txt"
			header = "Order, x_start, x_stop, a_n, a_(n-1)..., a_0"
			header = self.fit_function_string
			FileSaver(trace_fit_filename, header, self.order_fit_coefficients)
			QtYetiLogger(QT_YETI.MESSAGE,"Fit Order Coefficients saved.",True)
		return

	@classmethod
	def load_order_fit_coefficients(cls, requested_filename = ""):

		fit_coefficients = []

		if( requested_filename == ""):
			QtYetiLogger(QT_YETI.ERROR,"Empty filename.",True)
			return []
		# Remove a part of string from requested_filename
		try:
			part1,part2 = requested_filename.split(sep="Order_Fit_Coefficients_")
			matching_filename = f"{part1}{part2}" # part1 + part2

			cls.order_fit_coefficients = FileReader(requested_filename).read_rows().tolist()

			for order_number, x_start, x_stop, *fit_output in cls.order_fit_coefficients:
				cls.order_list.append( Order(order_number, np.arange(x_start, x_stop+1, 1, dtype=np.int32), np.asarray(fit_output)))

			QtYetiLogger(QT_YETI.MESSAGE,f"Fit Order Coefficients loaded from file via ClassMethod.",True)

		except ValueError:
			QtYetiLogger(QT_YETI.ERROR,f"Requested file to read in: \"{requested_filename}\" is not an Order Fit Coefficients file.")
			QtYetiLogger(QT_YETI.ERROR, f"dirty hack in", True)
		
		return fit_coefficients

	@classmethod
	def update_order_fit_coefficients(cls, order_fit_coefficients: list):
		if( order_fit_coefficients != []):
			cls.order_fit_coefficients = order_fit_coefficients
			cls.update_fit_function()
			QtYetiLogger(QT_YETI.MESSAGE,"Fit Order Coefficients updated via ClassMethod.",True)
		else:
			QtYetiLogger(QT_YETI.ERROR, f"Unable to set Order Fit Coefficients. Run Tracer first or load a coefficient file.")

	@classmethod
	def update_order_list(cls, order_list: list):
		if( cls.order_list != []):
			del cls.order_list
		if( order_list != []):
			cls.order_list = order_list
			QtYetiLogger(QT_YETI.MESSAGE,"Order List updated via ClassMethod.",True)

	@classmethod
	def update_order_centers_list(cls, order_centers_list: list):
		if( order_centers_list):
			cls.order_centers_list = order_centers_list
			QtYetiLogger(QT_YETI.MESSAGE,"Order centers list updated via ClassMethod.",True)
		else:
			QtYetiLogger(QT_YETI.ERROR,"Order centers list not set or updated.",True)

	@classmethod
	def insert_to_order_list(cls, y_coordinate: float):
		if (cls.order_centers_list):

			y_coordinate =int(np.rint(y_coordinate))

			x_position,_ = cls.order_centers_list[0]
			y_positions = np.asarray([dot_y for _,dot_y in cls.order_centers_list])
			array_length = len(cls.order_centers_list)
			insertion_index = np.NaN

			if( y_positions[0] > y_positions[-1]):
				y_positions = np.sort(y_positions)
				insertion_index = array_length - np.searchsorted(y_positions, y_coordinate)

			else:
				insertion_index = np.searchsorted(y_positions,y_coordinate)
			
			cls.order_centers_list.insert(insertion_index, [x_position, y_coordinate])
			QtYetiLogger(QT_YETI.MESSAGE,f"Inserted into order center list via ClassMethod. Inserted [x,y] = [{x_position},{y_coordinate}].",True)

		else:
			QtYetiLogger(QT_YETI.ERROR,f"Inserting into order center list via ClassMethod not possible. No order centers defined yet.", True)

	@classmethod
	def remove_from_order_list(cls, y_coordinate: float):
		QtYetiLogger(QT_YETI.ERROR,f"Nothing happened",True)

	@classmethod
	def get_order_list(cls):
		if( cls.order_list ):
			QtYetiLogger(QT_YETI.MESSAGE,"Order List returned via ClassMethod.",False)
			return cls.order_list

	@classmethod
	def set_absolute_order_number_m(cls, first_absolute_order: int):
		"""Set Orders to their physical and absolute order number

		Parameters:
			first_absolute_order (int): Correct physical order number m of first traced order
		"""

		if( cls.order_list ):
			QtYetiLogger(QT_YETI.MESSAGE,"Calibrating Orders for absolute order m via ClassMethod.",False)

		#first_absolute_order = int(QT_YETI.)
		
		orders_per_group = int(QT_YETI.SPECTROMETER_IMAGE_SLICER_ORDERS_PER_GROUP)
		subgroup_adder = QT_YETI.SPECTROMETER_IMAGE_SLICER_SUBORDER_SUFFIX

		number_of_orders = len(cls.order_list)
		has_image_slicer = QT_YETI.SPECTROMETER_HAS_IMAGE_SLICER
		orders_are_groupable = (number_of_orders % orders_per_group == 0)
		group_image_slicer = (has_image_slicer and orders_are_groupable)

		QtYetiLogger(QT_YETI.WARNING,f"Include the case into the programm when the spectra are reversed. e.g. curved upward.")

		absolute_m = None
		for index, order in enumerate(cls.order_list):

			if( group_image_slicer ):
				# Group via // operator. 0//3 = 0, 1//3 = 0, 2//3 = 0, 4//3  = 1 ...
				# Use % to add the correct subgroup adder
				absolute_m = (index // orders_per_group  + abs(first_absolute_order) + (index % orders_per_group) * subgroup_adder)

			else:
				# There are more or less orders that grouped in sets of QT_YETI.SpectrometerImageslicerOrdersPerGroup 
				absolute_m = (index) + abs(first_absolute_order)

			if( first_absolute_order < 0):
				absolute_m = -1 * absolute_m
			order.number_m = absolute_m
			order.order_number_calibtrated = True

	@classmethod
	def update_fit_function(cls):
		
		#print(cls.order_fit_coefficients[0])
		#print(cls.order_fit_coefficients[0][3:])

		func_string = f"Order, x_start, x_stop"
		for degree in range(len(cls.order_fit_coefficients[0][3:])-1,-1,-1):
			func_string += f", a_{degree}"
		cls.fit_function_string = func_string
		QtYetiLogger(QT_YETI.MESSAGE,"Fit Function String updated via ClassMethod.",True)

class OrderTracerSettings:
	def __init__(self, origin: str=None):

		self.spotsize_px = 					QT_YETI.DETECTOR_SPOT_SIZE_PX
		self.image_slicer = 				QT_YETI.SPECTROMETER_HAS_IMAGE_SLICER
		self.image_slicer_separation_px = 	QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX

		if( origin == 'from_file'):
			try:
				config = configparser.SafeConfigParser()
				config.optionxform=str
				config.read(QT_YETI.SETTINGS_INI_PATH)

				self.first_absolute_order =			config.getint("TRACER","TracerFirstAbsoluteOrder")
				self.order_alignment = 				config.getint("TRACER","TracerOrderAlignment")

				self.distance_to_image_edge_px = 	config.getint("TRACER","TracerDistanceToImageEdgePx")
				self.samples_per_order = 			config.getint("TRACER","TracerSamplesPerOrder")

				self.peak_distance_px = 			config.getint("TRACER","TracerPeakDistancePx")
				self.peak_height = 					config.getfloat("TRACER","TracerPeakHeight")
				self.peak_prominence = 				config.getfloat("TRACER","TracerPeakProminence")
				self.peak_width_px = 				config.getint("TRACER","TracerPeakWidthPx")

				self.smoothing_stiffness = 			config.getfloat("TRACER","TracerSmoothingStiffness")
				self.smoothing_order = 				config.getint("TRACER","TracerSmoothingOrder")

				del config

			except:
				QtYetiLogger(-1,"Error while reading OrderTracerSettings.",True)
		else:
			self.spotsize_px = 					0
			self.image_slicer = 				False
			self.image_slicer_separation_px = 	0

			self.distance_to_image_edge_px = 	0
			self.samples_per_order = 			0

			self.peak_distance_px = 			0
			self.peak_height = 					0.0
			self.peak_prominence = 				0.0
			self.peak_width_px = 				0

			self.smoothing_stiffness = 			0.0
			self.smoothing_order = 				0

	def to_file(self):
		config = configparser.SafeConfigParser()
		config.optionxform=str
		config.read(QT_YETI.SETTINGS_INI_PATH)

		config.set("TRACER","TracerFirstAbsoluteOrder",f"{self.first_absolute_order}")

		config.set("TRACER","TracerDistanceToImageEdgePx",f"{self.distance_to_image_edge_px}")
		config.set("TRACER","TracerSamplesPerOrder",f"{self.samples_per_order}")

		config.set("TRACER","TracerPeakDistancePx",f"{self.peak_distance_px}")
		config.set("TRACER","TracerPeakHeight",f"{self.peak_height}")
		config.set("TRACER","TracerPeakProminence",f"{self.peak_prominence}")
		config.set("TRACER","TracerPeakWidthPx",f"{self.peak_width_px}")

		config.set("TRACER","TracerSmoothingStiffness",f"{self.smoothing_stiffness}")
		config.set("TRACER","TracerSmoothingOrder",f"{self.smoothing_order}")

		with open(QT_YETI.SETTINGS_INI_PATH, 'w') as configfile:
			config.write(configfile)
		del config

#%% Tab Abstract classes 
###############################################################################################################################
"""
TEST: Gerneric Tabs to make code smaller. Prepared, but not used yet
"""
# Figure Canvas → Base Class → Needs derived classes FlatFieldCanvas,CalibratorCanvas,SpectrometerCanvas

class A(object):
	def __init__(self):
		super(A,self).__init__()
		print(f"A __init__() done.")
	
	def saga(self):
		print(f"Methode aus A")
		
class B(object):
	def __init__(self):
		super(B,self).__init__()
		print(f"B __init__() done.")

	def sagb(self):
		print(f"Methode aus B")

class C(B,A):
	def __init__(self):
		super(C,self).__init__()
		print(f"C __init__() done." )
	def sagc(self):
		print(f"Methode aus C")

class GernericTab(QWidget):
	def __init__(self, parent, canvas):
		super(GernericTab, self).__init__(parent)

		# Create Matplotlib Canvas
		self.figure_canvas = canvas

		# Setup and customize
		self.setupTabStructure()
		self.customizeTab()
		
		for child in self.findChildren((QWidget, QPushButton, QSpinBox)):
			child.setFocusPolicy(Qt.NoFocus)
		self.setFocusPolicy(Qt.NoFocus)

	def setupTabStructure(self):
		# Top Level Tab layout
		self.tab_layout = QVBoxLayout()
		self.setLayout(self.tab_layout)

		# Add Matplitlib Canvas
		self.tab_layout.addWidget(self.figure_canvas)
		self.tab_layout.addWidget(self.figure_canvas.return_navigation_bar())

		# Add Control Panel
		self.control_panel = QWidget()
		self.control_panel.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
		self.tab_layout.addWidget(self.control_panel)

	def customizeTab(self):
		# Fill Control Panel
		self.control_panel_layout = QGridLayout()
		self.control_panel_layout.setContentsMargins(0,0,0,0)
		self.control_panel.setLayout(self.control_panel_layout)
		
		#self.control_panel_layout.setSpacing(0)

		# Active order box
		self.current_order_box = QWidget()
		self.current_order_box.setLayout(QHBoxLayout())
		self.current_order_spinbox = QSpinBox()
		self.current_order_box.layout().addWidget(self.current_order_spinbox)
		self.current_order_box.layout().addWidget(QLabel("Current Order [m]"))
		self.current_order_box.layout().setContentsMargins(0,0,0,0)

		# Create Buttons
		self.action_load_spectrogram_btn	= QPushButton("Load Spectrogram")
		self.action_load_coefficients_btn	= QPushButton("Load Fit Coefficients")
		self.action_save_currentorder_btn	= QPushButton("Save Current Order")
		self.action_save_allorders_btn		= QPushButton("Save All Orders")

		self.control_panel_layout.addWidget(self.current_order_box,0,0)
		self.control_panel_layout.addWidget(self.action_load_spectrogram_btn,0,1)
		self.control_panel_layout.addWidget(self.action_load_coefficients_btn,0,2)
		self.control_panel_layout.addWidget(self.action_save_currentorder_btn,0,3)
		self.control_panel_layout.addWidget(self.action_save_allorders_btn,0,4)

		# connect Signals/Slots
		self.current_order_spinbox.editingFinished.connect(self.gui_set_order_index)
		self.current_order_spinbox.valueChanged.connect(self.gui_set_order_index)

		self.action_load_spectrogram_btn.clicked.connect(self.gui_load_spectrogram_file)
		self.action_load_coefficients_btn.clicked.connect(self.gui_load_fit_coefficients)
		
		self.action_save_currentorder_btn.clicked.connect(self.gui_save_single_order_to_fit)
		self.action_save_allorders_btn.clicked.connect(self.gui_save_all_orders_to_fit)

	# Signals / Slots
	@pyqtSlot()
	def gui_load_spectrogram_file(self):
		caption = "Select spectrogram file"
		initial_filter="Fits files (*.fits)"
		file_filter="Fits files (*.fits);; All files (*.*)"
		requested_filename, _  = QFileDialog.getOpenFileName(self, caption = caption, initialFilter=initial_filter, filter=file_filter)

		if(requested_filename != ""):
			int_min, int_max = self.figure_canvas.load_spectrogram(requested_filename)
			self.spectrogram_filename = requested_filename

	def gui_update_spectrum(self):
		QtYetiLogger(1,"gui_update_spectrum() triggered. No action.")
		pass

	def gui_load_fit_coefficients(self):
		pass
		QtYetiLogger(1,"gui_load_fit_coefficients() triggered. No action.")
		
	def gui_save_single_order_to_fit(self):
		# One file or two files? Depending on Image Slicer. Can FIT file handle 2 spectra?
		pass
		QtYetiLogger(1,"gui_save_single_order_to_fit() triggered. No action.")

	def gui_save_all_orders_to_fit(self):
		pass
		QtYetiLogger(1,"gui_save_all_orders_to_fit() triggered. No action.")
	
	def gui_set_order_index(self):
		self.current_order_spinbox.setValue( self.figure_canvas.update_spectrum( self.current_order_spinbox.value() ) )

class GenericCanvas( FigureCanvasQTAgg ):
	def __init__(self, parent=None, width=QT_YETI.MATPLOTLIB_CANVAS_WIDTH, height=QT_YETI.MATPLOTLIB_CANVAS_HEIGHT, dpi=QT_YETI.MATPLOTLIB_DPI):
		self.control_figure = plt.figure(figsize=(width, height), dpi=dpi)
		super(GenericCanvas, self).__init__(self.control_figure)

		# https://github.com/matplotlib/matplotlib/issues/707/
		# https://stackoverflow.com/questions/22043549/matplotlib-and-qt-mouse-press-event-key-is-always-none
		self.setFocusPolicy( Qt.ClickFocus )
		self.setFocus()

		self.navigationbar = None
		self.navigationbar = NavigationToolbar2QT(self, parent=None)

		# Setup sample spectrogram		
		self.CurrentSpectrogram = Spectrogram("QtYeti.Sample")

		# Setup all plots and callbacks
		# self.setup_plots()
		pass

		# Final touch
		# self.control_figure.tight_layout()
		pass

	### Plots and Callbacks ###
	def setup_plots(self):
		pass

	### Navigation Bar ###
	def return_navigation_bar(self):
		#self.navigationbar = NavigationToolbar2QT(self, parent=None)
		return self.navigationbar

	### MPL Callbacks ###
	def canvas_key_or_mouse_event(self, event):
		if(event.inaxes == self.axes_spectrogram):
			evt_x = np.int32(np.rint(event.xdata))
			evt_y = np.int32(np.rint(event.ydata))
			QtYetiLogger(0,f"Nearest order index {self.find_nearest_order_index(evt_x, evt_y)}")
			self.draw_idle()
		pass

	# Plotting
	def load_spectrogram(self, requested_filename = ""):
		if(requested_filename == ""):
			QtYetiLogger(-1,"No file name provided.")
			return -1
		QtYetiLogger(0,f"{requested_filename} loaded.")
		# Update CurrentSpectrogram
		int_min, int_max = self.CurrentSpectrogram.update_spectrogram(requested_filename)


		## Plot spectrogram
		# https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively
		self.spectrogram_plot.set_data(self.CurrentSpectrogram.data)
		self.spectrogram_plot.set_extent([0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1])
		self.spectrogram_plot.set_clim(int_min, int_max)

		self.spectrum_plot.set_data(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[1337,:])
		self.spectrum_plot.axes.set_ylim([0, int_max])

		self.draw_idle()
		return int_min, int_max

	def plot_spectrum(self, order_index):
		# Extract fit coefficients
		current_xrange = self.CurrentSpectrogram.xrange
		current_params = np.asarray(Spectrogram.order_fit_coefficients[order_index])[1:]

		# Create fit polinomial per order
		fitted_polynomial = np.asarray(echelle_order_fit_function(current_xrange, *current_params))

		# Discretize and convert to row indices from (x,y) to (r,c)
		discretized_polynomial = np.clip( np.rint(fitted_polynomial), 0, self.CurrentSpectrogram.shape[0])
		# discretized_polynomial = discretized_polynomial.astype(np.uint32) # row() takes care of this
		discretized_rows = row(discretized_polynomial, self.CurrentSpectrogram.ysize)

		## Experiment
		data_tuple,summation_tuple = echelle_order_spectrum_extraction(self.CurrentSpectrogram, order_index, OrderTracerSettings())
		summation_offsets = summation_tuple[0].repeat(self.CurrentSpectrogram.xsize)
		
		dynamic_masking_matrix = np.full(self.CurrentSpectrogram.shape,0.4)
		dynamic_masking_columns = np.tile(current_xrange, summation_tuple[1])
		dynamic_masking_rows = np.tile(discretized_rows, summation_tuple[1])
		dynamic_masking_matrix[dynamic_masking_rows+7 + summation_offsets, dynamic_masking_columns] = 1
		dynamic_masking_matrix[dynamic_masking_rows+7 - summation_offsets, dynamic_masking_columns] = 1
		
		masked_data = self.CurrentSpectrogram.data * dynamic_masking_matrix


		self.spectrogram_plot.set_data(masked_data)
		self.spectrum_plot.set_data(current_xrange,data_tuple[0])

		QtYetiLogger(-1,f"Remember: Echelle tracer does not work properly yet. Polynomial is wrong by half of IMG_SL")
		self.center_poly_plot.set_data(current_xrange, fitted_polynomial-QT_YETI.HARDWARE_IMAGE_SLICER_OFFSET/2)
		self.upper_poly_plot.set_data(current_xrange, fitted_polynomial)
		self.lower_poly_plot.set_data(current_xrange, fitted_polynomial-QT_YETI.HARDWARE_IMAGE_SLICER_OFFSET)

		#self.spectrum_plot.set_data(current_xrange, self.CurrentSpectrogram.data[int(Spectrogram.order_fit_coefficients[order_index, -1]),:])
		self.draw_idle()

	def update_spectrum(self, order_number: int):
		order_index = -1
		if( Spectrogram.order_list != []):	
			# Limit order index to the size of the fit coefficient list entries
			order_index = int(np.rint(np.clip( order_number-1, 0, len(Spectrogram.order_list)-1)))
			self.plot_spectrum(order_index)
		else:
			QtYetiLogger(QT_YETI.ERROR,f"No Fit Coefficients loaded.")
		
		return int(order_index + 1)

	def find_nearest_order_index(self,x = 0, y = 0):
		if (Spectrogram.order_fit_coefficients != []):
			orders_and_coefficients = np.asarray(Spectrogram.order_fit_coefficients)
			orders = orders_and_coefficients[:,0]
			coefficients = orders_and_coefficients[:,1:]

			ylist = []
			for i in np.arange(0,len(orders)-1):
				params = coefficients[i]
				y_fit = echelle_order_fit_function(x, *params)
				ylist.append(y_fit)

			nearest_index = np.argmin(np.abs( np.asarray(ylist)-y ))

			return nearest_index

		else:
			QtYetiLogger(-1,"Nearest Order not found. Returning np.NAN.")
			return np.NAN

class SpecificCanvas(GenericCanvas):
	def __init__(self, parent):
		super(SpecificCanvas,self).__init__(parent=parent)
		self.setup_plots()
		self.control_figure.tight_layout()

	### Plots and Callbacks ###
	def setup_plots(self):
		# Axes
		self.axes_spectrogram = plt.subplot2grid((16,16),(0,0),colspan=16, rowspan = 9, fig=self.control_figure, label="Full_Spectrogram")
		self.axes_spectrum = plt.subplot2grid((16,16),(9,0),colspan=16, rowspan = 6, fig=self.control_figure, label="Extracted_Spectrum")
		self.axes_spectrogram.set_ylabel("$m · \lambda(X,Y)$")
		self.axes_spectrum.set_xlabel("$\lambda(X,Y)$")
		self.axes_spectrum.set_ylabel("Counts (arb. u.)")

		self.spectrogram_plot = self.axes_spectrogram.imshow(self.CurrentSpectrogram.data,\
			vmin=self.CurrentSpectrogram.intmin,\
			vmax=self.CurrentSpectrogram.intmax,\
			#cmap = 'gray',\
			interpolation='none',\
			extent=[\
				0,\
				self.CurrentSpectrogram.xsize-1,\
				0,\
				self.CurrentSpectrogram.ysize-1],\
			aspect='auto',\
			label = "2D_Spectrogram")
		[self.spectrum_plot] = self.axes_spectrum.plot(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[256,:], linewidth=0.3, label="Data_Spectrum")
		self.axes_spectrum.set_xlim( self.CurrentSpectrogram.xrange.min(), self.CurrentSpectrogram.xrange.max() )
		self.axes_spectrum.set_ylim( self.CurrentSpectrogram.intmin, self.CurrentSpectrogram.intmax )

		[self.upper_poly_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.2,linewidth="0.5",color=YetiColors.BLUE, label="Upper_Poly_Plot")
		[self.center_poly_plot] = self.axes_spectrogram.plot(0,0,alpha=0.2,linewidth="0.5",color=YetiColors.GREEN,label="Center_Poly_Plot")
		[self.lower_poly_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.2,linewidth="0.5",color=YetiColors.RED,  label="Lower_Poly_Plot")

		# Event handling
		self.mpl_connect('button_release_event', self.canvas_key_or_mouse_event)
		self.mpl_connect('key_press_event', self.canvas_key_or_mouse_event)

class SpecificTab(GernericTab):
	def __init__(self, parent):
		super(SpecificTab,self).__init__(parent,SpecificCanvas(parent=self))

#%% possible Canvas functions
def delete_mpl_plt_object_by_label(mpl_axes_object_list, line_label: str):
	"""
	Delete all mpl_plt objects such as lines, texts, ... from a specific axes object

	Parameters:
		mpl_axes_object_list (_type_): ArtistList containing lines or texts ... axes.lines or axes.texts
		line_label (str): String label
	"""	
	all_plot_objects = mpl_axes_object_list

	if( not (isinstance(all_plot_objects, matplotlib.axes._base._AxesBase.ArtistList))):
		return
	
	found_lines = []

	for index, line in enumerate(all_plot_objects):
		if(line_label in line.get_label()):
			found_lines.append(index)
	found_lines.reverse()

	for index in found_lines:
		all_plot_objects.pop(index)


#%% Auxiliary functions
###############################################################################################################################

class QtYetiSettingsReader():
	def __init__(self):
		self.config = configparser.SafeConfigParser()
		self.config.optionxform=str
		self.config.read(QT_YETI.SETTINGS_INI_PATH)

	def __del__(self):
		del self.config

class FileSaver():
	def __init__(self, filename, header, data):
		with open( filename, "a", newline='') as file_handle:
			file_handle.write(f"{header}\r\n")
			CSVWriter = csv.writer(file_handle)
			CSVWriter.writerows(data)
			del CSVWriter
			file_handle.close()
			QtYetiLogger(0,f"FileSaver: written data to {filename}")

class FileReader():

	def __init__(self, filename = ""):
		if(filename == ""):
			QtYetiLogger(QT_YETI.ERROR, "No Filename provieded")
			return -666
		else:
			self.filename = filename

	def read_rows(self) -> np.ndarray:
		rows = []
		with open(self.filename, "r") as file_handle:
			CSVReader = csv.reader(file_handle)

			# Skip header
			header = next(CSVReader, None)

			if (header):
				for line,row in enumerate(CSVReader):
					line += 2
					# Check if there is any non-numeric data
					try:
						rows.append( np.asarray_chkfinite( row, dtype=float ) )
					
					# Yes, and thus replace everything by NaN
					except ValueError:
						row_length = len(row)
						rows.append( np.asarray( row_length * [np.NAN] ) )
						QtYetiLogger(QT_YETI.ERROR, f"Non-numeric data found in {bcolors.FAIL}\"{self.filename}\"{bcolors.ENDC} in line {line}. Check the file." + bcolors.ENDC)

		return np.asarray(rows)

class QtYetiLogger():
	def __init__(self, message_type=0, message="", show_call_stack = False):
		# Get function, that called this here
		self.caller_level = 1
		
		try:
			self.calling_module = sys._getframe( self.caller_level ).f_back.f_globals["__name__"]
		except:
			self.calling_module = "→no_calling_module←"

		try:
			self.calling_class = sys._getframe( self.caller_level ).f_back.f_locals["self"].__class__.__qualname__
		except:
			self.calling_class = "→no_calling_class←"

		try:
			self.caller = sys._getframe( self.caller_level ).f_locals["self"].__class__.__name__
		except:
			self.caller = "→no_caller←"
		
		try:
			self.called_function = sys._getframe( self.caller_level ).f_code.co_name
		except:
			self.called_function = "→no_called_function← This is rediculus now!!!"
		
		# print( sys._getframe(1).f_back.f_code.co_names)
		# print( sys._getframe( self.caller_level ).f_back.f_locals["self"].__class__.__qualname__ )
		# print( sys._getframe( self.caller_level ).f_locals["self"].__class__.__qualname__ )
		time_stamp = datetime.now().strftime("%X.%f")[:-3]
		if( message_type < 0):
			self.prefix = f"{bcolors.FAIL}[{time_stamp}] → Error: {bcolors.ENDC}"
		elif( message_type > 0):
			self.prefix = f"{bcolors.WARNING}[{time_stamp}] → Warning: {bcolors.ENDC}"
		else:
			self.prefix = f"{bcolors.OKGREEN}[{time_stamp}] → Logging: {bcolors.ENDC}"
		if( show_call_stack == True):
			self.call = f"\r\n{self.prefix}\t→ Call path: {self.calling_module}.{self.calling_class}.{self.caller}.{self.called_function}(){bcolors.ENDC}"
		else:
			self.call = ""

		print( self.prefix, message, self.call)

	def __del__(self):
		#pass
		1+1
		#print("→ DEBUG: QtYetiLogger object removed from memory.")

#%% Echelle ectraction
###############################################################################################################################

#### Stackoverflow / Eilers 200x
def speyediff(N, d, format = 'csc'):
	assert not (d < 0), "d must be non negative"
	shape     = (N-d, N)
	diagonals = np.zeros(2*d + 1)
	diagonals[d] = 1.
	for i in range(d):
		diff = diagonals[:-1] - diagonals[1:]
		diagonals = diff
	offsets = np.arange(d+1)
	spmat = scipy.sparse.diags(diagonals, offsets, shape, format=format)
	return spmat

def whittaker_smooth(data_array: np.ndarray, smoothing_stiffness=10.0, smoothing_order=2) -> np.ndarray:
	"""
	The smoothing stiffness sacrifices the fidelity of the fit to the data.
	The smoothing order defines the order at which the roughness is being measured
	This algorithm acts like a Butterworth bandpass filter without a phase shift/lag of the data.
	"""
	m = data_array.shape[0]
	E = scipy.sparse.eye(m, format='csc')
	D = speyediff(m, smoothing_order, format='csc')
	WhittakerEq = E + smoothing_stiffness *  D.conj().T.dot(D)
	z = scipy.sparse.linalg.splu(WhittakerEq).solve(data_array)
	return z

Wweights = np.ones(QT_YETI.DETECTOR_X_PIXELS)
Wweights[700:800] = 0.1
matW=scipy.sparse.spdiags(Wweights,0,QT_YETI.DETECTOR_X_PIXELS,QT_YETI.DETECTOR_X_PIXELS, format = 'csc')
def weighted_whittaker_smooth(y, lmbd, d, matW):
	m = len(y)
	D = speyediff(m, d, format='csc')
	WhittakerEq = matW + lmbd *  D.conj().T.dot(D)
	weighted_y = matW.dot(y)
	z = scipy.sparse.linalg.splu(WhittakerEq).solve(weighted_y)
	return z

"""
Tracing
"""
def echelle_find_orders( CurrentSpectrogram: Spectrogram = None, x_position = 0, TracerSettings: OrderTracerSettings = None):
	"""
	This function finds centers of echelle orders in y↑→x coordinates and not matrix coordinates
	Format [Order, X Position, Y Position]
	These positions are saved in CurrentSpectrogram

	Parameters:
		CurrentSpectrogram (Spectrogram, optional): _description_. Defaults to None.
		x_position (int, optional): _description_. Defaults to 0.
		TracerSettings (OrderTracerSettings, optional): _description_. Defaults to None.
	"""

	if( CurrentSpectrogram == None):
		print(f"Error in {__name__}.find_order_centers(): No SpectrogramObject loaded.")
		QtYetiLogger(-1,"No SpectrogramObject loaded.")
		return -666

	if( TracerSettings == None):
		print(f"Error in {__name__}.find_order_centers(): No TracerSettings provided.")
		QtYetiLogger(-1,"No SpectrogramObject loaded.")
		return -666

	mat_col = col(x_position, CurrentSpectrogram.xsize)
	mat_rows = CurrentSpectrogram.ysize

	# Find-peak and smoothing settings
	PEAK_WIDTH = np.int32(np.rint( TracerSettings.peak_width_px ))
	PEAK_HEIGHT = TracerSettings.peak_height
	PEAK_PROMINENCE = TracerSettings.peak_prominence
	PEAK_DISTANCE = TracerSettings.peak_distance_px
	SPOTSIZE_WIDTH = TracerSettings.spotsize_px
	SMOOTHING_STIFFNESS = TracerSettings.smoothing_stiffness
	SMOOTHING_ORDER = TracerSettings.smoothing_order

	# Systematic approach:
	# Enhance the visibility for darker peaks via np.log10()
	# Use the Whittaker smoothing to preserve the peak structures and reduce the noise
	# Normalize the smoothed spectrum
	# Use percentage values to decide about prominence and minimum height to find peaks

	current_spectrum = np.squeeze(np.abs(CurrentSpectrogram.data[:,mat_col]))
	current_spectrum_log10_sm = whittaker_smooth( np.log10(current_spectrum), SMOOTHING_STIFFNESS, SMOOTHING_ORDER )
	current_min_max_delta = current_spectrum_log10_sm.max() - current_spectrum_log10_sm.min()
	current_spectrum_log10_sm_norm = (current_spectrum_log10_sm - current_spectrum_log10_sm.min())/current_min_max_delta

	found_peak_indices, _ = find_peaks(current_spectrum_log10_sm_norm, distance= PEAK_DISTANCE, height= PEAK_HEIGHT, width= PEAK_WIDTH, prominence= PEAK_PROMINENCE)
	order_centers_list = []

	for peak_index in found_peak_indices:
		order_centers_list.append([mat_col, row(peak_index, mat_rows)])

	# The |physical_order_number| might increase r decrease when going from top to bottom
	# in the image depending on the detector setting. This here helps to have the orders properly sorted from the beginning
	if( QT_YETI.DETECTOR_ABSOLUTE_ORDER_INCREASE_DIRECTION == "up" ):
		order_centers_list.reverse()
	
	# Use a class method so set this list for all instances of Spectrogram
	CurrentSpectrogram.update_order_centers_list(order_centers_list)


# Fitting model function
#def echelle_order_fit_function(x,x0,a_6,a_5,a_4,a_3,a_2,a_1,a_0):
# 	return a_6 * (x-x0)**6+ a_5 * (x-x0)**5 + a_4 * (x-x0)**4 + a_3 * (x-x0)**3 + a_2 * (x-x0)**2 + a_1 * (x-x0) + a_0

def echelle_generate_fit_function(polynomial_order: int = 0, offset_x0 = False):
	"""
	Generate an arbitrary polynomial fit function
	=============================================

	:param polynomial_order: Highest degree of a polynomial function
	:type polynomial_order: int
	:return: Returns a function reference
	:rtype: object

	"""
	prefix = "lambda x,"
	base = f"x"
	if( offset_x0 ):
		prefix = "lambda x,x0,"
		base = f"(x-x0)"
	param = "a_"
	function_string = f""
	for i in range(polynomial_order,-1,-1):
		prefix += f"{param}{i},"
		function_string += f"+{param}{i}*({base}**{i})"

	return eval(f"{prefix}: {function_string}")

echelle_order_fit_function = echelle_generate_fit_function(polynomial_order=2, offset_x0=False)

def echelle_order_archlength_integrand(x,x0,a_6,a_5,a_4,a_3,a_2,a_1):
	# Sqrt(1 + f'(x)²)
	return np.sqrt( 1+ (6*a_6 * (x-x0)**5+ 5*a_5 * (x-x0)**4 + 4*a_4 * (x-x0)**3 + 3*a_3 * (x-x0)**2 + 2*a_2 * (x-x0) + 1*a_1)**2)

def echelle_order_tracer_old(order_list: list, CurrentSpectrogram: Spectrogram, x_start, y_start, direction, CurrentTracerSettings: OrderTracerSettings):

	x_list = []
	y_list = []

	errors_during_tracing = []

	spectrogram = CurrentSpectrogram.data
	sp_rows = CurrentSpectrogram.ysize
	sp_columns = CurrentSpectrogram.xsize

	X_GUESS = x_start
	Y_GUESS = y_start
	DIRECTION = direction

	# Sampling rate
	SAMPLES_PER_ORDER = CurrentTracerSettings.samples_per_order
	DISTANCE_TO_EDGE = CurrentTracerSettings.distance_to_image_edge_px

	# Whittaker Smoothing 
	SMOOTHING_STIFFNESS = CurrentTracerSettings.smoothing_stiffness
	SMOOTHING_ORDER = CurrentTracerSettings.smoothing_order

	COLUMN_SAMPLING_DISTANCE = np.int32( np.rint( sp_columns / SAMPLES_PER_ORDER ))
	STEP_SCALING_FACTOR = CurrentTracerSettings.step_scaling_factor
	PEAK_PROMINENCE = CurrentTracerSettings.peak_prominence
		
	# 	Correct this: the search_interval must be image_slicer_separation_px
	SEARCH_INTERVAL = CurrentTracerSettings.image_slicer_separation_px
	SEARCH_RANGE = np.int32( np.rint( 0.75 * SEARCH_INTERVAL ) )
	PEAK_FIND_DISTANCE = np.int32( np.rint(0.5 * SEARCH_RANGE ) )

	# Tracer loop
	trace_range = 0
	
	# Define positions at which to search for an order center 
	# SWITCH OF COORDINATES TO idx, jdx = row-Y, X
	
	if (DIRECTION > 0):
		trace_range = np.arange(col(X_GUESS, sp_columns), sp_columns-1 - DISTANCE_TO_EDGE, COLUMN_SAMPLING_DISTANCE)
	else:
		trace_range = np.arange(col(X_GUESS, sp_columns)-1, 0+DISTANCE_TO_EDGE, -1 * COLUMN_SAMPLING_DISTANCE)
	
	# Initial index & Initial accumulated error
	previous_idx = row(Y_GUESS, sp_rows)
	guess_next_idx = 0 # The guess is an integrated error (similar to I in a PID controller)

	...
	# Modification
	if( DIRECTION > 0):
		trace_range = np.arange( col(X_GUESS, sp_columns), sp_columns-10,5 )
	else:
		trace_range = np.arange( col(X_GUESS, sp_columns)-1,10,-5 )

	INTERVALWIDTH = 7

	for i, jdx in enumerate(trace_range):
		center_idx = previous_idx + guess_next_idx
		index_range = slice(center_idx-INTERVALWIDTH, center_idx+(INTERVALWIDTH+1), 1)
		
		#spec_of_interest = savgol_filter( np.squeeze(spectrogram[index_range, jdx]),2,1) # Takes ages! but shorter than whittaker
		#spec_of_interest = whittaker_smooth( np.squeeze(spectrogram[index_range, jdx]),SMOOTHING_STIFFNESS,SMOOTHING_ORDER) # Takes ages!
		spec_of_interest = np.squeeze(spectrogram[index_range, jdx])
		if( spec_of_interest.shape[-1] == INTERVALWIDTH):
			print(f"we hit the interval width")
			break
		try:
			found_idx = np.argmax(spec_of_interest)
			previous_idx = previous_idx + (found_idx-INTERVALWIDTH)
			if( (i+1)%4 == 0 ):
				x_list.append(jdx)
				y_list.append(row(previous_idx, sp_rows))
		except ValueError:
			pass

	return x_list, y_list

@elapsed_time
def echelle_order_tracer(CurrentSpectrogram: Spectrogram, CurrentTracerSettings: OrderTracerSettings, downsampling_value: int = 1):
	""" 
	Echelle order position extraction
	---------------------------------
	Description:
		Brute-force order tracer. Starting on any given intensity maximum (x_m,y_m) within an order,
		we progress pixel by pixel in X direction to find the next maximum.

	Parameters:
		CurrentSpectrogram (Spectrogram): Currently active spectrogram that needs to be traced.
		CurrentTracerSettings (OrderTracerSettings): Currently active tracer settings from the TracerSettingWindow class.
		downsampling_value (int): Save only every n-th result. 
		
	Returns:
		numpy.ndarray: The array contains arrays of x and y positions along all orders within a spectrogram.

		Example: [ [[x_coords @ m=1],[y_coords @ m=1]] , [[x_coords @ m=2],[y_coords @ m=2]] , [[],[]] , ...]
	"""
	### Settings ###############################################
	# Choose method to find the order maximum
	finding_fct = np.argmax
	SAVE_EVERY_Nth_ELEMENT = downsampling_value
	PRECISION_MODE = False
	INTENSITY_CUT_OFF_MODE = True
	############################################################

	# Definitions
	SPECTROGRAM = CurrentSpectrogram.data
	SP_ROWS = CurrentSpectrogram.ysize
	SP_COLUMNS = CurrentSpectrogram.xsize

	SPEC_MAX = CurrentSpectrogram.intmax
	SPEC_MIN = CurrentSpectrogram.intmin
	SPEC_DELTA_MINMAX = SPEC_MAX - SPEC_MIN

	order_centers_list = CurrentSpectrogram.order_centers_list

	# Sampling rate
	SAMPLES_PER_ORDER = CurrentTracerSettings.samples_per_order
	DISTANCE_TO_EDGE = CurrentTracerSettings.distance_to_image_edge_px

	# Whittaker Smoothing 
	SMOOTHING_STIFFNESS = CurrentTracerSettings.smoothing_stiffness
	SMOOTHING_ORDER = CurrentTracerSettings.smoothing_order

	COLUMN_SAMPLING_DISTANCE = np.int32( np.rint( SP_COLUMNS / SAMPLES_PER_ORDER ))
	PEAK_PROMINENCE = CurrentTracerSettings.peak_prominence
		
	SEARCH_INTERVAL = CurrentTracerSettings.spotsize_px
	SEARCH_RANGE = np.int32( np.rint( 0.25 * SEARCH_INTERVAL ) )
	#PEAK_FIND_DISTANCE = ...

	################################################
	# SWITCH OF COORDINATES TO idx, jdx = row-Y, X #
	################################################

	left_x_list = []
	left_y_list = []
	right_x_list = []
	right_y_list = []

	errors_during_tracing = []

	# Auxiliary functions
	def _quadratic_fit_function(x,x0,a_2,a_1,a_0):
		return a_2*(x-x0)**2 + a_1*(x-x0) + a_0

	order_fit_function = _quadratic_fit_function
	def _fitted_order_maximum(spectrum: np.ndarray) -> int:
		"""
		Description:
			Use any fit function to determin the intensity maximum in y direction at the current x position

		Parameters:
			spectrum (np.ndarray): Current spectrum of interest

		Returns:
			int: Index of the maximum in relative coordinates
		"""		
		fit_range = np.arange(-SEARCH_RANGE, SEARCH_RANGE+1, 1)
		middle_index = int(np.rint( (fit_range[-1] -1) /2))

		max_intensity = spectrum.max()
		normalized_spectrum = spectrum/max_intensity # Normalize for smaller fit coefficients

		fit_output, _ = scipy.optimize.curve_fit(order_fit_function, fit_range, normalized_spectrum)
		result = np.argmax(_quadratic_fit_function(fit_range, *fit_output)) - middle_index

		# plt.figure()
		# plt.plot(fit_range, normalized_spectrum,'.')
		# plt.plot(fit_range, quadratic_fit_function(fit_range, *fit_output))
		# plt.show(block=True)

		return result

	if (PRECISION_MODE):
		finding_fct = _fitted_order_maximum

	def _find_order_intensity_maxima(trace_range: range, initial_y: float) -> tuple[list,list]:
		"""
		Description:
			...
		Parameters:
			trace_range (range): Range of X pixels to be considered
			initial_y (float): Initial Y pixel to start the tracing at

		Returns:
			tuple[list,list]: Return a tuple of X and Y coordinate list
		"""

		x_list = []
		y_list = []

		# Initial index & Initial accumulated error
		previous_idx = row(initial_y, SP_ROWS)
		# The guess is an integrated error (similar to I in a PID controller)
		guess_next_idx = 0
		for i,jdx in enumerate(trace_range):

			center_idx = previous_idx + guess_next_idx
			index_range = slice(center_idx - SEARCH_RANGE, center_idx + SEARCH_RANGE + 1, 1)
			spec_of_interest = np.squeeze(SPECTROGRAM[index_range, jdx])
			norm_spec_of_interest = (spec_of_interest - SPEC_MIN)/SPEC_DELTA_MINMAX

			# if( signal_to_noise == crap):
			# 	break

			# Check whether we've hit the boundary of the spectrogram
			if( spec_of_interest.shape[-1] == SEARCH_RANGE):
				break
			
			if( INTENSITY_CUT_OFF_MODE ):
				# Check if there is only noise. Be better than any chosen intensity
				if( norm_spec_of_interest.max() < PEAK_PROMINENCE):
					break

			try:
				# Here is where the magic happens. found_idx = 0 .. 2*SEARCH_RANGE+1
				found_idx = finding_fct(spec_of_interest)

				# Prepare for next loop iterationa and correct for the Search Range
				previous_idx = previous_idx + (found_idx-SEARCH_RANGE)

				if( (i+1)% SAVE_EVERY_Nth_ELEMENT == 0):
					x_list.append(jdx)
					y_list.append(row(previous_idx, SP_ROWS))

			except ValueError as Error:
				QtYetiLogger(QT_YETI.ERROR, f"Error in Tracer: {Error}.", True)
				pass

		return x_list, y_list

	# Tracer loop
	result_list = []

	initial_x, _ = order_centers_list[0]
	left_trace_range  = range(col(initial_x, SP_COLUMNS), 0+DISTANCE_TO_EDGE, -1 * COLUMN_SAMPLING_DISTANCE)
	right_trace_range = range(col(initial_x, SP_COLUMNS)+1 , SP_COLUMNS-1 - DISTANCE_TO_EDGE, COLUMN_SAMPLING_DISTANCE)

	for _, initial_y in order_centers_list: # m, x, y in ...

		left_x_list, left_y_list = _find_order_intensity_maxima(left_trace_range, initial_y)
		right_x_list, right_y_list = _find_order_intensity_maxima(right_trace_range, initial_y)

		# left_x_list = [left_trace_range] But every Nth value not considered.
		# right_x_list= [right_trace_range]

		left_x_list.reverse()
		left_y_list.reverse()

		result_list.append( [left_x_list + right_x_list , left_y_list + right_y_list] )

	# KKTHX
	return result_list

		#fit_range = np.arange(-SEARCH_RANGE, SEARCH_RANGE+1, 1)

		# OLD STUFF
		# # Initial index & Initial accumulated error
		# previous_idx = row(y_guess, SP_ROWS)
		# guess_next_idx = 0 # The guess is an integrated error (similar to I in a PID controller)
		# for i,jdx in enumerate(left_trace_range):

		# 	center_idx = previous_idx + guess_next_idx
		# 	index_range = slice(center_idx - SEARCH_RANGE, center_idx + SEARCH_RANGE + 1, 1)
		# 	spec_of_interest = np.squeeze(SPECTROGRAM[index_range, jdx])

		# 	# if( signal_to_noise == crap):
		# 	# 	break

		# 	try:
		# 		# Here is where the magic happens. found_idx = 0 .. 2*SEARCH_RANGE+1
		# 		found_idx = np.argmax(spec_of_interest)
		# 		#found_idx = center_idx + fitted_order_maximum(fit_range, spec_of_interest, quadratic_fit_function)

		# 		# Prepare for next loop iterationa and correct for the Search Range
		# 		previous_idx = previous_idx + (found_idx-SEARCH_RANGE)

		# 		if( (i+1)% SAVE_EVERY_Nth_ELEMENT == 0):
		# 			left_x_list.append(jdx)
		# 			left_y_list.append(row(previous_idx, SP_ROWS))

		# 	except ValueError as Error:
		# 		QtYetiLogger(QT_YETI.ERROR, "Error in Tracer.", True)
		# 		pass

		# 	# Check whether we've hit the boundary of the spectrogram
		# 	if( spec_of_interest.shape[-1] == SEARCH_RANGE):
		# 		break

		# previous_idx = row(y_guess, SP_ROWS)
		# guess_next_idx = 0 # The guess is an integrated error (similar to I in a PID controller)
		# for i,jdx in enumerate(right_trace_range):

		# 	center_idx = previous_idx + guess_next_idx
		# 	index_range = slice(center_idx - SEARCH_RANGE, center_idx + SEARCH_RANGE + 1, 1)
		# 	spec_of_interest = np.squeeze(SPECTROGRAM[index_range, jdx])

		# 	# if( signal_to_noise == crap):
		# 	# 	break

		# 	try:
		# 		# Here is where the magic happens. found_idx = 0 .. 2*SEARCH_RANGE+1
		# 		found_idx = np.argmax(spec_of_interest)
		# 		#found_idx = center_idx + fitted_order_maximum(fit_range, spec_of_interest, quadratic_fit_function)

		# 		# Prepare for next loop iterationa and correct for the Search Range
		# 		previous_idx = previous_idx + (found_idx-SEARCH_RANGE)

		# 		if( (i+1)% SAVE_EVERY_Nth_ELEMENT == 0):
		# 			right_x_list.append(jdx)
		# 			right_y_list.append(row(previous_idx, SP_ROWS))

		# 	except ValueError as Error:
		# 		QtYetiLogger(QT_YETI.ERROR, "Error in Tracer.", True)
		# 		pass

		# 	# Check whether we've hit the boundary of the spectrogram
		# 	if( spec_of_interest.shape[-1] == SEARCH_RANGE):
		# 		break

	#return (left_x_list+right_x_list),(left_y_list+right_y_list)
	
	x_list = []
	y_list = []
	previous_idx = row(Y_GUESS, SP_ROWS)
	guess_next_idx = 0
	for i, jdx in enumerate(trace_range):
		center_idx = previous_idx + guess_next_idx
		index_range = slice(center_idx-SEARCH_RANGE, center_idx+(SEARCH_RANGE+1), 1)
		
		#spec_of_interest = savgol_filter( np.squeeze(spectrogram[index_range, jdx]),2,1) # Takes ages! but shorter than whittaker
		#spec_of_interest = whittaker_smooth( np.squeeze(spectrogram[index_range, jdx]),SMOOTHING_STIFFNESS,SMOOTHING_ORDER) # Takes ages!
		spec_of_interest = np.squeeze(SPECTROGRAM[index_range, jdx])
		if( spec_of_interest.shape[-1] == SEARCH_RANGE):
			print(f"we hit the interval width")
			break
		try:
			found_idx = np.argmax(spec_of_interest)
			previous_idx = previous_idx + (found_idx-SEARCH_RANGE)
			if( (i+1)%4 == 0 ):
				x_list.append(jdx)
				y_list.append(row(previous_idx, SP_ROWS))
		except ValueError as Error:
			# errors_during_tracing.append(Error.args)
			pass

	return [x_list], y_list

	loop_run = 1
	for jdx in trace_range:
		
		# On the first run, estimated_order_center_idx = row(y_guess) + error
		estimated_order_center_idx = previous_idx + guess_next_idx

		# If something goes out of bounds
		if (estimated_order_center_idx > SP_ROWS - SEARCH_RANGE):
			break
		elif (estimated_order_center_idx < SEARCH_RANGE):
			break

		# Prepare a small slice along the rows of a spectrogram at position jdx = col(x_guess, sp_columns)
		relative_index_range = slice( estimated_order_center_idx - SEARCH_RANGE, (estimated_order_center_idx + 1) + SEARCH_RANGE , 1)
		spec_of_interest = SPECTROGRAM[relative_index_range, jdx]

		try:
			spec_of_interest_min = spec_of_interest.min()
		except:
			print(SPECTROGRAM.shape, relative_index_range, SPECTROGRAM[relative_index_range, jdx])
		try:
			spec_of_interest_max = spec_of_interest.max()
		except:
			print(SPECTROGRAM.shape, relative_index_range, SPECTROGRAM[relative_index_range, jdx])

		# To have better log-plots:
		# Change the value of the global minimum in the spectrogram to the value of it's neighbour,
		# such that subtracting the min-value does not cause "division by zero" errors
		spec_of_interest_min_index =  np.argmin( spec_of_interest - spec_of_interest_min )
		try:
			spec_of_interest[ spec_of_interest_min_index ] = spec_of_interest[ spec_of_interest_min_index + 1]
		except:
			spec_of_interest[ spec_of_interest_min_index ] = spec_of_interest[ spec_of_interest_min_index - 1]

		""" Smoothing / Filtering """
		# Use a Whittaker or Savatzky Golay Filter
		filtered_spec_of_interest = whittaker_smooth( spec_of_interest, SMOOTHING_STIFFNESS, SMOOTHING_ORDER)
		filtered_spec_of_interest_min = filtered_spec_of_interest.min()
		filtered_spec_of_interest_max = filtered_spec_of_interest.max()
			#filtered_spec_of_interest = savgol_filter(spec_of_interest, window_length = SAVGOL_FILTER_WINDOW, polyorder = SAVGOL_FILTER_POLYORDER)

		# Subtract an offset close to the baseline. 
		spec_of_interest			= spec_of_interest			- filtered_spec_of_interest_min
		filtered_spec_of_interest	= filtered_spec_of_interest	- filtered_spec_of_interest_min
			#spec_of_interest = np.log10( spec_of_interest / ( spec_of_interest.max() - spec_of_interest.min()) )
			# The filtered cross section is practically offset corrected already. No need for subtraction
			#numerator = (filtered_spec_of_interest - filtered_spec_of_interest_min + 1e-6 )
			#numerator = (filtered_spec_of_interest - filtered_spec_of_interest_min)
			#denominator = (filtered_spec_of_interest_max - filtered_spec_of_interest_min)


		""" ###################################################################################################################
		FIND PEAKS
		
		# Peak indices are given in relation to 'spec_of_interest', i.e. relative_index_range = [0, 1, ..., 2*SEARCH_RANGE+1] 
		# 
		# spec_of_interest   [----/\---↓---------] = • ± SEARCH_RANGE => the index 'SEARCH_RANGE' is the middle
		# spectrum      [---------/\---•----------------] where • = estimated_order_center_idx and /\ = peak
		#
		"""
		found_peaks,_ = find_peaks( filtered_spec_of_interest, prominence = PEAK_PROMINENCE, distance = PEAK_FIND_DISTANCE)
			#found_peaks,_ = find_peaks( savgol_filter( -1*spec_of_interest + 0*np.abs(np.max(-1*spec_of_interest)), window_length=SAVGOL_FILTER_WINDOW, polyorder = SAVGOL_FILTER_POLYORDER), prominence=40)
			#found_peaks = find_peaks_cwt( savgol_filter( spec_of_interest, window_length=SAVGOL_FILTER_WINDOW, polyorder = SAVGOL_FILTER_POLYORDER), widths=8 )
			#found_peaks,_ = find_peaks(np.log10(filtered_spec_of_interest), prominence = PEAK_PROMINENCE, distance = PEAK_FIND_DISTANCE)
			#found_peaks,_ = find_peaks(np.log10(filtered_spec_of_interest), prominence = PEAK_PROMINENCE, distance = PEAK_FIND_DISTANCE)

		if( len(found_peaks) == 0 ):
			errors_during_tracing.append(f"Error: echelle_order_tracer(): No peaks detected in (loop_run, jdx, y_guess): {loop_run, jdx, Y_GUESS}. Exitting")
		try:
			pass
			#filtered_spec_of_interest = np.log10( numerator / denominator ) # log(a/b) = log a - log b
		except:
			errors_during_tracing.append(f"jdx: {jdx, filtered_spec_of_interest_min, filtered_spec_of_interest_max}")


		""" ###################################################################################################################
		Find the closest peak index relative to SEARCH_RANGE, which is the middle index of the "spectrum_of_interest".
		See description/comment above.
		"""
		# First, find the peak number within the result array "found_peaks" that is closest to the middle index
		try:
			closest_peak_number = np.argmin( np.abs( found_peaks - SEARCH_RANGE ))
		except:
			errors_during_tracing.append(f"Error: closest_peak_number = np.NaN.")
			closest_peak_number = np.NaN
		try:
			closest_pk_index = found_peaks[closest_peak_number]
			closest_pk_height = filtered_spec_of_interest[ closest_pk_index ]
		except:
			errors_during_tracing.append(f"Failed to find closest peak. Using middle of interval. Error {jdx, Y_GUESS, jdx , Y_GUESS} ")
			closest_pk_index = SEARCH_RANGE


		""" ###################################################################################################################
		Order center depending on spectrometer configuration
		If an "Image Slicer" was used, the order center is between the two peaks of an order in the spectrogram
		If no Image Slicer is used, the order center is exactly the center of the intensity peak		
		"""
		order_center_index = 0
		if( CurrentTracerSettings.image_slicer is False):
			# Return the closest found peak which is the order center
			order_center_index = closest_pk_index
			pass
		else:
			# Decide and return the index between two intensity peaks of an order which is the order center
			## Idea: always go to nearest peak relative to the middle of the search interval
			minpoint_index = 0		
			try:
				slice_between_peaks = slice( closest_pk_index, closest_pk_index + CurrentTracerSettings.image_slicer_separation_px + 1, 1)
				minpoint_index = closest_pk_index + np.argmin( spec_of_interest[slice_between_peaks] )
				#slice_between_peaks = slice( found_peaks[0], found_peaks[0] + Settings.image_slicer_separation_px + 1, 1)
				#minpoint_index = found_peaks[0] + np.argmin( spec_of_interest[slice_between_peaks] )
				#minpoint_index = closest_pk_index

				order_center_index = minpoint_index
			except:
				pass

		""" ###################################################################################################################
		Finalize
		"""
		# Calculate the delta between the middle of the spec_of_interest
		idx_delta = (order_center_index - SEARCH_RANGE)

		# Add the calculated difference to the estimated_order_center_index
		found_idx = estimated_order_center_idx + idx_delta
		previous_idx = found_idx

		# integrated error
		# Depending on the sampling_distance

		#	#	# Some bloack magic needed
		guess_next_idx = guess_next_idx + np.int32( STEP_SCALING_FACTOR * idx_delta )
		
		### Prepare result
		x_list.append(jdx)
		y_list.append( row(found_idx, SP_ROWS) )
		loop_run += 1

	return x_list, y_list

# Data extraction
def echelle_order_spectrum_extraction(SpectrogramObject: Spectrogram, order: int, Settings: OrderTracerSettings):
	"""
	Extraction of spectra off of Echelle Orders with and without an image slicer
	============================================================================

	To extract the intensities of an order, we use the current spectrogram "D" and multiply it with
	masking matrices "M" or a masking matrix, depending on if an image-slicer is used.
	
	Every masking matrix consists of ones and zeros which are then entrywise multiplied with the data.
	To save computational time, we define a region-of-interest (ROI) around the order-center fitting polynomial
	for a given order.
	
	With an Imageslicer:
	The entries of the masking matrix will be set to 1 at defined positions above and below the order center,
	depending on the fiber spot size and the image-slicer-separation value in pixels.
	The image slicer separation is defined by the distance between the two intensity peaks of an order.

	Without an Imageslicer:
	The entries of the masking matrix will be set to 1 at defined positions aboive and below the order center,
	depending on the fiber spot size in pixels.

	F = M * D yields a matrix with selected values out of the Spectrogram Data Matrix D.
	F.sum(axis=0) yields a 1D array with the sum of all entries along all rows.

	"""
	# All calculations done in (row, column) coordinates.
	# Polynomial is saved in (x,y) coordinates
	pass

	total_cols = SpectrogramObject.xsize
	col_range = col(SpectrogramObject.xrange, total_cols)
	
	# Spot size and margin
	spot_summation_width = np.uint32(np.floor(Settings.spotsize_px/2))
	spot_summation_margin = 1

	# We need the half of the image slicer separation, since we calculate from the order center
	image_slicer_separation_half = 0
	if(Settings.image_slicer is True):
		image_slicer_separation_half = np.uint32( np.floor(Settings.image_slicer_separation_px/2))

	# Get order coefficients for given order
	order_coefficients = SpectrogramObject.order_fit_coefficients[order][1:]
	order_center_polynomial = echelle_order_fit_function(col_range, *order_coefficients)
	discretized_polynomial=np.rint(order_center_polynomial)
	
	# ROI preparation. Get min max of the polynomial first, define ROI afterwards
	discretized_poly_rows = row(discretized_polynomial, SpectrogramObject.ysize)
	poly_min_idx = discretized_poly_rows.min()
	poly_max_idx = discretized_poly_rows.max()

	# Region of interest
	roi_start_idx = poly_min_idx - (image_slicer_separation_half + spot_summation_width + spot_summation_margin + 1)
	roi_stop_idx  = poly_max_idx + (image_slicer_separation_half + spot_summation_width + spot_summation_margin + 1)
	roi_row_length = np.uint32(roi_stop_idx - roi_start_idx + 1)

	"""
	Create masking matrices

	The overall goal is to create tuples of row and column indices (r_0, c_0), (r_1, c_1), ...
	at which the masking matrices are going to be 1. The rest needs to be zero.

	We will use the polynomial that locates the order center. Depending on if an image slicer is used
	we need to account for sub-orders that are separated by image_slicer_separation.
	"""
	# Reduced dataset to ROI = region of interest
	order_masking_matrix = np.zeros((roi_row_length, total_cols))

	### Generate ROI from data matrix
	# Add check if out of bounds etc.
	roi_slice = slice(roi_start_idx, roi_stop_idx+1, 1)
	relevant_data_matrix = SpectrogramObject.data[roi_slice,:]

	# Shift polynomial fit relative to the ROI
	discretized_poly_rows_in_roi = discretized_poly_rows - roi_start_idx

	"""
	Depending on if there is an image-slicer present or not, we want either to sum up the intensities
	of the sub-orders shifted from the polynomial (=order center) by ±image_slicer_separation_half,
	or the single main order that lies on the polynomial (=order center).
	The summation takes place symmetrically
	between -(spot_summation_width+spot_summation_margin) to +(spot_summation_width+spot_summation_margin)
	"""
	row_summation_indices = np.arange( image_slicer_separation_half - (spot_summation_width+spot_summation_margin), image_slicer_separation_half + (spot_summation_width+spot_summation_margin) + 1)
	row_summation_width = row_summation_indices.shape[0]

	"""
	Tile col_range a (summation_width)-times
	"""
	# Create matrix indices at which to set the masking matrix to 1
	# Tile the col_range [0,1,2...,n-1] k × times together to form an 1D-Array that is (k × n) long
	# → mask_cols = [0,1,2...,n-1, 0,1,2...,n-1, 0,1,2...,n-1, 0,1,2...,n-1 ... ]
	#
	# Equivaltenly: mask_cols = np.asarray(summation_width * [col_range]).ravel()
	mask_cols = np.tile(col_range, row_summation_width)

	# Tile the polynomial a (...)-times
	mask_rows = np.tile(discretized_poly_rows_in_roi, row_summation_width)

	"""
	Important step: here, we generate an array "mask_row_offsets", that will consist of a series of offset indices that will
	be added to the discretized polynomial. If using an image-slicer, we need to go from the order center
	to the middle of the upper main order or lower sub-order. We will then 
	"""
	## numpy.repeat() returns an ordered array from low to high values
	# This array shifts the polynomials to the suborders when an image-slicer is present
	# Else, it contains the indices at which the spot has to be summed up in the end
	mask_row_offsets = row_summation_indices.repeat(total_cols)

	order_mask_rows = mask_rows + mask_row_offsets
	order_masking_matrix[order_mask_rows, mask_cols]=1
	masked_lower_order = order_masking_matrix * relevant_data_matrix
	order_intensity = masked_lower_order.sum(axis=0)

	# baseline_masking_matrix[discretized_poly_rows_in_roi, col_range]=1
	# masked_baseline    = relevant_data_matrix * baseline_masking_matrix
	# background_intensity = np.mean( (spot_summation_width + spot_summation_margin) * masked_baseline.sum(axis=0))

	# Repeat calculation for second suborder in case of an image-slicer
	suborder_masking_matrix = 0
	suborder_mask_rows = 0
	masked_suborder = 0
	suborder_intensity = 0
	if( Settings.image_slicer ):
		suborder_masking_matrix = np.zeros((roi_row_length, total_cols))
		# upper gets minus-sign because we are in ROW,COL coordinates here and the spectrogram is in (x,y)
		# so we need to reduce the index in order to hit it
		suborder_mask_rows = mask_rows - mask_row_offsets
		suborder_masking_matrix[suborder_mask_rows, mask_cols]=1
		masked_suborder = relevant_data_matrix * suborder_masking_matrix
		suborder_intensity = masked_suborder.sum(axis=0)

	return (order_intensity, suborder_intensity),(row_summation_indices, row_summation_width)

#%% Geometrical optics
###############################################################################################################################

class wavelengths:
	"""
	Lowest and highest wavelength detectable
	"""
	λ_low = 350 #nm
	λ_high = 1100 #nm

class unit_vectors:
	"""
	Define laboratory coordinate system
	"""
	x = np.array([1,0,0]) # inplane axis perpendicular to optical axis
	y = np.array([0,1,0]) # out-of-plane direction
	z = np.array([0,0,1]) # optical axis towards prism

def Rotation_x(ψ) -> np.ndarray:
	"""
	Rotation matrix along Lab-x-axis
	"""
	# return np.asarray([ [1,0,0], [0,np.cos(ψ),-np.sin(ψ)], [0,np.sin(ψ),np.cos(ψ)] ])
	return R.from_rotvec(ψ * unit_vectors.x).as_matrix()

def Rotation_y(ψ) -> np.ndarray:
	"""
	Rotation matrix along Lab-y-axis
	"""
	# return np.asarray([ [np.cos(ψ),0,-np.sin(ψ)], [0,1,0], [np.sin(ψ),0,np.cos(ψ)] ])
	return R.from_rotvec(ψ * unit_vectors.y).as_matrix()

def Rotation_z(ψ) -> np.ndarray:
	"""
	Rotation matrix along Lab-z-axis
	"""
	# return np.asarray([ [np.cos(ψ),-np.sin(ψ),0], [np.sin(ψ),np.cos(ψ),0], [0,0,1] ])
	return R.from_rotvec(ψ * unit_vectors.y).as_matrix()

def Rotation_v(ψ: float, rotation_vector: np.ndarray) -> np.ndarray:
	"""
	Rotation matrix along an arbitrary vector v and angle ψ
	"""
	norm_vector = (rotation_vector)/np.linalg.norm(rotation_vector)
	return R.from_rotvec(ψ * norm_vector).as_matrix()

def beam_direction(φ: float, θ: float) -> np.ndarray:
	"""
	The optical axis direction in the laboratory coordinate system parallel to z.
	x lies in-plane
	y is out-of-plane and perpendicular to x and z
	Thus: k = [sin φ • cos θ , sin θ , cos φ • cos θ]
	A cyclic permutation of the standard spherical coordinates
	"""
	return np.array([np.sin(φ) * np.cos(θ), np.sin(θ), np.cos(φ) * np.cos(θ)])

def dr_dθ(φ: float, θ: float) -> np.ndarray:
	return np.array([-np.sin(φ) * np.sin(θ), np.cos(θ), -np.cos(φ) * np.sin(θ)])

def dr_dφ(φ: float, θ: float) -> np.ndarray:
	return np.array([np.cos(φ) * np.cos(θ), np.sin(θ), -np.sin(φ) * np.cos(θ)])

def n_air(λ):
	"""
	Wavelength dependent refractive index of air
	"""
	return 1.000273

def sellmeier_n_f2(λ):
	"""
	Return a wavelength dependend refractive index for F2 via SCHOTTs spec table
	"""
	# n² = 1 + Sum_i B_i * lambda² / (lambda² - C_i)
	# n2−1 = 1.39757037 λ2 / λ2−0.00995906143 + 0.159201403 λ2 / λ2−0.0546931752 + 1.2686543 λ2 / λ2−119.248346
	# https://refractiveindex.info/?shelf=glass&book=SCHOTT-F&page=F2
	
	# B_coeffs = np.array([1.39757037, 0.159201403, 1.2686543])
	# C_coeffs = np.array([0.00995906143, 0.0546931752, 119.248346])
	# n_sq = 1+ ((B_coeffs * λ**2) / (λ**2 - C_coeffs)).sum()

	# return np.sqrt(n_sq)
	λ = λ*1e-3 

	B1 = 1.34533359
	B2 = 0.209073176
	B3 = 0.937357162
	C1 = 0.00997743871
	C2 = 0.0470450767
	C3 = 111.8867640

	return np.sqrt(1+ \
		((B1 * λ**2) / (λ**2 - C1)) + \
		((B2 * λ**2) / (λ**2 - C2)) + \
		((B3 * λ**2) / (λ**2 - C3)))

def sellmeier_n_nf2(λ):
	"""
	Return a wavelength dependend refractive index for N-F2 via SCHOTTs spec table
	"""
	# n² = 1 + Sum_i B_i * lambda² / (lambda² - C_i)
	# n2−1 = 1.39757037 λ2 / λ2−0.00995906143 + 0.159201403 λ2 / λ2−0.0546931752 + 1.2686543 λ2 / λ2−119.248346
	# https://refractiveindex.info/?shelf=glass&book=SCHOTT-F&page=N-F2
	
	# B_coeffs = np.array([1.39757037, 0.159201403, 1.2686543])
	# C_coeffs = np.array([0.00995906143, 0.0546931752, 119.248346])
	# n_sq = 1+ ((B_coeffs * λ**2) / (λ**2 - C_coeffs)).sum()

	# return np.sqrt(n_sq)
	λ = λ*1e-3 

	B1 = 1.39757037
	B2 = 0.159201403
	B3 = 1.2686543
	C1 = 0.00995906143
	C2 = 0.0546931752
	C3 = 119.248346

	return np.sqrt(1+ \
		((B1 * λ**2) / (λ**2 - C1)) + \
		((B2 * λ**2) / (λ**2 - C2)) + \
		((B3 * λ**2) / (λ**2 - C3)))

def physical_orders_m(β, d, α, γ) -> np.ndarray:
	"""
	Calculate physically possible orders for a given geometry
	=========================================================

	:param β: Highest degree of a polynomial function
	:type β: float

	:return: Returns the lowest and highes order
	:rtype: list

	"""
	# detectable wavelength range 350 to 1100nm
	sign_m = 0

	m_low = (d * np.cos(γ)/wavelengths.λ_high)*(np.sin(α) + np.sin(β)) 
	m_high =(d * np.cos(γ)/wavelengths.λ_low)*(np.sin(α) + np.sin(β)) 

	if( m_low < 0):
		sign_m = -1
	else:
		sign_m = 1

	m_low = np.ceil( np.abs(m_low ))
	m_high= np.floor(np.abs(m_high))

	return np.asarray([m_low, m_high, sign_m], dtype=np.int32)

def grating_β(m, λ, d, α, γ) -> np.ndarray:
	"""
	Return the outgoing angle of a light ray for a given order m with a given wavelength and incident angles
	"""
	return np.arcsin( ((m * λ)/(d * np.cos(γ))) - np.sin(α) )

def refraction_on_surface(normalized_beam_vector: np.ndarray, surface_normal: np.ndarray, refractive_index_1, refractive_index_2) -> np.ndarray:
	"""
	Snell's law in vector form
	--------------------------
	"""
	k_i = np.asarray(normalized_beam_vector)
	s_n = np.asarray(surface_normal)
	n1 = np.asarray(refractive_index_1,dtype=np.float64)
	n2 = np.asarray(refractive_index_2,dtype=np.float64)
	#return_value =  (np.sqrt( 1 - ((n1/n2)**2) * (1 - np.dot(s_n,k_i)**2) ) - (n1/n2) * np.dot(s_n,k_i))*s_n + (n1/n2)*k_i
	radicand = 1 - ((n1/n2)**2) * (1 - np.dot(s_n,k_i)**2)
	if (radicand >= 0):
		return_value =  (np.sqrt( radicand ) - (n1/n2) * np.dot(s_n,k_i))*s_n + (n1/n2)*k_i
	else:
		return_value = np.asarray([np.NaN, np.NaN, np.NaN])

	return return_value

@dataclass(init=True, repr=True)
class Ray:
	λ: float
	order: int
	direction: np.ndarray
	intensity: float = 1.0

class Point(Spot):
	pass

@dataclass(init=True, repr=True)
class optical_element:
	outer_refractive_index_function: object
	inner_refractive_index_function: object
	entry_surface_normal: np.ndarray
	exit_surface_normal: np.ndarray

@dataclass(init=False, repr=True)
class prism( optical_element ):
	def __init__(self, n_outer: object, n_prism: object, apex_up: bool = False, apex_angle_α:float = 60.0/in_rad, base_tilt_angle:float = -24.7/in_rad):

		self.apex_up = apex_up
		self.apex_angle_α = apex_angle_α

		self.outer_refractive_index_function = n_outer
		self.inner_refractive_index_function =  n_prism

		#Create prism facet normal vectors
		rotation_direction = -1
		if( self.apex_up is True ):
			rotation_direction = 1
		self.prism_entry_facet_tilt = rotation_direction * apex_angle_α/2 + base_tilt_angle
		self.entry_surface_normal = np.asarray(Rotation_x( self.prism_entry_facet_tilt).dot(unit_vectors.z),dtype=np.float64)
		self.exit_surface_normal =  np.asarray(Rotation_x( -rotation_direction * apex_angle_α ).dot(self.entry_surface_normal),dtype=np.float64)

@dataclass
class detector:
	x_pixels: int
	y_pixels: int
	focal_length: float
	pixel_size: float
	x_pixel_binning: int
	y_pixel_binning: int
	φ: float
	θ: float
	
	def __post_init__(self):
		self.field_of_view_x = 2.0*np.arctan(0.5 * self.x_pixels * self.pixel_size / self.focal_length)
		self.field_of_view_y = 2.0*np.arctan(0.5 * self.y_pixels * self.pixel_size / self.focal_length)
		
		# Lens unit vectors
		self.lens_optical_axis_vector = Rotation_x(self.φ).dot(unit_vectors.z)
		self.Lx = unit_vectors.x
		self.Ly = np.cross(self.lens_optical_axis_vector, self.Lx)

		# Rotate detector coordinate system around the optical axis of the lens by an angle θ
		R_Lz = Rotation_v(ψ= self.θ, rotation_vector= self.lens_optical_axis_vector)
		self.Lx = R_Lz.dot(self.Lx)
		self.Ly = R_Lz.dot(self.Ly)

		QtYetiLogger(QT_YETI.ERROR,f"ToDo: implement increment rotation and / or rotation and tilt from scrätch", True )
		QtYetiLogger(QT_YETI.ERROR,f"ToDo: put in here the ray selection and FOV considerations", True )
		QtYetiLogger(QT_YETI.ERROR,f"ToDo: Use local Spherical coordinates and derivatives dr_dθ and dr_dφ", True )


	def set_rotation(self,θ):
		R_Lz = Rotation_v(ψ= θ, rotation_vector= self.lens_optical_axis_vector)
		self.Lx = R_Lz.dot(self.Lx)
		self.Ly = R_Lz.dot(self.Ly)

#def create_diffracted_beams(λ: np.ndarray, optical_β: float, d: int, α: float, γ: float) -> np.ndarray:
def create_diffracted_beams(λ: np.ndarray, intensities: np.ndarray, optical_β: float, d: int, α: float, γ: float, camera: detector) -> np.ndarray:
	start_time = time.perf_counter()

	"""
	:param optical_β: Angle from grating normal to optical axis. β = -Γ
	:param d: Grating constant in units of nanometers (1e6 / grooves/mm)[nm]
	:param α: In-plane angle α measured from the grating normal to the incident beam. α = -δ -Γ
	:param γ: Out-of-plane angle γ measured from the grating normal to the incident beam. The outgoing beam is -γ
	"""
	
	γ_out = -γ

	beams = []
	m_low, m_high, sign_m = physical_orders_m(optical_β, d, α, γ)

	# Loop over orders
	for order_m in np.arange(m_low * sign_m, m_high * sign_m + sign_m, sign_m):
		#for order_m in [-44]:

		β_relative_array = grating_β(order_m, λ, d, α, γ) - optical_β
		λ_array = λ

		# Filter NaNs
		filter_nan_condition = ~np.isnan(β_relative_array)
		β_relative_array = β_relative_array[filter_nan_condition]
		λ_array = λ_array[filter_nan_condition]
		intensity_array = intensities[filter_nan_condition]

		# Filter out of view values
		filter_fov_condition = abs(β_relative_array) <= (0.5 * camera.field_of_view_x)
		β_relative_array = β_relative_array[filter_fov_condition]
		λ_array = λ_array[filter_fov_condition]
		intensity_array = intensity_array[filter_fov_condition]
		

		for λ_out, β_out, intensity in zip(λ_array, β_relative_array, intensity_array):
			beams.append(Ray(λ=λ_out, order=int(order_m), direction=beam_direction(φ=β_out, θ=γ_out), intensity=intensity))

	stop_time = time.perf_counter()
	print(f"Time for diffracted beams {stop_time - start_time}")
	
	return np.asarray(beams)
	
def pass_through_optical_element(incident_beam: Ray, element: optical_element) -> Ray:
	"""
	Use Snells Law in vector form and calculate the wavelength dependent beam direction in 3D
	"""
	k_i = incident_beam.direction

	# Use Snells Law in vector form and calculate the wavelength dependent beam direction in 3D
	k_in = k_i/np.linalg.norm(k_i)
	k_medium = refraction_on_surface(\
		k_in,\
		element.entry_surface_normal,\
		element.outer_refractive_index_function(incident_beam.λ),\
		element.inner_refractive_index_function(incident_beam.λ)\
	)
	k_out = refraction_on_surface(\
		k_medium,\
		element.exit_surface_normal,\
		element.inner_refractive_index_function(incident_beam.λ),\
		element.outer_refractive_index_function(incident_beam.λ)\
	)
	k_on = k_out/np.linalg.norm(k_out)
	
	return Ray(λ = incident_beam.λ, order= incident_beam.order, direction = k_on, intensity= incident_beam.intensity)

def project_onto_detector(incident_beam: Ray, camera: detector) -> Point:

	k_in = incident_beam.direction

	mirror_x = 1
	mirror_y = -1

	# Use trig identities rather than angles to calculate the projection angles (see below)
	sin_α = np.dot(k_in, camera.Lx)
	sin_γ = np.dot(k_in, camera.Ly)

	# tan(arcsin(x)) = x/sqrt(1-x²)
	tan_α = sin_α / np.sqrt(1-sin_α**2)
	tan_γ = sin_γ / np.sqrt(1-sin_γ**2)

#	if( np.abs(tan_α) > np.tan(camera.field_of_view_x/2)):
#		return np.array((np.NaN, np.NaN))

	x_pixel = camera.x_pixels/(2*camera.x_pixel_binning) + np.rint( (1/(camera.x_pixel_binning * camera.pixel_size)) * camera.focal_length * mirror_x * tan_α )
	y_pixel = camera.y_pixels/(2*camera.y_pixel_binning) + np.rint( (1/(camera.y_pixel_binning * camera.pixel_size)) * camera.focal_length * mirror_y * tan_γ )

#	if( (x_pixel > 5000) or (x_pixel < 0)):
#		print(f"px>5000 or<0: fov = {np.tan(camera.field_of_view_x/2)} and tan_α = {tan_α}")

	return Point(x_pixel, y_pixel, incident_beam.order, incident_beam.λ, intensity=incident_beam.intensity)

def simulate_spectrometer(draw=False, α=0, γ=0):
	# Simulation of light rays on a detector
	grating_constant = 1e6/79
	grating_angle = 58.0/in_rad

	optical_axis_angle = -grating_angle
	inplane_input_angle = α/in_rad
	outofplane_input_angle = γ/in_rad

	#lambdas = np.asarray([450.0,508.0,632.0])

	lambdas = None
	# dirty hack
	intensities = None
	try:
		from Resources.ArAmplitudes import ArAmpl
		intensities = np.asarray([cat_line[0] for cat_line in ArAmpl])
		lambdas = 0.1 * np.asarray([cat_line[1] for cat_line in ArAmpl])
	except:
		lambdas = 0.1 * np.arange(3500,11500,5)
		intensities = np.ones(lambdas.shape)
	lambdas = np.clip(lambdas,a_min=385,a_max=1150)

	cam = detector(x_pixels=QT_YETI.DETECTOR_X_PIXELS, y_pixels=QT_YETI.DETECTOR_Y_PIXELS, focal_length=QT_YETI.DETECTOR_FOCAL_LENGTH_MM * 1e-3, \
		pixel_size=QT_YETI.DETECTOR_PIXEL_SIZE_UM *1e-6, x_pixel_binning=QT_YETI.DETECTOR_PIXEL_BINNING, y_pixel_binning=QT_YETI.DETECTOR_PIXEL_BINNING,\
		φ=QT_YETI.DETECTOR_OUTOFPLANE_ANGLE_DEG/in_rad, θ= QT_YETI.DETECTOR_INPLANE_ANGLE_DEG/in_rad)
	new_prism = prism( n_outer=n_air, n_prism=sellmeier_n_f2, apex_up=False, apex_angle_α=60/in_rad, base_tilt_angle=-24.7/in_rad)
	rays = create_diffracted_beams(lambdas, intensities, optical_axis_angle, grating_constant, inplane_input_angle, outofplane_input_angle, camera=cam)

	if(draw):
		__fig = plt.figure(figsize=(5,5), dpi=200)
		__ax = __fig.add_subplot(111)

	points = []
	for ray in rays:

		x_dispersed_ray = pass_through_optical_element(ray, new_prism)
		point = project_onto_detector(x_dispersed_ray,cam)

		current_wavelength = x_dispersed_ray.λ
		points.append(point)
		
		if(draw):
			color = 'black'
			if( current_wavelength < 500.0):
				color = 'blue'
			elif ((current_wavelength >= 500.0) and (current_wavelength <= 600.0)):
				color = 'green'
			else:
				color = 'red'

			__ax.plot(point.x,point.y,'.',color=color,markersize=0.2)
		
		
	if (draw):
		plt.xlim([0, cam.x_pixels/cam.x_pixel_binning])
		plt.ylim([0, cam.x_pixels/cam.x_pixel_binning])
		plt.show(block=False)

	return np.asarray(points)


# %%

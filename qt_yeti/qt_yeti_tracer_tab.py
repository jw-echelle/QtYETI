from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from dataclasses import dataclass

import sys
import os

from qt_yeti.qt_yeti_general import *
from qt_yeti.qt_yeti_functions import *
from qt_yeti.qt_yeti_hardware_settings_tab import *

import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
from matplotlib.figure import Figure
#print(plt.rcParams.keys())

from scipy.optimize import curve_fit

#%%
###############################################################################################################################

class FlatfieldCanvas( FigureCanvasQTAgg, C ):

	def __init__(self, parent=None, width=QT_YETI.MATPLOTLIB_CANVAS_WIDTH, height=QT_YETI.MATPLOTLIB_CANVAS_HEIGHT, dpi=QT_YETI.MATPLOTLIB_DPI):
		self.control_figure = plt.figure(figsize=(width, height), dpi=dpi)
		super(FlatfieldCanvas, self).__init__(self.control_figure)

		self.saga()

		# https://github.com/matplotlib/matplotlib/issues/707/
		# https://stackoverflow.com/questions/22043549/matplotlib-and-qt-mouse-press-event-key-is-always-none
		self.setFocusPolicy( Qt.ClickFocus )
		self.setFocus()
		
		self.navigationbar = None
		self.navigationbar = NavigationToolbar2QT(self, parent=None)
		# Setup sample spectrogram		
		self.CurrentSpectrogram = Spectrogram("QtYeti.Sample")

		# Setup all plots and callbacks
		self.setup_plots()

		# Final touch
		self.control_figure.tight_layout()

		# Tracer Specific
		## Setup important lists
		# Could be omitted
		self.TracerSettings = OrderTracerSettings("from_file")

	### Plots and Callbacks ###
	def setup_plots(self):

		self.scale_type = "linear"

		# Axes
		self.axes_spectrogram = plt.subplot2grid((8,8),(0,1),colspan=7, rowspan = 7, fig=self.control_figure, label="Flatfield_Image")
		self.axes_row_profile = plt.subplot2grid((8,8),(0,0), colspan=1, rowspan = 7, fig=self.control_figure, label="Y-Profile")
		self.axes_col_profile = plt.subplot2grid((8,8),(7,1), colspan=7, rowspan = 1, fig=self.control_figure, label="X-Profile")
		
		self.axes_spectrogram.axes.xaxis.set_visible(False)
		self.axes_spectrogram.axes.yaxis.set_visible(False)
		self.axes_row_profile.axes.xaxis.set_visible(False)
		self.axes_row_profile.axes.yaxis.set_visible(False)
		self.axes_col_profile.axes.xaxis.set_visible(False)
		self.axes_col_profile.axes.yaxis.set_visible(False)

		self.spectrogram_plot = self.axes_spectrogram.imshow(self.CurrentSpectrogram.data, vmin=self.CurrentSpectrogram.intmin, vmax=self.CurrentSpectrogram.intmax,\
					cmap = 'afmhot', interpolation='none', extent=[0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1],\
					aspect='auto', label = "2D_Spectrogram")
		# variable, = [variable] = command_returning_variable()[0]
		[self.y_profile] = self.axes_row_profile.plot(self.CurrentSpectrogram.data[:,0], self.CurrentSpectrogram.yrange, color = 'red', linewidth=0.25, label="y_profile")
		[self.x_profile] = self.axes_col_profile.plot(self.CurrentSpectrogram.xrange,	self.CurrentSpectrogram.data[0,:], color = 'blue', linewidth=0.25, label = "x_profile")

		origin_x, origin_y = (int(0.5*QT_YETI.DETECTOR_X_PIXELS), int(0.5*QT_YETI.DETECTOR_Y_PIXELS))
		self.spectrogram_line_indicator = self.axes_spectrogram.axvline(origin_x, color='r', linewidth=0.5, alpha=0.75, label='0_z_ff_line_indicator')
		self.spectrogram_dot_indicator, = self.axes_spectrogram.plot(origin_x, origin_y,color='gold', marker='.', markersize=2, label='1_ff_d0t_indicator')
		self.row_profile_line_indicator = self.axes_row_profile.axhline(origin_y,color='b', linewidth=0.5, alpha=0.75)
		self.row_profile_dot_indicator, = self.axes_row_profile.plot(0,origin_y,'b.', markersize=3)

		# Text
		self.axes_spectrogram_text = self.axes_spectrogram.text(transform=self.axes_spectrogram.transAxes, ha='left', va='top', x=0.005, y=0.98, label="loaded_file_path", weight="bold", color="#AAAAAA", s=f"No data loaded.")

		# Event handling
		self.mpl_connect('button_release_event', self.canvas_key_or_mouse_event)
		self.mpl_connect('key_press_event', self.canvas_key_or_mouse_event)
		#self.mpl_connect('key_release_event', self.canvas_key_or_mouse_event)

	### Navigation Bar ###
	def return_navigation_bar(self):
		#self.navigationbar = NavigationToolbar2QT(self, parent=None)
		return self.navigationbar

	### MPL Callbacks ###
	@pyqtSlot()
	def canvas_key_or_mouse_event(self, event: matplotlib.backend_bases.Event):
		print(type(event))
		# include limits that event is inside the specific image axes
		if(event.inaxes is not self.axes_spectrogram):
			return
		
		# Get mouse coordinates
		evt_x = np.int32(np.rint(event.xdata))
		evt_y = np.int32(np.rint(event.ydata))

		# Add a new point to the order list
		if( isinstance(event, matplotlib.backend_bases.MouseEvent) and event.key is "shift"):
			print(f"shift + click")
			# Dont proceed of no order dots present
			if( not self.CurrentSpectrogram.order_centers_list ):
				return
			
			self.insert_order_center(evt_x, evt_y)
			self.update_order_centers_in_plot( evt_x , redraw=False)

		# Delete point
		if( isinstance(event, matplotlib.backend_bases.MouseEvent) and event.key == "alt"):
			print(f"alt + click")
			self.pop_order_center(evt_x, evt_y)
			self.update_order_centers_in_plot( evt_x , redraw=False)

		# Arrow key control
		if( isinstance(event, matplotlib.backend_bases.KeyEvent) ):
			#Move cursor via keypress
			curr_x,curr_y = self.spectrogram_dot_indicator.get_data()
			if(   event.key == "up"):
				curr_y= (curr_y + 1) % self.CurrentSpectrogram.ysize
			elif( event.key == "down"):
				curr_y= (curr_y - 1) % self.CurrentSpectrogram.ysize
			elif( event.key == "left"):
				curr_x = (curr_x - 1) % self.CurrentSpectrogram.xsize
			elif( event.key == "right"):
				curr_x = (curr_x + 1) % self.CurrentSpectrogram.xsize
			else:
				pass

			self.update_profiles(curr_x, curr_y, redraw=False)
			self.update_indicators(curr_x, curr_y, redraw=False)

		# Other mouse events
		if( hasattr(event,"button") ):
			
			if( event.button > 1):
				#
				# Find order centers and save them within the Spectrogram
				#
				self.find_order_centers_on_canvas(evt_x)

			else:
				self.update_profiles(evt_x, evt_y, redraw=False)
				self.update_indicators(evt_x, evt_y, redraw=False)

		self.draw_idle()

	# Plotting
	def load_spectrogram(self, requested_filename, HeaderDataUnit: fits.PrimaryHDU | fits.ImageHDU = None):
		if(requested_filename == None):
			QtYetiLogger(QT_YETI.ERROR,"No file name object provided.")

		# Update CurrentSpectrogram
		int_min, int_max = self.CurrentSpectrogram.update_spectrogram(requested_filename, HeaderDataUnit)

		self.row_of_intensity_max, self.column_of_intensity_max = np.unravel_index(np.argmax(self.CurrentSpectrogram.data), self.CurrentSpectrogram.shape)

		## Plot spectrogram
		# https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively
		self.spectrogram_plot.set_data(self.CurrentSpectrogram.data)
		self.spectrogram_plot.set_extent([0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1])
		self.spectrogram_plot.set_clim(int_min, int_max)

		# Change text on spectrogram
		self.axes_spectrogram_text.set_text(f"{self.CurrentSpectrogram.filename}")

		QtYetiLogger(QT_YETI.MESSAGE,f"{self.CurrentSpectrogram.filename} loaded.")

		self.draw_idle()

		return int_min, int_max

	def update_intensities(self, int_min=0, int_max=1):
		""" Set color / intensity limit """
		self.spectrogram_plot.set_clim(vmin=int_min, vmax=int_max)
		self.axes_row_profile.set_xlim(int_min, int_max)
		self.axes_col_profile.set_ylim(int_min, int_max)

		self.update_scale_type(self.scale_type, int_min, int_max)
		self.draw_idle()

	def update_scale_type(self, new_scale_type: str, int_min=0, int_max=1):
		"""
		Set new scale type: choose between Linear and Logarithmic.
		"""
		new_normalization = None
		if( (new_scale_type != None) and (int_max > int_min) ):
			if( new_scale_type == "log"):
				if(np.int32(np.rint(int_min)) <= 0):
					int_min = 1e-1
				new_normalization = matplotlib.colors.LogNorm(vmin=int_min, vmax=int_max)
					
			elif( new_scale_type == "linear"):
				new_normalization = matplotlib.colors.Normalize(vmin=int_min, vmax=int_max)

			else:
				return

			self.scale_type = new_scale_type
			self.axes_row_profile.set_xscale(new_scale_type)
			self.axes_col_profile.set_yscale(new_scale_type)
			self.spectrogram_plot.set_norm(new_normalization)
			self.draw_idle()

	def update_indicators(self, x=0, y=0, redraw=True):

		self.spectrogram_line_indicator.set_xdata(x)
		self.spectrogram_dot_indicator.set_data(x,y)
		self.row_profile_line_indicator.set_ydata(y)

		self.row_profile_dot_indicator.set_data(self.CurrentSpectrogram.data[row(y, self.CurrentSpectrogram.ysize),x],y)
		if (redraw):
			self.draw_idle()

	def update_profiles(self, x=0, y=0, redraw=True):

		mat_row = row(y, self.CurrentSpectrogram.ysize)
		mat_col = col(x, self.CurrentSpectrogram.xsize)
		data_slice_at_constant_x = self.CurrentSpectrogram.data[:, mat_col][::-1]
		data_slice_at_constant_y = self.CurrentSpectrogram.data[mat_row, :]
		x_range = self.CurrentSpectrogram.xrange
		x_range_max = self.CurrentSpectrogram.xrange.max()
		x_range_min =self.CurrentSpectrogram.xrange.min()
		y_range = self.CurrentSpectrogram.yrange
		y_range_max = self.CurrentSpectrogram.yrange.max()
		y_range_min = self.CurrentSpectrogram.yrange.min()

		self.x_profile.set_data( x_range, data_slice_at_constant_y )
		self.axes_col_profile.set_xlim(x_range_min, x_range_max)
		self.axes_col_profile.set_ylim( data_slice_at_constant_y.min(), QT_YETI.MATPLOTLIB_XY_LIM_SCALE * data_slice_at_constant_y.max() )
		
		self.y_profile.set_data( data_slice_at_constant_x , y_range) 
		self.axes_row_profile.set_xlim( data_slice_at_constant_x.min(), QT_YETI.MATPLOTLIB_XY_LIM_SCALE * data_slice_at_constant_x.max() )
		self.axes_row_profile.set_ylim( y_range_min,y_range_max)

		if (redraw):
			self.draw_idle()

	def update_order_centers_in_plot(self, x=0, redraw=True):
		# Goal Draw dots along a line at a fixed jdx
		# Get all objects in plot
		delete_mpl_plt_object_by_label(self.axes_spectrogram.lines,"orderdot")
		delete_mpl_plt_object_by_label(self.axes_spectrogram.texts,"orderdottext")

		for index,(dot_x, dot_y) in enumerate(self.CurrentSpectrogram.order_centers_list):
			self.axes_spectrogram.plot(dot_x, dot_y, color = "#00AA00", marker="s", fillstyle="none", markersize=5, markeredgewidth=0.7, label="orderdot")
			self.axes_spectrogram.text(QT_YETI.ANNOTATION_X_COORDINATE, dot_y+0,f"Relative trace number {index+1}",fontsize=6,color=YetiColors.YETI_WHITE,label="orderdottext")
		
		if (redraw):
			self.draw_idle()

	def find_order_centers_on_canvas(self, x_coordinate: int = 0) -> None:
		#
		# Find order centers and save them within the Spectrogram
		#
		echelle_find_orders(self.CurrentSpectrogram, x_coordinate, TracerSettings=self.TracerSettings )
		self.update_order_centers_in_plot(x_coordinate , redraw=True)

	def find_nearest_order(self, event):
		# if(event.inaxes == self.axes_flat_field):

		# 	evt_x = np.int32(np.rint(event.xdata))
		# 	evt_y = np.int32(np.rint(event.ydata))

		# 	nearest_order = np.argmin( np.abs( np.asarray( order_y_minima ) - row(idx) ) )
		# 	show_nearest_order(nearest_order,-1)
		# 	textbox_order_number.set_val(nearest_order)
		# return
		pass

	def insert_order_center(self, clicked_x: float, clicked_y: float):

		x_position,_ = self.CurrentSpectrogram.order_centers_list[0]

		# Abort when the click was too far away
		if( abs(clicked_x - x_position) > QT_YETI.DETECTOR_SPOT_SIZE_PX/4 ):
			return

		self.CurrentSpectrogram.insert_to_order_centers_list(clicked_y)
		
	def pop_order_center(self, clicked_x: float, clicked_y: float):

		x_position,_ = self.CurrentSpectrogram.order_centers_list[0]

		# Abort when the click was too far away
		if( abs(clicked_x - x_position) > QT_YETI.DETECTOR_SPOT_SIZE_PX/4 ):
			return

		self.CurrentSpectrogram.remove_from_order_list(clicked_y)

	# Tracing
	@elapsed_time
	def tracer_start_tracing(self, ReceivedTracerSettings: OrderTracerSettings):

		# Trace an order and find its center positions
		self.TracerSettings = ReceivedTracerSettings

		if( not self.CurrentSpectrogram.order_centers_list ):
			QtYetiLogger(QT_YETI.ERROR,"Order centers list is empty. Nothing to trace.")

		delete_mpl_plt_object_by_label(self.axes_spectrogram.lines, "fit_marker")
		delete_mpl_plt_object_by_label(self.axes_spectrogram.lines, "fit_line")

		# List of Order objects
		order_list = []

		DOWNSAMPLING_BY = 4
		# Where the magic happens
		x_y_series = echelle_order_tracer(self.CurrentSpectrogram, self.TracerSettings, downsampling_value = DOWNSAMPLING_BY, precision_mode=False, intensity_cut_off_value=500)

		""" Experimental """
		FIT_REDUCTION = False
		fit_reduction_factor = 1.
		if( FIT_REDUCTION ):
			fit_reduction_factor = 1000.
		correction_array = np.asarray([fit_reduction_factor**(-3),fit_reduction_factor**(-2),fit_reduction_factor**(-1), fit_reduction_factor**(0), fit_reduction_factor**(1)])
		correction_array = np.asarray([fit_reduction_factor**(-1), fit_reduction_factor**(0), fit_reduction_factor**(1)])
		"""##############"""

		#bounds = ([0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,0],[4,0,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])

		# Setup loop
		current_order_number = 1.000

		for x_series, y_series in x_y_series:

			x_series = np.asarray(x_series)
			y_series = np.asarray(y_series)

			try:
				# Fit the found points with a polynomial function
				# To reduce the numerical values, the data arrays are divided by `fit_reduction_factor` and later corrected
				fit_output, _ = curve_fit(echelle_order_fit_function, x_series / fit_reduction_factor, y_series / fit_reduction_factor)#, bounds=bounds)
				if( FIT_REDUCTION ):
					fit_output = fit_output * correction_array

				x_start = x_series[0]
				x_stop = x_series[-1]
				order_list.append( Order(number_m=current_order_number, x_range=np.arange(x_start, x_stop+1, 1), fit_parameters=fit_output) )

				# Plotting
				self.axes_spectrogram.plot(x_series,y_series,'.', markersize=1, linewidth=1, color="#00AA00",label="fit_marker")
				self.axes_spectrogram.plot(x_series, echelle_order_fit_function(x_series, *fit_output),'-', linewidth=0.75, color="#00AA00", label="fit_line")
				
				current_order_number += 1.0

			except Exception as Error:
				# QtYetiLogger(QT_YETI.ERROR, f"Error in Tracer: {Error}.", True)
				# QtYetiLogger(QT_YETI.ERROR,\
				#  f"XSERIES = {x_series}\r\nYSERIES={y_series}"\
				# )
				pass

		# Update list for all instances of the type "Spectrogram" via class methods
		Spectrogram.update_order_list( order_list )

		# Redraw after loop
		self.draw_idle()

	def tracer_find_lines_dots(self) -> list:
		"""
		Find all lines and dots and their indices created by the order tracer
		return: list of tracer line indices
		"""
		all_lines = self.axes_spectrogram.lines
		tracer_line_indices = []

		# Find all fit dots and fit lines
		for line_index, line in enumerate(all_lines):
			if("fit_marker" in line.get_label()):
				tracer_line_indices.append(line_index)
			elif("fit_line" in line.get_label()):
				tracer_line_indices.append(line_index)

		tracer_line_indices.sort()
		tracer_line_indices.reverse()
		return tracer_line_indices
	
	def tracer_lines_set_visibility(self, visible=True):
		
		tracer_line_indices = self.tracer_find_lines_dots()
		all_lines = self.axes_spectrogram.lines

		for idx in tracer_line_indices:
			all_lines[idx].set_visible(visible)
		self.draw_idle()

	def save_order_information(self, loaded_filename = ""):
		self.CurrentSpectrogram.save_order_information(loaded_filename)

	def load_order_information(self, requested_filename = ""):
		Spectrogram.load_order_information(requested_filename)

# Tracer Window
class TracerSettingWindow(QWidget):
	def __init__(self, MPLCanvas: FlatfieldCanvas):
		super(TracerSettingWindow, self).__init__(parent=None)
		
		self.setWindowTitle(QT_YETI.TRACER_WINDOW_NAME)
		#self.setGeometry(QRect(10,10,10,10))
		self.resize(QT_YETI.TRACER_WINDOW_WIDTH, QT_YETI.TRACER_WINDOW_HEIGHT)
		self.setWindowFlags(Qt.WindowStaysOnTopHint)
		self.setWindowIcon(QIcon(QT_YETI.IMAGE_PATH))

		self.setup_tracer_window()

		self.canvas = MPLCanvas
		self.CurrentSettings = -1
		self.load_tracer_settings_from_file()

	def setup_tracer_window(self):
		self.tracer_window_layout = QVBoxLayout()

		self.first_absolute_order_box = YetiSpinBox()
		self.abs_order_number_m_direction_down_btn = QRadioButton("Increasing |m| from top to bottom: TOP-DOWN")
		self.abs_order_number_m_direction_up_btn = QRadioButton("Increasing |m| from bottom to top: BOTTOM-UP")

		self.spotsize_box = YetiSpinBox()
		self.image_slicer_box = QCheckBox()
		self.image_slicer_separation_box = YetiSpinBox()
		self.distance_to_image_edge_box = YetiSpinBox()
		self.samples_per_order_box = YetiSpinBox()
		self.peak_distance_box = YetiSpinBox()
		self.peak_height_box = YetiDoubleSpinBox()
		self.peak_prominence_box = YetiDoubleSpinBox()
		self.peak_width_box = YetiSpinBox()
		self.smoothing_stiffness_box = YetiDoubleSpinBox()
		self.smoothing_order_box = YetiSpinBox()
		self.load_settings_btn = QPushButton("Load Settings")
		self.save_settings_btn = QPushButton("Save Settings")
		self.trace_btn = QPushButton("Start Tracing")
		self.visible_traces = QCheckBox("View / Hide tracer lines")

		self.abs_order_number_m_direction_down_btn.setChecked(True)

		qspinboxlimit = (np.power(2,16)-1)
		self.first_absolute_order_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.spotsize_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		#self.image_slicer_box
		self.image_slicer_separation_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.distance_to_image_edge_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.samples_per_order_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.peak_distance_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.peak_height_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.peak_prominence_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.peak_width_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.smoothing_stiffness_box.setRange(-1 * qspinboxlimit, qspinboxlimit)
		self.smoothing_order_box.setRange(-1 * qspinboxlimit, qspinboxlimit)

		self.visible_traces.setChecked(True)

		self.spotsize_box.setReadOnly(True)
		self.spotsize_box.setStyleSheet("background-color: #DDDDDD;")
		self.image_slicer_box.setCheckable(False)
		self.image_slicer_box.setStyleSheet("background-color: #DDDDDD;")
		self.image_slicer_separation_box.setReadOnly(True)
		self.image_slicer_separation_box.setStyleSheet("background-color: #DDDDDD;")

		lines = [QFrame(),QFrame(),QFrame(),QFrame()]
		for line in lines:
			line.setFrameShape(QFrame.HLine)
			line.setFrameShadow(QFrame.Sunken)

		self.tracer_control = QFormLayout()
		self.tracer_control.addRow(QLabel("<b>Peak Finding & Tracer Settings</b>"))
		self.tracer_control.addRow(QLabel("\tConvention: mλ/d cos(γ) = sin(α) + sin(β)"))
		self.tracer_control.addRow(self.first_absolute_order_box, QLabel("First absolute Order"))
		self.tracer_control.addRow(self.abs_order_number_m_direction_down_btn)
		self.tracer_control.addRow(self.abs_order_number_m_direction_up_btn)
		self.tracer_control.addRow(lines[0])
		self.tracer_control.addRow(QLabel("Read-only. Update values in the Hardware Tab"))
		self.tracer_control.addRow(self.spotsize_box, QLabel("Spotsize of fiber on detector"))
		self.tracer_control.addRow(self.image_slicer_box, QLabel("Image Slicer Yes/No"))
		self.tracer_control.addRow(self.image_slicer_separation_box, QLabel("Image Slicer Offset"))
		self.tracer_control.addRow(lines[1])
		self.tracer_control.addRow(self.distance_to_image_edge_box, QLabel("Distance to Image edge (px)"))
		self.tracer_control.addRow(self.samples_per_order_box, QLabel("Samples per Order"))
		self.tracer_control.addRow(self.peak_distance_box, QLabel("Peak Distance (px)"))
		self.tracer_control.addRow(self.peak_height_box, QLabel("Peak Height (%)"))
		self.tracer_control.addRow(self.peak_prominence_box, QLabel("Peak Prominence (%)"))
		self.tracer_control.addRow(self.peak_width_box, QLabel("Peak Width (px)"))
		self.tracer_control.addRow(QLabel("Whittaker Smoothing & Peak Finding"))
		self.tracer_control.addRow(self.smoothing_stiffness_box, QLabel("Smoothing Stiffness"))
		self.tracer_control.addRow(self.smoothing_order_box, QLabel("Smoothing Order"))
		self.tracer_control.addRow(lines[2])

		self.tracer_window_layout.addLayout(self.tracer_control)
		self.tracer_window_layout.addWidget(self.save_settings_btn)
		self.tracer_window_layout.addWidget(self.load_settings_btn)
		self.tracer_window_layout.addWidget(lines[3])
		self.tracer_window_layout.addWidget(self.trace_btn)
		self.tracer_window_layout.addWidget(self.visible_traces)

		self.setLayout(self.tracer_window_layout)

		"""
		Stupid test?
		"""
		#print(self.findChildren(QSpinBox))
		#for child in self.tracer_control.findChildren(QSpinBox):
		#	child.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)
		#	child.setRange(-1*qspinboxlimit, qspinboxlimit)

		# Signals/Slots
		self.abs_order_number_m_direction_down_btn.clicked.connect(self.update_current_settings)
		self.abs_order_number_m_direction_up_btn.clicked.connect(self.update_current_settings)
		self.save_settings_btn.clicked.connect(self.save_tracer_settings_to_file)
		self.load_settings_btn.clicked.connect(self.load_tracer_settings_from_file)
		self.trace_btn.clicked.connect(self.start_tracer)
		self.visible_traces.toggled.connect(self.toggle_tracer_line_visiblity)

	def load_tracer_settings_from_file(self):
		self.CurrentSettings = -1
		self.CurrentSettings = OrderTracerSettings("from_file")
		self.canvas.TracerSettings = self.CurrentSettings
		self.update_controls()

	def update_controls(self):
		"""
		`TracerSettingsWindow.update_controls()`

		Method → update_controls()
		--------------------------
		Update all Qt buttons and boxes from the settings file
		"""

		self.spotsize_box.setValue(					self.CurrentSettings.spotsize_px)
		self.image_slicer_box.setChecked(			self.CurrentSettings.image_slicer)
		self.image_slicer_separation_box.setValue(	self.CurrentSettings.image_slicer_separation_px)

		self.first_absolute_order_box.setValue(		self.CurrentSettings.first_absolute_order)
		# Check direction of orders
		current_direction_value = self.CurrentSettings.abs_order_number_m_direction
		if  (current_direction_value == "up"):
			self.abs_order_number_m_direction_up_btn.setChecked(True)
		elif(current_direction_value == "down"):
			self.abs_order_number_m_direction_down_btn.setChecked(True)
		else:
			QtYetiLogger(QT_YETI.ERROR,f"Unknown abs_order_number_m_direction: {current_direction_value}",True)
			raise ValueError(f"Unknown abs_order_number_m_direction: {current_direction_value}. It has to be up or down. Please check {QT_YETI.SETTINGS_INI_PATH}")

		self.distance_to_image_edge_box.setValue(	self.CurrentSettings.distance_to_image_edge_px)
		self.samples_per_order_box.setValue(		self.CurrentSettings.samples_per_order)
		self.peak_distance_box.setValue(			self.CurrentSettings.peak_distance_px)
		self.peak_height_box.setValue(				self.CurrentSettings.peak_height * 100)
		self.peak_prominence_box.setValue(			self.CurrentSettings.peak_prominence * 100)
		self.peak_width_box.setValue(				self.CurrentSettings.peak_width_px)
		self.smoothing_stiffness_box.setValue(		self.CurrentSettings.smoothing_stiffness)
		self.smoothing_order_box.setValue(			self.CurrentSettings.smoothing_order)

	@pyqtSlot()
	def update_current_settings(self):
		"""
		`TracerSettingsWindow.update_current_settings()`
		
		Method → update_current_settings()
		--------------------------
		Update CurrentSettings (TracerSettings) from Qt button and box inputs
		"""		
		self.CurrentSettings.spotsize = 					self.spotsize_box.value()
		self.CurrentSettings.image_slicer = 				self.image_slicer_box.isChecked()
		self.CurrentSettings.image_slicer_separation_px = 	self.image_slicer_separation_box.value()

		self.CurrentSettings.first_absolute_order = 		self.first_absolute_order_box.value()
		# Handle radio buttons
		current_down_btn_value = self.abs_order_number_m_direction_down_btn.isChecked()
		current_up_btn_value = self.abs_order_number_m_direction_up_btn.isChecked()

		direction_value = ""
		if   (current_down_btn_value is True) and (current_up_btn_value is False):
			direction_value = "down"
			QtYetiLogger(QT_YETI.MESSAGE, f"Order number |m| increases from top to bottom. Order sequence: TOP-DOWN.")

		elif (current_down_btn_value is False) and (current_up_btn_value is True):
			direction_value = "up"
			QtYetiLogger(QT_YETI.MESSAGE, f"Order number |m| increases from bottom to top. Order sequence: BOTTOM-UP")

		else:
			QtYetiLogger(QT_YETI.ERROR, f"Direction was neither up nor down.", True)
			raise ValueError(f"Unknown state for abs_order_number_m_direction: {self.abs_order_number_m_direction}. It has to be up or down. Please check {QT_YETI.SETTINGS_INI_PATH}")
		self.CurrentSettings.abs_order_number_m_direction = direction_value
		QT_YETI.DETECTOR_ORDER_NUMBER_MAGNITUDE_INCREASE_DIRECTION = direction_value

		self.CurrentSettings.distance_to_edge = 			self.distance_to_image_edge_box.value()
		self.CurrentSettings.samples_per_order = 			self.samples_per_order_box.value()
		self.CurrentSettings.peak_distance_px = 			self.peak_distance_box.value()
		self.CurrentSettings.peak_height = 					self.peak_height_box.value() / 100.0
		self.CurrentSettings.peak_prominence = 				self.peak_prominence_box.value() / 100.0
		self.CurrentSettings.peak_width_px =				self.peak_width_box.value()
		self.CurrentSettings.smoothing_stiffness = 			self.smoothing_stiffness_box.value()
		self.CurrentSettings.smoothing_order = 				self.smoothing_order_box.value()
		
		self.canvas.TracerSettings = self.CurrentSettings

	def save_tracer_settings_to_file(self):
		"""
		`TracerSettingsWindow.save_tracer_settings_to_file()`\r\n
		Method → save_tracer_settings_to_file() 
		---------------------------------------
		Save the current state to the settings ini file
		"""		
		self.update_current_settings()
		self.CurrentSettings.to_file()
		self.canvas.TracerSettings = self.CurrentSettings

	@pyqtSlot()
	def start_tracer(self):
		self.save_tracer_settings_to_file()
		NewSettings = self.CurrentSettings
		# Start tracing in Canvas Class
		self.canvas.tracer_start_tracing(NewSettings)
	
	@pyqtSlot()
	def toggle_tracer_line_visiblity(self):
		new_state = self.visible_traces.checkState()
		visible = True
		if( new_state == 0):
			visible = False
		self.canvas.tracer_lines_set_visibility(visible)

	def show(self):
		super(TracerSettingWindow, self).show()
	
	def closeEvent(self,event):
		super(TracerSettingWindow,self).closeEvent(event)

# Tab for MainWindow
class TabOrderTracer(QWidget):
	def __init__(self, parent):
		super(TabOrderTracer,self).__init__(parent)

		# Create Matplotlib Canvas
		self.figure_canvas = FlatfieldCanvas(parent=self)
		self.TracerWindow = TracerSettingWindow(self.figure_canvas)

		# Setup and customize
		self.setupTabStructure()
		self.customizeTab()

		self.spectrogram_filename = ""

		# for child in self.findChildren((QPushButton, QSpinBox)):
		# 	child.setFocusPolicy(Qt.NoFocus)
		# self.setFocusPolicy(Qt.ClickFocus)
		# self.setFocus(Qt.NoFocusReason)
		# self.activateWindow()
		self.setFocusPolicy(Qt.StrongFocus)
		self.setFocus()

		self.figure_canvas.draw()

	def setupTabStructure(self):
		# Top Level Tab layout
		self.tab_layout = QVBoxLayout()
		self.setLayout(self.tab_layout)

		# Add Matplotlib Canvas
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

		# Intensity spinbox widget
		self.intensity_control = QWidget()
		self.intensity_control_layout = QFormLayout()
		self.intensity_control_layout.setContentsMargins(0,0,0,0)
		self.intensity_control.setLayout( self.intensity_control_layout )

		# Axes control spinbox widget
		self.axes_control = QWidget()
		self.axes_control_layout = QFormLayout()
		self.axes_control_layout.setContentsMargins(0,0,0,0)
		self.axes_control.setLayout( self.axes_control_layout )

		# Create spinboxes
		self.intensity_max = YetiSpinBox()
		self.intensity_min = YetiSpinBox()
		self.log_scale_chkbx = QCheckBox("Log Scale")
		self.x_max	= YetiSpinBox()
		self.x_min	= YetiSpinBox()
		self.y_max	= YetiSpinBox()
		self.y_min	= YetiSpinBox()

		# Create Buttons
		self.action_load_spectrogram_btn	= QPushButton("Load Flatfield File")
		self.action_open_tracer_btn			= QPushButton("Open Tracer")
		self.action_save_coefficients_btn	= QPushButton("Save Trace Fit Coefficients to file")
		self.action_load_coefficients_btn	= QPushButton("Load Trace Fit Coefficients from file")
		self.action_free_0_btn	= QPushButton("Free to be programmed (0)")
		self.action_free_1_btn	= QPushButton("Free to be programmed (1)")
		self.action_free_2_btn	= QPushButton("Free to be programmed (2)")
		self.action_custom_action_btn		= QPushButton("Custom Action - Set abs order number m")


		# Fill layouts
		self.intensity_control_layout.addRow(self.intensity_max, QLabel("Intensity maximum"))
		self.intensity_control_layout.addRow(self.intensity_min, QLabel("Intensity minimum"))
		self.intensity_control_layout.addRow(self.log_scale_chkbx)

		self.axes_control_layout.addRow(self.x_max, QLabel("X Axis Maximum"))
		self.axes_control_layout.addRow(self.x_min, QLabel("X Axis Minimum"))
		self.axes_control_layout.addRow(self.y_max, QLabel("Y Axis Maximum"))
		self.axes_control_layout.addRow(self.y_min, QLabel("Y Axis Minimum"))

		self.control_panel_layout.addWidget(self.intensity_control,0,0,4,2)
		self.control_panel_layout.addWidget(self.axes_control,0,1,4,2)
		
		self.control_panel_layout.addWidget(self.action_load_spectrogram_btn,	0,2,1,2)
		self.control_panel_layout.addWidget(self.action_open_tracer_btn,		1,2,1,2)
		self.control_panel_layout.addWidget(self.action_save_coefficients_btn,	0,4,1,2)
		self.control_panel_layout.addWidget(self.action_load_coefficients_btn,	1,4,1,2)

		self.control_panel_layout.addWidget(self.action_free_0_btn,	2,2,1,2)
		self.control_panel_layout.addWidget(self.action_free_1_btn,	3,2,1,2)
		self.control_panel_layout.addWidget(self.action_free_2_btn,	2,4,1,2)
		self.control_panel_layout.addWidget(self.action_custom_action_btn,	3,4,1,2)

		# Widths and limits
		self.intensity_max.setMaximumWidth(QT_YETI.SPIN_BOX_MAX_WIDTH)
		self.intensity_max.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)
		self.intensity_min.setMaximumWidth(QT_YETI.SPIN_BOX_MAX_WIDTH)
		self.intensity_min.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)

		self.x_max.setMaximumWidth(QT_YETI.SPIN_BOX_MAX_WIDTH)
		self.x_max.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)
		self.x_min.setMaximumWidth(QT_YETI.SPIN_BOX_MAX_WIDTH)
		self.x_min.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)
		self.y_max.setMaximumWidth(QT_YETI.SPIN_BOX_MAX_WIDTH)
		self.y_max.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)
		self.y_min.setMaximumWidth(QT_YETI.SPIN_BOX_MAX_WIDTH)
		self.y_min.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)

		self.intensity_max.setMaximum(QT_YETI.DETECTOR_MAX_INTENSITY)
		self.intensity_max.setMinimum( -1 * QT_YETI.DETECTOR_MAX_INTENSITY)
		self.intensity_min.setMaximum(QT_YETI.DETECTOR_MAX_INTENSITY)
		self.intensity_min.setMinimum( -1 * QT_YETI.DETECTOR_MAX_INTENSITY)
		self.intensity_max.setValue(np.int32(self.figure_canvas.CurrentSpectrogram.intmax))
		self.intensity_min.setValue(np.int32(self.figure_canvas.CurrentSpectrogram.intmin))

		self.x_max.setValue(self.figure_canvas.CurrentSpectrogram.xrange.max())
		self.x_min.setValue(self.figure_canvas.CurrentSpectrogram.xrange.min())
		self.y_max.setValue(self.figure_canvas.CurrentSpectrogram.yrange.max())
		self.y_min.setValue(self.figure_canvas.CurrentSpectrogram.yrange.min())

		### ### ### Connect signals/slots ### ### ###
		self.intensity_max.valueChanged.connect(self.gui_intensity_changed)
		self.intensity_min.valueChanged.connect(self.gui_intensity_changed)
		self.log_scale_chkbx.stateChanged.connect(self.gui_log_scale_changed)

		self.x_max.valueChanged.connect(self.gui_axes_changed)
		self.x_min.valueChanged.connect(self.gui_axes_changed)
		self.y_max.valueChanged.connect(self.gui_axes_changed)
		self.y_min.valueChanged.connect(self.gui_axes_changed)

		self.action_load_spectrogram_btn.clicked.connect(self.gui_load_spectrogram_file)
		self.action_open_tracer_btn.clicked.connect(self.TracerWindow.show)
		self.action_save_coefficients_btn.clicked.connect(self.gui_save_order_information)
		self.action_load_coefficients_btn.clicked.connect(self.gui_load_order_information)
		self.action_free_0_btn.clicked.connect(self.gui_dummy_callback)
		self.action_free_1_btn.clicked.connect(self.gui_dummy_callback)
		self.action_free_2_btn.clicked.connect(self.gui_dummy_callback)
		self.action_custom_action_btn.clicked.connect(self.gui_custom_action)

	# Signals / Slots
	@pyqtSlot()
	def gui_load_spectrogram_file(self):
		caption = "Select spectrogram file"
		initial_filter="Fits files (*.fit *.fits)"
		file_filter="Fits files (*.fit *.fits);; All files (*.*)"
		requested_filename, _  = QFileDialog.getOpenFileName(self, caption = caption, initialFilter=initial_filter, filter=file_filter)

		if(requested_filename == ""):
			return

		CurrentHDU = qt_yeti_handle_fits_file(self, requested_filename)
		if( CurrentHDU == None):
			return
		
		int_min, int_max = self.figure_canvas.load_spectrogram( requested_filename, CurrentHDU )

		self.spectrogram_filename = requested_filename
		self.intensity_max.setValue(int_max)
		self.intensity_min.setValue(int_min)
		self.x_max.setValue(self.figure_canvas.CurrentSpectrogram.xsize-1)
		self.y_max.setValue(self.figure_canvas.CurrentSpectrogram.ysize-1)
		self.x_min.setValue(0)
		self.y_min.setValue(0)
		print(self.figure_canvas.CurrentSpectrogram)

	@pyqtSlot()
	def gui_intensity_changed(self):
		max = self.intensity_max.value()
		min = self.intensity_min.value()
		QtYetiLogger(QT_YETI.MESSAGE,f"New Intensity Range: {min,max}",True)
		if( min < max):
			self.figure_canvas.update_intensities(min, max)

	@pyqtSlot()
	def gui_log_scale_changed(self):
		# Toggle Log scale in plots
		int_max = self.intensity_max.value()
		int_min = self.intensity_min.value()
		scale_type = None
		if(self.log_scale_chkbx.checkState() != 0):
			scale_type = "log"
		else:
			scale_type = "linear"
		self.figure_canvas.update_scale_type(scale_type, int_min, int_max)

	@pyqtSlot()
	def gui_axes_changed(self):
		x_max = self.x_max.value()
		x_min = self.x_min.value()
		y_max = self.y_max.value()
		y_min = self.y_min.value()
		if((x_min < x_max) | (y_min < y_max)):
			self.figure_canvas.axes_spectrogram.set_xlim(x_min, x_max)
			self.figure_canvas.axes_spectrogram.set_ylim(y_min, y_max)
			self.figure_canvas.axes_row_profile.set_ylim(y_min, y_max)
			self.figure_canvas.axes_col_profile.set_xlim(x_min, x_max)
			self.figure_canvas.draw_idle()

	@pyqtSlot()
	def gui_save_order_information(self):
		QtYetiLogger(QT_YETI.MESSAGE,f"Entered Callbäck function",True)
		self.figure_canvas.save_order_information(self.spectrogram_filename)

	@pyqtSlot()
	def gui_load_order_information(self):
		caption="Select Order Information File"
		initial_filter="Order Information Files (*.txt)"
		file_filter="Order Information Files (*.txt);; All files (*.*)"
		requested_filename, _  = QFileDialog.getOpenFileName(caption = caption, initialFilter=initial_filter, filter=file_filter)
		if(requested_filename != ""):
			self.figure_canvas.load_order_information(requested_filename)

	@pyqtSlot()
	def gui_dummy_callback(self):
		pass
		QtYetiLogger(QT_YETI.WARNING,f"Nothing done.",True)

	@pyqtSlot()
	def gui_custom_action(self):
		QtYetiLogger(QT_YETI.WARNING, f"Custom action botton clicked",True)
		"""
		Add custom action below.
		"""
		self.figure_canvas.CurrentSpectrogram.set_absolute_order_number_m(self.figure_canvas.TracerSettings.first_absolute_order)
		QtYetiLogger(QT_YETI.MESSAGE,f"Printing list of order.",True)
		for order in self.figure_canvas.CurrentSpectrogram.order_list:
			print(np.array(order.number_m))
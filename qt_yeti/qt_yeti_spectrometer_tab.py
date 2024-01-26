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

#%%
###############################################################################################################################

class SpectrometerCanvas( FigureCanvasQTAgg ):

	def __init__(self, parent=None, width=QT_YETI.MATPLOTLIB_CANVAS_WIDTH, height=QT_YETI.MATPLOTLIB_CANVAS_HEIGHT, dpi=QT_YETI.MATPLOTLIB_DPI):
		self.control_figure = plt.figure(figsize=(width, height), dpi=dpi)
		super(SpectrometerCanvas, self).__init__(self.control_figure)

		# https://github.com/matplotlib/matplotlib/issues/707/
		# https://stackoverflow.com/questions/22043549/matplotlib-and-qt-mouse-press-event-key-is-always-none
		self.setFocusPolicy( Qt.ClickFocus )
		self.setFocus()

		self.navigationbar = NavigationToolbar2QT(self, parent=None)

		# Setup sample spectrogram		
		self.CurrentSpectrogram = Spectrogram("QtYeti.Sample")
		self.active_order_index = 0
		self.summation_method = "simple_sum"

		# Setup all plots and callbacks
		self.setup_plots()

		# Final touch
		self.control_figure.tight_layout()

	### Plots and Callbacks ###
	def setup_plots(self):

		self.scale_type = "linear"

		# Axes
		self.axes_spectrogram = plt.subplot2grid((16,16),(0,0),colspan=16, rowspan = 9, fig=self.control_figure, label="Full_Spectrogram")
		self.axes_spectrum = plt.subplot2grid((16,16),(9,0),colspan=16, rowspan = 6, fig=self.control_figure, label="Extracted_Spectrum")
		self.axes_spectrogram.set_ylabel(r"$m · \lambda(X,Y)$")
		self.axes_spectrum.set_xlabel(r"$\lambda(X,Y)$")
		self.axes_spectrum.set_ylabel(r"Counts (arb. u.)")

		self.spectrogram_plot = self.axes_spectrogram.imshow(self.CurrentSpectrogram.data,\
			vmin=self.CurrentSpectrogram.intmin,\
			vmax=self.CurrentSpectrogram.intmax,\
			cmap = 'inferno',\
			interpolation='none',\
			extent=[0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1],\
			aspect='auto',\
			label = "2D_Spectrogram")
		[self.spectrum_plot] 		= self.axes_spectrum.plot(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[self.CurrentSpectrogram.ysize//2,:], linewidth=0.6, label="Data_Spectrum")
		[self.spectrum_plot_pixels] = self.axes_spectrum.plot(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[self.CurrentSpectrogram.ysize//2,:], alpha=0.6, linewidth=0.45, drawstyle="steps-mid", label="Data_Spectrum_Pixel_Intensity")
		self.axes_spectrum.set_xlim( self.CurrentSpectrogram.xrange.min(), self.CurrentSpectrogram.xrange.max() )
		self.axes_spectrum.set_ylim( self.CurrentSpectrogram.intmin, self.CurrentSpectrogram.intmax )

		[self.order_poly_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.75,linewidth="0.5",color=YetiColors.RED, label="Order_Poly_Plot")
		[self.order_higher_row_plot] = self.axes_spectrogram.plot(0,0,alpha=0.25,linewidth="0.5",color=YetiColors.YETI_WHITE, label="Order_Summation_Higher_Row")
		[self.order_lower_row_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.25,linewidth="0.5",color=YetiColors.YETI_WHITE, label="Order_Summation_Lower_Row")

		# Text
		self.axes_spectrogram_text = self.axes_spectrogram.text(transform=self.axes_spectrogram.transAxes, ha='left', va='top', x=0.005, y=0.98, label="loaded_file_path", weight="bold", color="#AAAAAA", s=f"No data loaded.")
		self.axes_spectrum_text = self.axes_spectrum.text(transform=self.axes_spectrum.transAxes, ha='left', va='top', x=0.005, y=0.98, s=f"No data loaded.", label="texts_current_order")

		# Event handling
		self.mpl_connect('button_release_event', self.canvas_key_or_mouse_event)
		self.mpl_connect('key_press_event', self.canvas_key_or_mouse_event)
		self.mpl_connect("scroll_event", self.canvas_scroll_event)

	### Navigation Bar ###
	def return_navigation_bar(self):
		return self.navigationbar

	### MPL Callbacks ###
	def canvas_key_or_mouse_event(self, event):
		if(event.inaxes == self.axes_spectrogram):
			evt_x = np.int32(np.rint(event.xdata))
			evt_y = np.int32(np.rint(event.ydata))

			nearest_order_index = self.find_nearest_order_index(evt_x, evt_y)
			if(nearest_order_index != np.NaN):
				self.update_spectrum(nearest_order_index)

			QtYetiLogger(QT_YETI.MESSAGE,f"Nearest order number {nearest_order_index + 1}")

		pass

	def canvas_scroll_event(self, event):
		if(event.inaxes == self.axes_spectrogram):
			#### REMOVE #### QtYetiLogger(QT_YETI.MESSAGE,f"Event {event}, Event Type: {type(event)}, Event Button: {event.button}, Event Step: {event.step}")
			new_index = int(np.clip(self.active_order_index + int(event.step),a_min=1,a_max=len(self.CurrentSpectrogram.order_list)-1))
			self.active_order_index = new_index
			self.update_spectrum(self.active_order_index)

	# Plotting
	def load_spectrogram(self, requested_filename, HeaderDataUnit: fits.PrimaryHDU | fits.ImageHDU = None):
		if(requested_filename == None):
			QtYetiLogger(QT_YETI.ERROR,"No file name object provided.")

		# Update CurrentSpectrogram
		int_min, int_max = self.CurrentSpectrogram.update_spectrogram(requested_filename, HeaderDataUnit)
		sample_data = (self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[np.uint32(self.CurrentSpectrogram.ysize/2),:])

		self.row_of_intensity_max, self.column_of_intensity_max = np.unravel_index(np.argmax(self.CurrentSpectrogram.data), self.CurrentSpectrogram.shape)

		## Plot spectrogram
		# https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively
		#self.spectrogram_plot.set_data(self.CurrentSpectrogram.data)
		self.spectrogram_plot.set_data(self.CurrentSpectrogram.data)
		self.spectrogram_plot.set_extent([0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1])
		self.spectrogram_plot.set_clim(int_min, int_max)

		# Change text on spectrogram
		self.axes_spectrogram_text.set_text(f"{requested_filename}")

		self.spectrum_plot.set_data(sample_data)
		self.spectrum_plot_pixels.set_data(sample_data)
		# self.spectrum_plot.axes.set_ylim([0, int_max])
		self.axes_spectrum.set_xlim([0, self.CurrentSpectrogram.xsize-1])
		#self.axes_spectrum.set_ylim([0, 1.1*self.CurrentSpectrogram.data.max()])


		
		QtYetiLogger(QT_YETI.MESSAGE,f"{requested_filename} loaded.")

		self.draw_idle()

		return int_min, int_max

	def update_intensities(self, int_min=0, int_max=1):
		""" Set color / intensity limit """
		self.spectrogram_plot.set_clim(vmin=int_min, vmax=int_max)
#		self.axes_spectrum.set_xlim(int_min, int_max)
		self.axes_spectrum.set_ylim(int_min, int_max)

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
			self.spectrogram_plot.set_norm(new_normalization)
			self.axes_spectrum.set_yscale(new_scale_type)
			self.draw_idle()

	def load_order_information(self, requested_filename = ""):
		Spectrogram.load_order_information(requested_filename)

	def plot_spectrum(self, order_index: int) -> None:
		"""
		Plot the spectrum along a fitted order

		### Details
		• Extractring data from the spectrogram along a polynomial\r\n
		• Updating plots and annotations

		#### Parameters:
			`order_index` (int): Requested order index
		"""
		self.active_order_index = order_index

		# Optimum Extraction
		x_range, spectral_data, extracted_rows, extracted_columns \
			= echelle_trace_optimal_extraction(self.CurrentSpectrogram, self.active_order_index, self.summation_method)

		CurrentOrder = self.CurrentSpectrogram.order_list[self.active_order_index]
		order_number = CurrentOrder.number_m
		order_number_calibrated = CurrentOrder.order_number_calibrated
		fit_parameters = CurrentOrder.fit_parameters
		fitted_polynomial = np.asarray(echelle_order_fit_function(x_range, *fit_parameters))

		# Plotting
		self.spectrum_plot.set_data( x_range, spectral_data )
		self.spectrum_plot_pixels.set_data( x_range, spectral_data )
		#### REMOVE ##### self.calibration_help_plot.set_data( echelle_trace_quick_plot(self.CurrentSpectrogram, self.active_order_index, spectral_data.max()) )
		# Plot an indication of there the order was traced
		self.order_poly_plot.set_data(x_range, fitted_polynomial)

		# Adapt axes values
		self.axes_spectrum.set_ylim([0, 1.1 * spectral_data.max()])

		# Delete previous
		delete_mpl_plt_object_by_label(self.axes_spectrogram.texts,"trace_description")

		# Annotate trace
		text_x_coordinate = QT_YETI.ANNOTATION_X_COORDINATE
		text_y_coordinate = 5 + np.asarray(echelle_order_fit_function(text_x_coordinate, *fit_parameters)).max()
		self.axes_spectrogram.text(text_x_coordinate, text_y_coordinate,f"Relative trace number {order_index+1}",fontsize=6,color=YetiColors.YETI_WHITE,label="trace_description")

		# Change Order Number
		if(self.axes_spectrum.texts):
			order_type = f"Relative order"
			if( order_number_calibrated ):
				order_type = f"Absolute order"
			self.axes_spectrum_text.set_text(f"{order_type}: {order_number}")

		"""" Experimental - Dynamic masking"""
		dynamic_masking_matrix = np.ones(self.CurrentSpectrogram.shape)
		dynamic_masking_matrix[extracted_rows, extracted_columns] = 1.75
		
		masked_data = self.CurrentSpectrogram.data * dynamic_masking_matrix
		self.spectrogram_plot.set_data(masked_data)
		#### REMOVE ##### self.spectrogram_plot_cal.set_data(dynamic_masking_matrix)
		""""""

		self.draw_idle()

	def update_spectrum(self, order_index: int) -> int:
		"""
		Update the spectrum according to the requested `order_index`

		#### Parameters:
			`order_index` (int): Requested order_index from GUI

		#### Returns:
			`int`: Retruns the currently active order/trace index
		"""		
		if( Spectrogram.order_list != []):	
			# Limit order index to the size of the order list entries
			order_index = int(np.clip(order_index, 0, len(Spectrogram.order_list) - 1))

			self.active_order_index = order_index
			self.plot_spectrum(self.active_order_index)

		else:
			QtYetiLogger(QT_YETI.ERROR,f"No Order Information loaded.")
		
		return int(self.active_order_index)

	def find_nearest_order_index(self, evt_x:int, evt_y:int) -> int:
		"""
		Click on any position within the canvas and get back the nearest order index

		#### Parameters:
			`evt_x` (int): x coordinate of mouse click as integer
			`evt_y` (int): y coordinate of mouse click as integer

		#### Returns:
			`int`: Returns the nearest order index to click position on the canvas
		"""
		if (Spectrogram.order_list):
			y_list = []
			for order in Spectrogram.order_list:
				fit_parameters = order.fit_parameters
				y_list.append( echelle_order_fit_function(evt_x, *fit_parameters))
			nearest_index = np.argmin(np.abs( np.asarray(y_list)-evt_y ))
			return nearest_index

		else:
			QtYetiLogger(QT_YETI.ERROR,"Nearest Order not found. Returning np.NAN.")
			return np.NAN

	def trigger_order_extraction( self, extraction_mode:str ) -> None:
		"""
		Trigger order extraction
		### Details
		This method extracts either one selected order into a FITs file or loops over all orders to generate one file per order.
		#### Parameters:
			`extraction_mode` (str): Extraction mode: `"single"` for extracting the current order and `"all"` for looping over all orders
		"""
		# Check if orders are available
		if(Spectrogram.order_list):
			
			echelle_order_spectrum_to_fits(self.CurrentSpectrogram,\
				extraction_mode=extraction_mode, summation_method=self.summation_method,\
				order_index=self.active_order_index, single_trace_mode=False \
			)

		else:
			QtYetiLogger(QT_YETI.ERROR,f"No Fit Coefficients / Orders loaded.")

# Tab for MainWindow
class TabSpectrometer(QWidget):
	def __init__(self, parent):
		super(TabSpectrometer, self).__init__(parent)

		# Create Matplotlib Canvas
		self.figure_canvas = SpectrometerCanvas(parent=self)

		# Setup and customize
		self.setupTabStructure()
		self.customizeTab()
		self.connect_slots()
		
		# for child in self.findChildren((QPushButton, QSpinBox)):
		# 	child.setFocusPolicy(Qt.NoFocus)
		# self.setFocusPolicy(Qt.ClickFocus)
		# self.setFocus(Qt.NoFocusReason)
		# self.activateWindow()
		self.setFocusPolicy(Qt.StrongFocus)
		self.setFocus()

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

#		self.control_panel_layout.setSpacing(0)

		# Intensity spinbox widget
		self.intensity_control = QWidget()
		self.intensity_control_layout = QHBoxLayout()
		self.intensity_control_layout.setContentsMargins(0,0,0,0)
		self.intensity_control.setLayout( self.intensity_control_layout )

		# Active order box
		self.current_order_box = QWidget()
		self.current_order_box.setLayout(QHBoxLayout())
		self.current_order_spinbox = YetiSpinBox()

		#### HACK ####
		self.current_order_spinbox_value = self.current_order_spinbox.value()

		self.current_order_box.layout().addWidget(self.current_order_spinbox)
		self.current_order_box.layout().addWidget(QLabel("Current Trace/Order (relative)"))
		self.current_order_box.layout().setContentsMargins(0,0,0,0)

		# Create Intensity Spinboxes
		self.intensity_max = YetiSpinBox()
		self.intensity_min = YetiSpinBox()
		self.log_scale_chkbx = QCheckBox("Log Scale")
		self.intensity_control_layout.addWidget(self.intensity_max)
		self.intensity_control_layout.addWidget(QLabel("Intensity maximum"))
		self.intensity_control_layout.addWidget(self.intensity_min)
		self.intensity_control_layout.addWidget(QLabel("Intensity minimum"))
		self.intensity_control_layout.addWidget(self.log_scale_chkbx)

		# ItemBoxes
		self.summation_method_box = QComboBox()
		self.summation_method_box.addItems(QT_YETI.SUMMATIONS)
		self.summation_method_box.currentTextChanged.connect(self.gui_update_summation)
		# self.summation_method_box.setEditable(True)
		# self.summation_method_box.lineEdit().setAlignment(Qt.AlignCenter)
		# self.summation_method_box.lineEdit().setReadOnly(True)

		# Create Buttons
		self.action_load_spectrogram_btn	= QPushButton(text="Load Spectrogram", clicked=self.gui_load_spectrogram_file)
		self.action_load_coefficients_btn	= QPushButton(text="Load Order Information", clicked=self.gui_load_order_information)

		self.action_save_currentorder_btn	= QPushButton(text="Save Current Order", clicked=self.gui_save_current_order_to_fit)
		self.action_save_allorders_btn		= QPushButton(text="Save All Orders", clicked=self.gui_save_all_orders_to_fit)

		self.control_panel_layout.addWidget(self.intensity_control,0,0)
		self.control_panel_layout.addWidget(self.current_order_box,0,1)
		self.control_panel_layout.addWidget(self.action_load_spectrogram_btn,0,2)
		self.control_panel_layout.addWidget(self.action_load_coefficients_btn,0,3)

		self.control_panel_layout.addWidget(self.summation_method_box,0,4)
		self.control_panel_layout.addWidget(self.action_save_currentorder_btn,0,5)
		self.control_panel_layout.addWidget(self.action_save_allorders_btn,0,6)


		# Widths and limits
		self.intensity_max.setValue(np.int32(self.figure_canvas.CurrentSpectrogram.intmax))
		self.intensity_min.setValue(np.int32(self.figure_canvas.CurrentSpectrogram.intmin))

	def connect_slots(self):
		### ### ### Connect signals/slots ### ### ###
		self.intensity_max.valueChanged.connect(self.gui_intensity_changed)
		self.intensity_min.valueChanged.connect(self.gui_intensity_changed)
		self.log_scale_chkbx.stateChanged.connect(self.gui_log_scale_changed)

		self.current_order_spinbox.editingFinished.connect(self.gui_set_order_index)
		self.current_order_spinbox.valueChanged.connect(self.gui_set_order_index)

		# Signal connected at QPushButton creation. Example below:
		# self.action_load_spectrogram_btn.clicked.connect(self.gui_load_spectrogram_file)

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
		print(self.figure_canvas.CurrentSpectrogram)

	@pyqtSlot()
	def gui_update_spectrum(self):
		QtYetiLogger(QT_YETI.WARNING,"gui_update_spectrum() triggered. No action.")
		pass

	@pyqtSlot()
	def gui_intensity_changed(self):
		max = self.intensity_max.value()
		min = self.intensity_min.value()
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
			#self.figure_canvas.axes_flat_rows.set_ylim(y_min, y_max)
			self.figure_canvas.axes_spectrum.set_xlim(x_min, x_max)
			self.figure_canvas.draw_idle()

	@pyqtSlot()
	def gui_load_order_information(self):
		caption="Select Order Information File"
		initial_filter="Order Information Files (*.txt)"
		file_filter="Order Information Files (*.txt);; All files (*.*)"
		requested_filename, _  = QFileDialog.getOpenFileName(self, caption = caption, initialFilter=initial_filter, filter=file_filter)	
		if(requested_filename != ""):
			self.figure_canvas.load_order_information(requested_filename)

	@pyqtSlot()
	def gui_save_current_order_to_fit(self):
		# One file or two files? Depending on Image Slicer. Can FIT file handle 2 spectra?
		QtYetiLogger(QT_YETI.WARNING,f"gui_save_current_order_to_fit(\"single\") triggered.")
		self.figure_canvas.trigger_order_extraction("single")
		
	@pyqtSlot()
	def gui_save_all_orders_to_fit(self):
		QtYetiLogger(QT_YETI.WARNING,f"gui_save_current_order_to_fit(\"all\") triggered.")
		self.figure_canvas.trigger_order_extraction("all")

	@pyqtSlot()
	def gui_set_order_index(self):
		#### HACK ####
		previous_value = self.current_order_spinbox_value
		new_value = self.current_order_spinbox.value()
		active_order_number = self.figure_canvas.active_order_index + 1

		delta = new_value - previous_value

		if(active_order_number != previous_value):
			new_value = active_order_number + delta

		self.current_order_spinbox_value = self.figure_canvas.update_spectrum(new_value -1) +1

		self.current_order_spinbox.setValue( self.current_order_spinbox_value )

	def gui_update_summation(self, new_item_string: str):
		"""
		Update summation method for data extraction.
		#### Parameters:
			`new_item_string` (str): String that was chosen in the GUI is sent into this slot as parameter
		"""
		self.figure_canvas.summation_method = QT_YETI.SUMMATIONS[new_item_string]

class SpectrometerSettings(QWidget):

	def __init__(self, widget_parent: TabSpectrometer):
		super().__init__(parent=None)
		self.widget_parent = widget_parent
		self.resize(QT_YETI.TRACER_WINDOW_WIDTH, QT_YETI.TRACER_WINDOW_HEIGHT)
		self.setWindowFlags(Qt.WindowStaysOnTopHint)
		# for item in widget_parent.findChildren((QSpinBox, QObject)):
		# 	print(item)
		self.lay = QVBoxLayout()
		self.setLayout(self.lay)
		
		#if (widget_parent != None):
		# self.tscheild = 0
		# for child in self.widget_parent.findChildren(QSpinBox):
		# 	print(child)
		# 	self.tscheild = child
		# 	self.lay.addWidget(child)
		
	def closeEvent(self,event):
		# self.widget_parent.spectrometer_tab_action_layout.addWidget(self.tscheild)
		super().closeEvent(event)

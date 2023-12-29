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

		self.navigationbar = None
		self.navigationbar = NavigationToolbar2QT(self, parent=None)

		# Setup sample spectrogram		
		self.CurrentSpectrogram = Spectrogram("QtYeti.Sample")

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
		self.axes_spectrogram.set_ylabel("$m · \lambda(X,Y)$")
		self.axes_spectrum.set_xlabel("$\lambda(X,Y)$")
		self.axes_spectrum.set_ylabel("Counts (arb. u.)")

		self.spectrogram_plot = self.axes_spectrogram.imshow(self.CurrentSpectrogram.data,\
			vmin=self.CurrentSpectrogram.intmin,\
			vmax=self.CurrentSpectrogram.intmax,\
			#cmap = 'gray',\
			interpolation='none',\
			extent=[0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1],\
			aspect='auto',\
			label = "2D_Spectrogram")
		[self.spectrum_plot] = self.axes_spectrum.plot(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[256,:], linewidth=0.3, label="Data_Spectrum")
		self.axes_spectrum.set_xlim( self.CurrentSpectrogram.xrange.min(), self.CurrentSpectrogram.xrange.max() )
		self.axes_spectrum.set_ylim( self.CurrentSpectrogram.intmin, self.CurrentSpectrogram.intmax )

		[self.order_poly_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.72,linewidth="0.5",color=YetiColors.RED, label="Order_Poly_Plot")
		[self.background_poly_plot] = self.axes_spectrogram.plot(0,0,alpha=0.72,linewidth="0.5",color=YetiColors.YELLOW, label="Background_Poly_Plot")
		[self.sliced_poly_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.72,linewidth="0.5",color=YetiColors.ORANGE, label="Sliced_Poly_Plot")

		# Event handling
		self.mpl_connect('button_release_event', self.canvas_key_or_mouse_event)
		self.mpl_connect('key_press_event', self.canvas_key_or_mouse_event)

	### Navigation Bar ###
	def return_navigation_bar(self):
		#self.navigationbar = NavigationToolbar2QT(self, parent=None)
		return self.navigationbar

	### MPL Callbacks ###
	def canvas_key_or_mouse_event(self, event):

		if(event.inaxes is not self.axes_spectrogram):
			return
		
		evt_x = np.int32(np.rint(event.xdata))
		evt_y = np.int32(np.rint(event.ydata))

		QtYetiLogger(QT_YETI.MESSAGE,f"Nearest order number {self.find_nearest_order_index(evt_x, evt_y) + 1}")

		self.draw_idle()
		pass

	# Plotting
	def load_spectrogram(self, requested_filename = ""):
		if(requested_filename == ""):
			QtYetiLogger(QT_YETI.ERROR,"No file name provided.")
			return
		
		# Update CurrentSpectrogram
		int_min, int_max = self.CurrentSpectrogram.update_spectrogram(requested_filename)
		QtYetiLogger(QT_YETI.MESSAGE,f"{requested_filename} loaded.")

		## Plot spectrogram
		# https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively
		self.spectrogram_plot.set_data(self.CurrentSpectrogram.data)
		self.spectrogram_plot.set_extent([0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1])
		self.spectrogram_plot.set_clim(int_min, int_max)

		self.spectrum_plot.set_data(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[np.uint32(self.CurrentSpectrogram.ysize/2),:])
		# self.spectrum_plot.axes.set_ylim([0, int_max])
		self.axes_spectrum.set_xlim([0, self.CurrentSpectrogram.xsize-1])
		self.axes_spectrum.set_ylim([0, 1.1*self.CurrentSpectrogram.data.max()])

		self.draw_idle()
		return int_min, int_max

	def load_order_fit_coefficients(self, requested_filename = ""):
		
		self.order_fit_coefficients_list = Spectrogram.load_order_fit_coefficients(requested_filename)
		QtYetiLogger(QT_YETI.ERROR, "======================================================================================================")
		QtYetiLogger(QT_YETI.ERROR, "Here, we need to implement something, that creates the orders in a nice way with bounded x_ranges etc.", True)
		QtYetiLogger(QT_YETI.ERROR, "======================================================================================================")
		
		quit()

	def plot_spectrum(self, order_index):
		# Extract fit coefficients

		#current_xrange = self.CurrentSpectrogram.xrange
		""" Experimental """
		current_xrange = self.CurrentSpectrogram.order_list[order_index].x_range
		current_params = self.CurrentSpectrogram.order_list[order_index].fit_parameters
		"""##############"""
		#current_params = np.asarray(Spectrogram.order_fit_coefficients[order_index])[3:]
		
		# Create fit polinomial per order
		fitted_polynomial = np.asarray(echelle_order_fit_function(current_xrange, *current_params))

		# An order can leave the frame of the camer early. We accont for that by shortening the x and y arrays.
		clip_condition = (fitted_polynomial >= 0) & (fitted_polynomial < self.CurrentSpectrogram.ysize)
		self.current_xrange = np.asarray(current_xrange[clip_condition])
		fitted_polynomial = fitted_polynomial[clip_condition]

		# Discretize and convert to row indices from (x,y) to (r,c)
		discretized_polynomial = np.clip( np.rint(fitted_polynomial), 0, self.CurrentSpectrogram.ysize)
		# discretized_polynomial = discretized_polynomial.astype(np.uint32) # row() takes care of this
		discretized_rows = row(discretized_polynomial, self.CurrentSpectrogram.ysize)

		matricized_rows = np.asanyarray([discretized_rows-2,discretized_rows-1,discretized_rows,discretized_rows+1,discretized_rows+2])

		self.current_spectral_data = np.asarray(self.CurrentSpectrogram.data[discretized_rows,self.current_xrange])

		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows+1,self.current_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows+2,self.current_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows+3,self.current_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows-1,self.current_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows-2,self.current_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows-3,self.current_xrange])
		#current_spectral_data = np.sum(self.CurrentSpectrogram.data[matricized_rows,current_xrange],axis=0)

		## Experiment
		# data_tuple,summation_tuple = echelle_order_spectrum_extraction(self.CurrentSpectrogram, order_index, OrderTracerSettings())
		# summation_offsets = summation_tuple[0].repeat(self.CurrentSpectrogram.xsize)
		
		# dynamic_masking_matrix = np.full(self.CurrentSpectrogram.shape,0.4)
		# dynamic_masking_columns = np.tile(current_xrange, summation_tuple[1])
		# dynamic_masking_rows = np.tile(discretized_rows, summation_tuple[1])
		# dynamic_masking_matrix[dynamic_masking_rows+7 + summation_offsets, dynamic_masking_columns] = 1
		# dynamic_masking_matrix[dynamic_masking_rows+7 - summation_offsets, dynamic_masking_columns] = 1
		
		# masked_data = self.CurrentSpectrogram.data * dynamic_masking_matrix


		# self.spectrogram_plot.set_data(masked_data)
		# self.spectrum_plot.set_data(current_xrange,data_tuple[0])

		self.order_poly_plot.set_data(self.current_xrange, fitted_polynomial)
		# if( QT_YETI.SPECTROMETER_HAS_IMAGE_SLICER ):
		# 	self.background_poly_plot.set_data(current_xrange, fitted_polynomial + QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX/2)
		# 	self.sliced_poly_plot.set_data(current_xrange, fitted_polynomial + QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX)

		self.spectrum_plot.set_data(self.current_xrange, self.current_spectral_data)
		self.axes_spectrum.set_ylim([0, 1.1 * self.current_spectral_data.max()])
		
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
			#for i in np.arange(0,len(orders)-1):
			for idx,i in enumerate(orders):
				params = coefficients[idx]
				y_fit = echelle_order_fit_function(x, *params)
				ylist.append(y_fit)

			nearest_index = np.argmin(np.abs( np.asarray(ylist)-y ))

			return nearest_index

		else:
			QtYetiLogger(QT_YETI.ERROR,"Nearest Order not found. Returning np.NAN.")
			return np.NAN

# Tab for MainWindow
class TabSpectrometer(QWidget):
	def __init__(self, parent):
		super(TabSpectrometer, self).__init__(parent)

		# Create Matplotlib Canvas
		self.figure_canvas = SpectrometerCanvas(parent=self)

		# Setup and customize
		self.setupTabStructure()
		self.customizeTab()
		
		for child in self.findChildren((QWidget, QPushButton, QSpinBox)):
			child.setFocusPolicy(Qt.NoFocus)
		self.setFocusPolicy(Qt.NoFocus)

		self.order_fit_coefficients_list = []

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
		
		#self.control_panel_layout.setSpacing(0)

		# Active order box
		self.current_order_box = QWidget()
		self.current_order_box.setLayout(QHBoxLayout())
		self.current_order_spinbox = YetiSpinBox()
		self.current_order_box.layout().addWidget(self.current_order_spinbox)
		self.current_order_box.layout().addWidget(QLabel("Current Order (relative)"))
		self.current_order_box.layout().setContentsMargins(0,0,0,0)
		self.log_scale_chkbx = QCheckBox("Log Scale")

		# Create Buttons
		self.action_load_spectrogram_btn	= QPushButton("Load Spectrogram")
		self.action_load_coefficients_btn	= QPushButton("Load Fit Coefficients")
		self.action_save_currentorder_btn	= QPushButton("Save Current Order")
		self.action_save_allorders_btn		= QPushButton("Save All Orders")

		self.control_panel_layout.addWidget(self.log_scale_chkbx,0,0)
		self.control_panel_layout.addWidget(self.current_order_box,0,1)
		self.control_panel_layout.addWidget(self.action_load_spectrogram_btn,0,2)
		self.control_panel_layout.addWidget(self.action_load_coefficients_btn,0,3)
		self.control_panel_layout.addWidget(self.action_save_currentorder_btn,0,4)
		self.control_panel_layout.addWidget(self.action_save_allorders_btn,0,5)

		### ### ### Connect signals/slots ### ### ###
		self.current_order_spinbox.editingFinished.connect(self.gui_set_order_index)
		self.current_order_spinbox.valueChanged.connect(self.gui_set_order_index)
		self.log_scale_chkbx.stateChanged.connect(self.gui_log_scale_changed)

		self.action_load_spectrogram_btn.clicked.connect(self.gui_load_spectrogram_file)
		self.action_load_coefficients_btn.clicked.connect(self.gui_load_order_fit_coefficients)
		
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

	@pyqtSlot()
	def gui_update_spectrum(self):
		QtYetiLogger(QT_YETI.WARNING,"gui_update_spectrum() triggered. No action.")
		pass

	@pyqtSlot()
	def gui_log_scale_changed(self):
		# Toggle Log scale in plots
		int_max = self.intensity_max.value()
		int_min = self.intensity_min.value()
		new_normalization = -1

		if(self.log_scale_chkbx.checkState() != 0):
			scale_type = "log"
			if(np.int32(np.rint(int_min)) <= 0):
				int_min = 1e-1
			new_normalization = matplotlib.colors.LogNorm(vmin=int_min, vmax=int_max)
		else:
			scale_type = "linear"
			new_normalization = matplotlib.colors.Normalize(vmin=int_min, vmax=int_max)

		self.figure_canvas.axes_flat_rows.set_xscale(scale_type)
		self.figure_canvas.axes_flat_cols.set_yscale(scale_type)
		self.figure_canvas.spectrogram_plot.set_norm(new_normalization)

		self.figure_canvas.draw_idle()

	@pyqtSlot()
	def gui_load_order_fit_coefficients(self):
		caption="Select Fit Coefficient File"
		initial_filter="Order Fit Coefficient Files (*.txt)"
		file_filter="Order Fit Coefficient Files (*.txt);; All files (*.*)"
		requested_filename, _  = QFileDialog.getOpenFileName(self, caption = caption, initialFilter=initial_filter, filter=file_filter)	

		if(requested_filename != ""):
			self.figure_canvas.load_order_fit_coefficients(requested_filename)

	@pyqtSlot()
	def gui_save_single_order_to_fit(self):
		# One file or two files? Depending on Image Slicer. Can FIT file handle 2 spectra?
		pass
		FileSaver(self.figure_canvas.CurrentSpectrogram.filename+".txt", "Pixel, Spectrum",[[i,j] for i,j in zip(self.figure_canvas.current_xrange, self.figure_canvas.current_spectral_data)])
		QtYetiLogger(QT_YETI.WARNING,"gui_save_single_order_to_fit() triggered. No action.")

	@pyqtSlot()
	def gui_save_all_orders_to_fit(self):
		pass
		QtYetiLogger(QT_YETI.WARNING,"gui_save_all_orders_to_fit() triggered. No action.")

	@pyqtSlot()
	def gui_set_order_index(self):
		self.current_order_spinbox.setValue( self.figure_canvas.update_spectrum( self.current_order_spinbox.value() ) )

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

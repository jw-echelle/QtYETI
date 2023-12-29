from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

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

class CalibratorCanvas( FigureCanvasQTAgg ):

	def __init__(self, parent=None, width=QT_YETI.MATPLOTLIB_CANVAS_WIDTH, height=QT_YETI.MATPLOTLIB_CANVAS_HEIGHT, dpi=QT_YETI.MATPLOTLIB_DPI):
		self.control_figure = plt.figure(figsize=(width, height), dpi=dpi)
		super(CalibratorCanvas, self).__init__(self.control_figure)

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

		# Text
		self.axes_spectrum_text = self.axes_spectrum.text(transform=self.axes_spectrum.transAxes, ha='left', va='top', x=0.005, y=0.98, s=f"No data loaded.", label="texts_current_order")

		self.spectrogram_plot = self.axes_spectrogram.imshow(self.CurrentSpectrogram.data,\
			vmin=self.CurrentSpectrogram.intmin,\
			vmax=self.CurrentSpectrogram.intmax,\
			cmap = 'afmhot',\
			interpolation='none',\
			extent=[0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1],\
			aspect='auto',\
			label = "2D_Spectrogram")
		[self.spectrum_plot] = self.axes_spectrum.plot(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[self.CurrentSpectrogram.ysize//2,:], linewidth=0.3, label="Data_Spectrum")
		self.axes_spectrum.set_xlim( self.CurrentSpectrogram.xrange.min(), self.CurrentSpectrogram.xrange.max() )
		self.axes_spectrum.set_ylim( self.CurrentSpectrogram.intmin, self.CurrentSpectrogram.intmax )

		""" Experimental """
		self.spectrogram_plot_cal = self.axes_spectrogram.imshow(self.CurrentSpectrogram.data,\
			vmin=0,\
			vmax=100000,\
			cmap = 'gist_ncar',\
			alpha = 0.25, \
			interpolation='none',\
			extent=[0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1],\
			aspect='auto',\
			label = "2D_Spectrogram_Cal")
		[self.calibration_help_plot] = self.axes_spectrum.plot(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[self.CurrentSpectrogram.ysize//2,:], linewidth=0.3, label="Data_Spectrum_Calibration")
		""""""

		[self.order_poly_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.72,linewidth="0.75",color=YetiColors.MIDAS_GREEN, label="Order_Poly_Plot")
		[self.background_poly_plot] = self.axes_spectrogram.plot(0,0,alpha=0.72,linewidth="0.5",color=YetiColors.YELLOW, label="Background_Poly_Plot")
		[self.sliced_poly_plot] =  self.axes_spectrogram.plot(0,0,alpha=0.72,linewidth="0.5",color=YetiColors.ORANGE, label="Sliced_Poly_Plot")
		
		# Event handling
		self.mpl_connect('button_release_event', self.canvas_key_or_mouse_event)
		self.mpl_connect('key_press_event', self.canvas_key_or_mouse_event)
		self.mpl_connect("scroll_event", self.canvas_scroll_event)

	### Navigation Bar ###
	def return_navigation_bar(self):
		#self.navigationbar = NavigationToolbar2QT(self, parent=None)
		return self.navigationbar

	### MPL Callbacks ###
	def canvas_key_or_mouse_event(self, event):
		if(event.inaxes == self.axes_spectrogram):
			evt_x = np.int32(np.rint(event.xdata))
			evt_y = np.int32(np.rint(event.ydata))
			QtYetiLogger(QT_YETI.MESSAGE,f"Nearest order index {self.find_nearest_order_index(evt_x, evt_y) + 1}")
			self.draw_idle()
		pass

	def canvas_scroll_event(self, event):
		if(event.inaxes == self.axes_spectrogram):
			print(event)
			print(type(event)) # MouseEvent
			print(event.button, event.step)
			new_index = int(np.clip(self.order_index + int(event.step),a_min=1,a_max=len(self.CurrentSpectrogram.order_fit_coefficients)-1))
			self.order_index = new_index
			_ = self.update_spectrum(self.order_index)

	# Plotting
	def load_spectrogram(self, requested_filename = ""):
		if(requested_filename == ""):
			QtYetiLogger(-1,"No file name provided.")
			return
		
		QtYetiLogger(QT_YETI.MESSAGE,f"{requested_filename} loaded.")

		# Update CurrentSpectrogram
		int_min, int_max = self.CurrentSpectrogram.update_spectrogram(requested_filename)


		## Plot spectrogram
		# https://stackoverflow.com/questions/17835302/how-to-update-matplotlibs-imshow-window-interactively
		#self.spectrogram_plot.set_data(self.CurrentSpectrogram.data)
		self.spectrogram_plot.set_data(self.CurrentSpectrogram.data)
		self.spectrogram_plot.set_extent([0, self.CurrentSpectrogram.xsize-1, 0, self.CurrentSpectrogram.ysize-1])
		self.spectrogram_plot.set_clim(int_min, int_max)

		self.spectrum_plot.set_data(self.CurrentSpectrogram.xrange, self.CurrentSpectrogram.data[np.uint32(self.CurrentSpectrogram.ysize/2),:])
		# self.spectrum_plot.axes.set_ylim([0, int_max])
		self.axes_spectrum.set_xlim([0, self.CurrentSpectrogram.xsize-1])
		self.axes_spectrum.set_ylim([0, 1.1*self.CurrentSpectrogram.data.max()])

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
		""" Set new scale type: choose between Linear and Logarithmic. """
#######
######## Check the type of objects
######## if isinstance(image, mpimg.AxesImage):
######## 	print("This is an AxesImage")
######## if isinstance(lines, list) and all(isinstance(line, plt.Line2D) for line in lines):
######## 	print("This is a list of Lines2D")
#######
		
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
			#self.axes_spectrum.set_xscale(new_scale_type)
			
			self.scale_type = new_scale_type
			self.spectrogram_plot.set_norm(new_normalization)
			self.axes_spectrum.set_yscale(new_scale_type)
			self.draw_idle()

	def load_order_fit_coefficients(self, requested_filename = ""):
		self.CurrentSpectrogram.load_order_fit_coefficients(requested_filename)

	def plot_spectrum(self, order_index: int) -> None:
		""" Plot the spectrum along a fitted order
		-----------------------------------------
		Parameters:
			order_index (int): relative order index
		"""		

		""" Experimental """
		current_order = self.CurrentSpectrogram.order_list[order_index]
		current_order_xrange = current_order.x_range
		current_order_params = current_order.fit_parameters
		current_order_number = current_order.number_m
		"""##############"""

		
		# Create fit polinomial per order
		fitted_polynomial = np.asarray(echelle_order_fit_function(current_order_xrange, *current_order_params))

		# # Discretize and convert to row indices from (x,y) to (r,c)
		# discretized_polynomial = np.clip( np.rint(fitted_polynomial), 0, self.CurrentSpectrogram.ysize)
		# # discretized_polynomial = discretized_polynomial.astype(np.uint32) # row() takes care of this
		# discretized_rows = row(discretized_polynomial, self.CurrentSpectrogram.ysize)
		discretized_rows = row(fitted_polynomial, self.CurrentSpectrogram.ysize)

		matricized_rows = np.asarray([discretized_rows-2,discretized_rows-1,discretized_rows,discretized_rows+1,discretized_rows+2])

		self.current_spectral_data = np.asarray(self.CurrentSpectrogram.data[discretized_rows,current_order_xrange])

		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows+1,current_order_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows+2,current_order_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows+3,current_order_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows-1,current_order_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows-2,current_order_xrange])
		self.current_spectral_data +=np.asarray(self.CurrentSpectrogram.data[discretized_rows-3,current_order_xrange])
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

		self.order_poly_plot.set_data(current_order_xrange, fitted_polynomial)
		# if( QT_YETI.SPECTROMETER_HAS_IMAGE_SLICER ):
		# 	self.background_poly_plot.set_data(current_order_xrange, fitted_polynomial + QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX/2)
		# 	self.sliced_poly_plot.set_data(current_order_xrange, fitted_polynomial + QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX)

		if(self.axes_spectrum.texts):
			order_type = f"Relative order"
			if( current_order.order_number_calibtrated ):
				order_type = f"Absolute order"
			self.axes_spectrum_text.set_text(f"{order_type}: {current_order_number}")

		self.spectrum_plot.set_data(current_order_xrange, self.current_spectral_data)
		self.axes_spectrum.set_ylim([0, 1.1 * self.current_spectral_data.max()])
		
		""" EXPERIMENTAL """
		#self.calibration_help_plot.set_data(current_order_xrange,self.calibration_help_matrix[discretized_rows,current_order_xrange])

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
		if ( not Spectrogram.order_fit_coefficients):
			orders_and_coefficients = np.asarray(Spectrogram.order_fit_coefficients)
			orders = orders_and_coefficients[:,0]
			coefficients = orders_and_coefficients[:,3:]

			ylist = []
			#for i in np.arange(0,len(orders)-1):
			for index,_ in enumerate(orders):
				params = coefficients[index]
				y_fit = echelle_order_fit_function(x, *params)
				ylist.append(y_fit)
			nearest_index = np.argmin(np.abs( np.asarray(ylist)-y ))

			return nearest_index

		else:
			QtYetiLogger(QT_YETI.ERROR,"Nearest Order not found. Returning np.NAN.")
			return np.NAN

	# Geometrical Calibration
	def grating_calculation(self,grating_alpha,grating_gamma):
		"""_summary_

		Args:
			grating_alpha (_type_): _description_
			grating_gamma (_type_): _description_

		Returns:
			_type_: _description_
		"""

		# Get the current simulated catalog points for the current set of geometric spectrometer parameters
		simulated_points = simulate_spectrometer(draw=False, α=grating_alpha, γ=grating_gamma)
		simulated_points_xyi = np.asarray([[point.x, point.y, point.intensity] for point in simulated_points])
		
		# Update the plots and clear previous points and annotations
		delete_mpl_plt_object_by_label(self.axes_spectrogram.lines, "calibrationdot_")
		delete_mpl_plt_object_by_label(self.axes_spectrogram.texts, "calibrationdot_")

		# Generate for every catalogue point a kindof realistic echellogram
	
		def spot_shape(R,r,w):
			"""_summary_

			Args:
				R (_type_): _description_
				r (_type_): _description_
				w (_type_): _description_

			Returns:
				_type_: _description_
			"""
			x,y = R
			x0,y0 = r
			s = w/(2*np.sqrt(2 * np.log(2))) # FWHM to Sigma
			s = w/4 # 2 sigma level for full width
			r = np.sqrt( (x-x0)**2 + (y-y0)**2 )
			return np.exp(-0.5*(r/s)**2)
			
		
		# # Approach: 
		# # Create a matrix that is zero everywhere except at the x,y or r,c positions of the simulated peaks.
		# # Insert a simulated peak at every (r,c) position weighted with the amplitude found in the ThAr Catalogue on nist.gov
		# SIM_SPOT_SIZE = QT_YETI.DETECTOR_SPOT_SIZE_PX/2
		# NOISE_LEVEL = 10
		# AMPLIFICATION = 10000
		# simulated_spot_matrix_size = int(SIM_SPOT_SIZE*8+1)
		# simulated_spot_center = simulated_spot_matrix_size // 2
	
		# simulated_spot_matrix = np.zeros((simulated_spot_matrix_size,simulated_spot_matrix_size))
		# sim_spot_xy = np.arange(0,simulated_spot_matrix_size,1)
		# X,Y = np.meshgrid(sim_spot_xy,sim_spot_xy)
		# simulated_spot_matrix = spot_shape((X,Y), (simulated_spot_center,simulated_spot_center), SIM_SPOT_SIZE)
		# random_noise_matrix = np.abs( NOISE_LEVEL * np.random.rand( simulated_spot_matrix_size,simulated_spot_matrix_size ))
		
		# self.calibration_help_matrix = np.zeros(self.CurrentSpectrogram.shape)
		# aux_calibration_submatrix = np.zeros(simulated_spot_matrix.shape)

		# # Filter out cases where the simulated point is too close to the image boundary 
		# lower_edge_cases = np.logical_and(simulated_points_xyi[:,0] > simulated_spot_center, simulated_points_xyi[:,1] > simulated_spot_center)
		# upper_edge_cases = np.logical_and(simulated_points_xyi[:,0] < self.CurrentSpectrogram.xsize - simulated_spot_center, simulated_points_xyi[:,1] < self.CurrentSpectrogram.ysize - simulated_spot_center)
		# mask = np.logical_and(lower_edge_cases, upper_edge_cases)

		# filtered_points = simulated_points_xyi[mask]

		# for coordinate in filtered_points:
		# 	r,c,intensity = row(coordinate[1], self.CurrentSpectrogram.ysize),col(coordinate[0],self.CurrentSpectrogram.xsize),coordinate[2]

		# 	row_slice_of_interest = slice(r - simulated_spot_center, r+1 + simulated_spot_center)
		# 	col_slice_of_interest = slice(c - simulated_spot_center, c+1 + simulated_spot_center)

		# 	aux_calibration_submatrix = self.calibration_help_matrix[row_slice_of_interest, col_slice_of_interest]
		# 	self.calibration_help_matrix[row_slice_of_interest, col_slice_of_interest] = AMPLIFICATION * (intensity * simulated_spot_matrix) + random_noise_matrix + aux_calibration_submatrix + 6000
		# 	aux_calibration_submatrix = np.zeros(simulated_spot_matrix.shape)

		imgslcr_offset = QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX

		""" EXPERIMENTAL """
		#self.spectrogram_plot_cal.set_data(self.calibration_help_matrix)
		""""""

		for point in simulated_points:
			self.axes_spectrogram.plot(point.x,point.y,"<",fillstyle="none", markeredgewidth=0.5, label="calibrationdot_", markersize=5, color="#00AA00")
			#self.axes_spectrogram.plot(point.x,point.y+imgslcr_offset,">",fillstyle="none", markeredgewidth=0.5, label="calibrationdot_", markersize=5, color="#00AA00")
			self.axes_spectrogram.text(point.x+10,point.y,f"m = {int(point.order):02d}", fontsize=7,color="#00AA00", label="calibrationdot_text",alpha=0.8)
			self.axes_spectrogram.text(point.x+10,point.y-10,f"λ = {point.wavelength:02.3f}", fontsize=5,color="#DDDDDD", label="calibrationdot_text",alpha=0.8)

		self.draw_idle()

	def set_absolute_order_numbers(self):
		self.CurrentSpectrogram.set_absolute_order_number_m(-21)
		QtYetiLogger(QT_YETI.MESSAGE,f"Printing list of orders",True)
		for order in self.CurrentSpectrogram.get_order_list():
			print(order.number_m)
		
# Gemetric Calibrator
class GeometricCalibratorWindow(QWidget):
	"""
	Use spectrometer gemetry parameters to determin orders and facilitate calibration
	"""
	def __init__(self, MPLCanvas: CalibratorCanvas):
		super(GeometricCalibratorWindow, self).__init__(parent=None)

		self.setWindowTitle(QT_YETI.CALIBRATOR_WINDOW_NAME)
		self.resize(QT_YETI.CALIBRATOR_WINDOW_WIDTH, QT_YETI.CALIBRATOR_WINDOW_HEIGHT)
		self.setWindowFlags(Qt.WindowStaysOnTopHint)
		self.setWindowIcon(QIcon(QT_YETI.IMAGE_PATH))

		self.setup_calibrator_window()

		self.canvas = MPLCanvas

	def setup_calibrator_window(self):
		# Introducing a delay between user input and any acutal parameter changes
		# in case someone is scrolling too heavily
		self.timer = QTimer()
		self.timer.timeout.connect(self.delayed_onValueChange)
		self.value_change_sender_object = 0

		self.layout = QGridLayout()
		self.setLayout(self.layout)

		# Auto-recalc
		# Button update

		self.labels = [\
			QLabel(f"Grating α"),\
			QLabel(f"Grating β"),\
			QLabel(f"Grating γ"),\
			QLabel(f"Prism roll"),\
			QLabel(f"Prism pitch"),\
			QLabel(f"Prism yaw"),\
			QLabel(f"Camera roll"),\
			QLabel(f"Camera pitch"),\
			QLabel(f"Camera yaw")\
		]
		self.sliders = [\
			QSlider(valueChanged = self.onValueChange, objectName = "slider_grating_alpha"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_grating_beta"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_grating_gamma"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_prism_roll"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_prism_pitch"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_prism_yaw"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_camera_roll"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_camera_pitch"),
			QSlider(valueChanged = self.onValueChange, objectName = "slider_camera_yaw")
		]

		self.indicators = [\
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_grating_alpha"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_grating_beta"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_grating_gamma"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_prism_roll"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_prism_pitch"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_prism_yaw"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_camera_roll"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_camera_pitch"),
			YetiDoubleSpinBox(valueChanged = self.onValueChange, objectName = "indicator_camera_yaw")
		]

		# Add to layout
		label_row = 0
		slider_row = 1
		indicator_row = 2
		for idx, slider in enumerate(self.sliders):
			slider.setMaximum(100)
			slider.setMinimum(-100)
			slider.setTickPosition(QSlider.TickPosition.TicksRight)
			self.layout.addWidget(slider, slider_row, idx)
		
		for idx, indicator in enumerate(self.indicators):
			indicator.setSingleStep(0.01)
			self.layout.addWidget(indicator, indicator_row, idx)

		for idx, label in enumerate(self.labels):
			self.layout.addWidget( label, label_row, idx )

	@pyqtSlot()
	def onValueChange(self):
		# Dirty hack to save the initial sender of the singal
		self.value_change_sender_object = self.sender()
		self.timer.start(250)

	def delayed_onValueChange(self):

		self.timer.stop()
		QtYetiLogger(QT_YETI.MESSAGE, f"Timer has been stopped", True)
		QtYetiLogger(QT_YETI.WARNING, f"Is there really no input delay in QT?", False)

		sender = self.value_change_sender_object
		sender_name = sender.objectName()
		sender_value = sender.value()
		print(sender_value)

		if( "slider_" in sender_name):
			receiver = f"indicator_{sender_name[7:]}"
			for ind in self.indicators:
				if(receiver in ind.objectName()):
					with QSignalBlocker(ind):
						ind.setValue(int(sender_value))
	
		elif( "indicator_" in sender_name):
			receiver = f"slider_{sender_name[10:]}"
			for sli in self.sliders:
				if(receiver in sli.objectName()):
					with QSignalBlocker(sli):
						sli.setValue(int(sender_value))
		else:
			return

		self.canvas.grating_calculation(QT_YETI.SPECTROMETER_GRATING_INCIDENT_INPLANE_ANGLE_DEG + float(sender_value*0.01),0)

	def update_beam_directions(self):
		"""
		Recalulate all lightrays based on new geometric inputs.
		"""
		
		pass

	def update_detector_image(self):
		"""
		Update the projection basis vectors {Lx,Ly,Lz} and recalulate the detector image
		"""
		pass
		
	def show(self):
		super(GeometricCalibratorWindow, self).show()
	
	def closeEvent(self, event) -> None:
		super(GeometricCalibratorWindow, self).closeEvent(event)

# Tab for MainWindow
class TabCalibrator(QWidget):
	def __init__(self, parent):
		super(TabCalibrator, self).__init__(parent)

		# Create Matplotlib Canvas
		self.figure_canvas = CalibratorCanvas(parent=self)

		# Setup and customize
		self.setupTabStructure()
		self.customizeTab()
		
		# for child in self.findChildren((QWidget, QPushButton, QSpinBox)):
		# 	child.setFocusPolicy(Qt.NoFocus)
		# self.setFocusPolicy(Qt.NoFocus)
		self.setFocusPolicy(Qt.StrongFocus)
		self.setFocus()

		self.figure_canvas.draw()

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
		
		# Intensity spinbox widget
		self.intensity_control = QWidget()
		self.intensity_control_layout = QHBoxLayout()
		self.intensity_control_layout.setContentsMargins(0,0,0,0)
		self.intensity_control.setLayout( self.intensity_control_layout )

		# Active order box
		self.current_order_box = QWidget()
		self.current_order_box.setLayout(QHBoxLayout())
		self.current_order_spinbox = YetiSpinBox()
		self.current_order_box.layout().addWidget(self.current_order_spinbox)
		self.current_order_box.layout().addWidget(QLabel("Current Order (relative)"))
		self.current_order_box.layout().setContentsMargins(0,0,0,0)
		
		# Create Spinboxes
		self.intensity_max = YetiSpinBox()
		self.intensity_min = YetiSpinBox()
		self.log_scale_chkbx = QCheckBox("Log Scale")
		self.intensity_control_layout.addWidget(self.intensity_max)
		self.intensity_control_layout.addWidget(QLabel("Intensity maximum"))
		self.intensity_control_layout.addWidget(self.intensity_min)
		self.intensity_control_layout.addWidget(QLabel("Intensity minimum"))
		self.intensity_control_layout.addWidget(self.log_scale_chkbx)

		# Create Buttons
		self.action_load_spectrogram_btn	= QPushButton(text="Load Spectrogram", clicked=self.gui_load_spectrogram_file)
		self.action_load_coefficients_btn	= QPushButton(text="Load Fit Coefficients", clicked=self.gui_load_order_fit_coefficients)
		self.action_load_geo_calibrator_btn	= QPushButton(text="Geometric Calibrator", clicked=self.gui_open_geometric_calibrator)
		self.action_save_currentorder_btn	= QPushButton(text="Save Current Order", clicked=self.gui_save_current_order_to_fit)
		self.action_save_allorders_btn		= QPushButton(text="Save All Orders", clicked=self.gui_save_all_orders_to_fit)

		self.control_panel_layout.addWidget(self.intensity_control,0,0)
		self.control_panel_layout.addWidget(self.current_order_box,0,1)
		self.control_panel_layout.addWidget(self.action_load_spectrogram_btn,0,2)
		self.control_panel_layout.addWidget(self.action_load_coefficients_btn,0,3)
		self.control_panel_layout.addWidget(self.action_load_geo_calibrator_btn,0,4)
		self.control_panel_layout.addWidget(self.action_save_currentorder_btn,0,5)
		self.control_panel_layout.addWidget(self.action_save_allorders_btn,0,6)

		# Add QTableWidget & Rearrange
		self.calibrator_list = QTableWidget(32,3)

		self.calibrator_widget = QWidget()
		self.calibrator_widget.setContentsMargins(0,0,0,0)
		self.calibrator_widget_layout = QHBoxLayout()
		self.calibrator_widget_layout.setContentsMargins(0,0,0,0)
		self.calibrator_widget.setLayout(self.calibrator_widget_layout)
		
		self.tab_layout.replaceWidget(self.figure_canvas,self.calibrator_widget)
		self.calibrator_widget_layout.addWidget(self.figure_canvas)
		self.calibrator_widget_layout.addWidget(self.calibrator_list)
		self.calibrator_widget_layout.setStretchFactor(self.figure_canvas,3)
		self.calibrator_widget_layout.setStretchFactor(self.calibrator_list,1)

		self.calibrator_list.setHorizontalHeaderLabels(['Order','Pixel','Wavelength (Å)'])
		self.calibrator_list.setAlternatingRowColors(True)
		self.calibrator_list.verticalHeader().setVisible(False)

		# Widths and limits
		self.intensity_max.setValue(np.int32(self.figure_canvas.CurrentSpectrogram.intmax))
		self.intensity_min.setValue(np.int32(self.figure_canvas.CurrentSpectrogram.intmin))

		### ### ### Connect signals/slots ### ### ###
		self.intensity_max.valueChanged.connect(self.gui_intensity_changed)
		self.intensity_min.valueChanged.connect(self.gui_intensity_changed)
		self.log_scale_chkbx.stateChanged.connect(self.gui_log_scale_changed)

		self.current_order_spinbox.editingFinished.connect(self.gui_set_order_index)
		self.current_order_spinbox.valueChanged.connect(self.gui_set_order_index)

		# Signal connected at QPushButton creation
		# self.action_load_spectrogram_btn.clicked.connect(self.gui_load_spectrogram_file)
		# self.action_load_coefficients_btn.clicked.connect(self.gui_load_order_fit_coefficients)
		# self.action_save_currentorder_btn.clicked.connect(self.gui_save_current_order_to_fit)
		# self.action_save_allorders_btn.clicked.connect(self.gui_save_all_orders_to_fit)

		# Instance of Geometric Calibrator Tool
		self.CalibratorTool = GeometricCalibratorWindow(self.figure_canvas)
	
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
		QtYetiLogger(QT_YETI.WARNING,"gui_update_spectrum() triggered. No action.")
		pass

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

	def gui_load_order_fit_coefficients(self):
		caption="Select Fit Coefficient File"
		initial_filter="Order Fit Coefficient Files (*.txt)"
		file_filter="Order Fit Coefficient Files (*.txt);; All files (*.*)"
		requested_filename, _  = QFileDialog.getOpenFileName(self, caption = caption, initialFilter=initial_filter, filter=file_filter)	

		if(requested_filename != ""):
			self.figure_canvas.load_order_fit_coefficients(requested_filename)
		
	def gui_open_geometric_calibrator(self):
		self.CalibratorTool.show()

	def gui_save_current_order_to_fit(self):
		# One file or two files? Depending on Image Slicer. Can FIT file handle 2 spectra?
		pass
		QtYetiLogger(1,"gui_save_single_order_to_fit() triggered. No action.")

	def gui_save_all_orders_to_fit(self):
		pass
		QtYetiLogger(1,"gui_save_all_orders_to_fit() triggered. No action.")
	
	def gui_set_order_index(self):
		self.current_order_spinbox.setValue( self.figure_canvas.update_spectrum( self.current_order_spinbox.value() ) )
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from dataclasses import dataclass

import sys
from os import environ

#from qt_yeti.qt_yeti_general import *
#from qt_yeti.qt_yeti_functions import *
#from qt_yeti.qt_yeti_hardware_settings_tab import *

import numpy as np
import scipy
import csv
import configparser # https://docs.python.org/3/library/configparser.html
import time
from typing import List,Tuple

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")
plt.rcParams['axes.linewidth'] = 0.5 # Alternative: plt.setp(self.axes_flat_field.spines.values(), linewidth=0.5)
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['font.size']=7
plt.rcParams['font.family']='sans-serif'

###############################################################################################################################

def qt_yeti_suppress_qt_warnings():
	environ["QT_DEVICE_PIXEL_RATIO"] = "0"
	environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
	environ["QT_SCREEN_SCALE_FACTORS"] = "1"
	environ["QT_SCALE_FACTOR"] = "1"

class YetiColors:
	BLUE="#1864AB"
	RED="#B92A2A"
	GREEN="#2B8A3E"
	GREENISH = "#0B7285"
	ORANGE="#E67700"
	YELLOW="#DDDD00"
	MIDAS_GREEN="#00AA00"
	YETI_GREY = "#DDDDDD"
	YETI_WHITE = "#AAAAAA"

# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
# Check also module: colorama
class ShellColors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'

def elapsed_time(timed_function: object):
	def wrapper(*args, **kwargs):
		start_time = time.perf_counter()
		return_value = timed_function(*args, **kwargs)
		end_time = time.perf_counter()

		tmd_func_str = timed_function.__str__()
		idx = tmd_func_str.find(" at 0x")
		tmd_func_str = tmd_func_str[10:idx]
		
		dt = end_time-start_time
		dt_color = ""
		if ((0.3 <= dt) and (dt <= 0.5)):
			dt_color = ShellColors.WARNING
		elif(dt < 0.3):
			dt_color = ShellColors.OKGREEN
		else:
			dt_color = ShellColors.FAIL

		timing_string = f"\t{ShellColors.OKCYAN}Elapsed time of {tmd_func_str}(): {ShellColors.BOLD}{dt_color}{(dt):.6f} s.{ShellColors.ENDC}"
		print(timing_string)
		
		return return_value
	return wrapper

###############################################################################################################################

QT_YETI = 0

class QtYetiSettings:
	_instance = None

	WINDOW_NAME = "QtYETI - Yeti\'s Extra-Terrestrial Investigations."
	WINDOW_WIDTH = 1345
	WINDOW_HEIGHT = 800

	TRACER_WINDOW_NAME = "QtYETI - Order Tracer Settings"
	TRACER_WINDOW_WIDTH = 256
	TRACER_WINDOW_HEIGHT = 256

	CALIBRATOR_WINDOW_NAME = "QtYETI - Geometric Calibrator"
	CALIBRATOR_WINDOW_WIDTH = 512
	CALIBRATOR_WINDOW_HEIGHT = 256

	MATPLOTLIB_DPI = 128
	MATPLOTLIB_XY_LIM_SCALE = 1.2
	MATPLOTLIB_CANVAS_WIDTH = 10 # times 100 or dpi
	MATPLOTLIB_CANVAS_HEIGHT = 8 # times 100 or dpi

	SPIN_BOX_MIN_WIDTH = 80
	SPIN_BOX_MAX_WIDTH = 256

	IMAGE_PATH = "./qt_yeti/yeti.png"
	SETTINGS_INI_PATH = "./qt_yeti_settings.ini"

	MESSAGE = 0
	WARNING = 1
	ERROR = -1

	ANNOTATION_X_COORDINATE = 42

	SUMMATIONS = {\
		"Extraction: Simple Sum":"simple_sum",\
		"Extraction: On Trace Center":"order_center_pixel",\
		"Extraction: Square-root weighted":"sqrt_weighted_sum",\
		"Extraction: Optimal Extraction":"optimal"\
	}

	TRACING_MODES = {\
		"Maximum": False,\
		"Fitted": True\
	}

	func_echelle_fit_function = 0

	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)

		return cls._instance
	
	def __init__(self):
			self.readHardwareConfig()
			self.initializeTracerSettings()
			self.ReferenceData=QtYetiReferenceSpectrograms()

	def readHardwareConfig(self):
		try:
			config = configparser.ConfigParser()
			config.optionxform=str
			config.read(self.SETTINGS_INI_PATH)

			self.SPECTROMETER_GRATING_DENSITY_PER_MM = config.getint("HARDWARE.Spectrometer","SpectrometerGratingDensityLpMM")
			self.SPECTROMETER_GRATING_ANGLE_DEG = config.getfloat("HARDWARE.Spectrometer","SpectrometerGratingAngleDeg")
			self.SPECTROMETER_GRATING_INCIDENT_INPLANE_ANGLE_DEG = config.getfloat("HARDWARE.Spectrometer","SpectrometerGratingInplaneAngleDeg")
			self.SPECTROMETER_GRATING_INCIDENT_OUTOFPLANE_ANGLE_DEG = config.getfloat("HARDWARE.Spectrometer","SpectrometerGratingOutofplaneAngleDeg")
			self.SPECTROMETER_GRATING_OUTGOING_ANGLE_DEG = config.getfloat("HARDWARE.Spectrometer","SpectrometerGratingOutgoingAngleDeg")

			self.SPECTROMETER_HAS_IMAGE_SLICER = config.getboolean("HARDWARE.Spectrometer","SpectrometerHasImageslicer")
			self.SPECTROMETER_IMAGE_SLICER_TRACES_PER_ORDER = config.getint("HARDWARE.Spectrometer","SpectrometerImageslicerTracesPerOrder")
			self.SPECTROMETER_IMAGE_SLICER_TRACE_SUFFIX = config.getfloat("HARDWARE.Spectrometer","SpectrometerImageslicerTraceSuffix")
			self.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX = config.getint("HARDWARE.Spectrometer","SpectrometerImageslicerSeparationPixels")

			self.DETECTOR_FOCAL_LENGTH_MM = config.getfloat("HARDWARE.Detector","DetectorFocalLengthMM")
			self.DETECTOR_INPLANE_ANGLE_DEG = config.getfloat("HARDWARE.Detector","DetectorInplaneAngleDeg")
			self.DETECTOR_OUTOFPLANE_ANGLE_DEG = config.getfloat("HARDWARE.Detector","DetectorOutofplaneAngleDeg")
			self.DETECTOR_X_PIXELS = config.getint("HARDWARE.Detector","DetectorXPixels")
			self.DETECTOR_Y_PIXELS = config.getint("HARDWARE.Detector","DetectorYPixels")
			self.DETECTOR_PIXEL_SIZE_UM = config.getfloat("HARDWARE.Detector","DetectorPixelSizeUM")
			self.DETECTOR_PIXEL_BINNING = config.getint("HARDWARE.Detector","DetectorPixelBinning")
			self.DETECTOR_SPOT_SIZE_PX = config.getint("HARDWARE.Detector","DetectorSpotSizePixels")
			self.DETECTOR_BIT_DEPTH = config.getint("HARDWARE.Detector","DetectorBitDepth")
			self.DETECTOR_MAX_INTENSITY = np.power(2,self.DETECTOR_BIT_DEPTH)-1
			self.DETECTOR_ORDER_NUMBER_MAGNITUDE_INCREASE_DIRECTION = config.get("HARDWARE.Detector","DetectorOrderNumberMagnitudeIncreaseDirection")

			del config

			#### DEBUG_NOTE ####
			#print(ShellColors.OKGREEN+ f"\r\nSucessfully read QtYetiSettings from File." + ShellColors.ENDC)
			return
		
		except Exception as error:
			print(ShellColors.FAIL+ f"\r\nError while reading QtYetiSettings(). Result: {error}" + ShellColors.ENDC)
			return

	def writeHardwareConfig(self):
		try:
			config = configparser.ConfigParser()
			config.optionxform=str
			config.read(self.SETTINGS_INI_PATH)

			config.set("HARDWARE.Spectrometer","SpectrometerGratingDensityLpMM",f"{self.SPECTROMETER_GRATING_DENSITY_PER_MM}")
			config.set("HARDWARE.Spectrometer","SpectrometerGratingAngleDeg",f"{self.SPECTROMETER_GRATING_ANGLE_DEG}")
			config.set("HARDWARE.Spectrometer","SpectrometerGratingInplaneAngleDeg",f"{self.SPECTROMETER_GRATING_INCIDENT_INPLANE_ANGLE_DEG}")
			config.set("HARDWARE.Spectrometer","SpectrometerGratingOutofplaneAngleDeg",f"{self.SPECTROMETER_GRATING_INCIDENT_OUTOFPLANE_ANGLE_DEG}")
			config.set("HARDWARE.Spectrometer","SpectrometerGratingOutgoingAngleDeg",f"{self.SPECTROMETER_GRATING_OUTGOING_ANGLE_DEG}")

			config.set("HARDWARE.Spectrometer","SpectrometerHasImageslicer",f"{self.SPECTROMETER_HAS_IMAGE_SLICER}")
			config.set("HARDWARE.Spectrometer","SpectrometerImageslicerTracesPerOrder",f"{self.SPECTROMETER_IMAGE_SLICER_TRACES_PER_ORDER}")
			config.set("HARDWARE.Spectrometer","SpectrometerImageslicerTraceSuffix",f"{self.SPECTROMETER_IMAGE_SLICER_TRACE_SUFFIX}")
			config.set("HARDWARE.Spectrometer","SpectrometerImageslicerSeparationPixels",f"{self.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX}")

			config.set("HARDWARE.Detector","DetectorFocalLengthMM",f"{self.DETECTOR_FOCAL_LENGTH_MM}")
			config.set("HARDWARE.Detector","DetectorInplaneAngleDeg",f"{self.DETECTOR_INPLANE_ANGLE_DEG}")
			config.set("HARDWARE.Detector","DetectorOutofplaneAngleDeg",f"{self.DETECTOR_OUTOFPLANE_ANGLE_DEG}")
			config.set("HARDWARE.Detector","DetectorXPixels",f"{self.DETECTOR_X_PIXELS}")
			config.set("HARDWARE.Detector","DetectorYPixels",f"{self.DETECTOR_Y_PIXELS}")
			config.set("HARDWARE.Detector","DetectorPixelSizeUM",f"{self.DETECTOR_PIXEL_SIZE_UM}")
			config.set("HARDWARE.Detector","DetectorPixelBinning",f"{self.DETECTOR_PIXEL_BINNING}")
			config.set("HARDWARE.Detector","DetectorSpotSizePixels",f"{self.DETECTOR_SPOT_SIZE_PX}")
			config.set("HARDWARE.Detector","DetectorBitDepth",f"{self.DETECTOR_BIT_DEPTH}")
			config.set("HARDWARE.Detector","DetectorOrderNumberMagnitudeIncreaseDirection",f"{self.DETECTOR_ORDER_NUMBER_MAGNITUDE_INCREASE_DIRECTION}")
			
			with open(QT_YETI.SETTINGS_INI_PATH, 'w') as configfile:
				config.write(configfile)
			del config

			self.TracerSettings.readTracerConfig()
			print(ShellColors.OKGREEN+ f"Hardware Config saved" + ShellColors.ENDC)

			return
		
		except Exception as error:
			print(ShellColors.FAIL+ f"Error while writing QtYetiSettings() to file. Result: {error}" + ShellColors.ENDC)
			return

	def initializeTracerSettings(self):
		self.TracerSettings = QtYetiTracerSettings("from_file", self.SETTINGS_INI_PATH)

	def _calculate_summation_range(self) -> np.ndarray:

		odd_spot_size = int(2* (self.DETECTOR_SPOT_SIZE_PX // 2) +1)

		relative_summation_range = np.arange(-odd_spot_size//2, +odd_spot_size//2 +1)

		if( self.SPECTROMETER_HAS_IMAGE_SLICER == True):
			# With an imageslicer, we want to sum up N times the image slicer peak separation, with an offset by half a spot
			relative_summation_range = np.arange(-odd_spot_size//2,((self.SPECTROMETER_IMAGE_SLICER_TRACES_PER_ORDER - 1) * self.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX) + odd_spot_size//2 +1)
			
		if(  QT_YETI.DETECTOR_ORDER_NUMBER_MAGNITUDE_INCREASE_DIRECTION == "up"):
			relative_summation_range = -relative_summation_range

		return relative_summation_range

#%%
class QtYetiTracerSettings:
	"""
	Class for keeping all necessary tracer settings mostly for the tracing tab.
	"""	
	def __init__(self, origin: str = None, settings_path = ""):
		self.settings_path = settings_path

		self.effective_slit_height = 1
		self.readTracerConfig()


	def readTracerConfig(self):
		try:
			config = configparser.ConfigParser()
			config.optionxform=str
			config.read(self.settings_path)

			self.first_absolute_order =			config.getint("TRACER","TracerFirstAbsoluteOrder")
			self.abs_order_number_m_direction = config.get("HARDWARE.Detector","DetectorOrderNumberMagnitudeIncreaseDirection").lower()

			self.spotsize_px = 					config.getint("HARDWARE.Detector","DetectorSpotSizePixels")
			self.image_slicer = 				config.getboolean("HARDWARE.Spectrometer","SpectrometerHasImageslicer")
			self.image_slicer_traces_per_order= config.getint("HARDWARE.Spectrometer","SpectrometerImageslicerTracesPerOrder")
			self.image_slicer_separation_px = 	config.getint("HARDWARE.Spectrometer","SpectrometerImageslicerSeparationPixels")

			self.fit_function_poly_order =		config.getint("TRACER","TracerFitFunctionPolynomialOrder")
			self.fit_function_use_x_offset =	config.getboolean("TRACER","TracerFitFunctionUseXOffset")

			self.distance_to_image_edge_px = 	config.getint("TRACER","TracerDistanceToImageEdgePx")
			self.samples_per_order = 			config.getint("TRACER","TracerSamplesPerOrder")

			self.peak_distance_px = 			config.getint("TRACER","TracerPeakDistancePx")
			self.peak_height = 					config.getfloat("TRACER","TracerPeakHeight")
			self.peak_prominence = 				config.getfloat("TRACER","TracerPeakProminence")
			self.peak_width_px = 				config.getint("TRACER","TracerPeakWidthPx")

			self.smoothing_stiffness = 			config.getfloat("TRACER","TracerSmoothingStiffness")
			self.smoothing_order = 				config.getint("TRACER","TracerSmoothingOrder")

			self.effective_slit_height = self.get_effective_slit_height()

			del config

			#### DEBUG_NOTE ####
			# print(ShellColors.OKGREEN+ f"→ Sucessfully read TracerSettings from File." + ShellColors.ENDC)
			return

		except Exception as error:
			print(ShellColors.FAIL+ f"→ Error while reading TracerSettings in QtYetiTracerSettings(). Result: {error}" + ShellColors.ENDC)

			self.first_absolute_order =			1
			self.abs_order_number_m_direction = f"up"

			self.fit_function_poly_order =		2
			self.fit_function_use_x_offset =	True

			self.distance_to_image_edge_px = 	0
			self.samples_per_order = 			0

			self.peak_distance_px = 			0
			self.peak_height = 					0.0
			self.peak_prominence = 				0.0
			self.peak_width_px = 				0

			self.smoothing_stiffness = 			0.0
			self.smoothing_order = 				0

			self.spotsize_px = 					1
			self.image_slicer = 				False
			self.image_slicer_separation_px = 	1

	def saveTracerConfig(self):
		config = configparser.ConfigParser()
		config.optionxform=str#
		config.read(self.settings_path)

		config.set("TRACER","TracerFirstAbsoluteOrder",f"{self.first_absolute_order}")
		config.set("HARDWARE.Detector","DetectorOrderNumberMagnitudeIncreaseDirection",f"{self.abs_order_number_m_direction}")

		config.set("HARDWARE.Detector","DetectorSpotSizePixels",f"{self.spotsize_px}")
		config.set("HARDWARE.Spectrometer","SpectrometerHasImageslicer",f"{self.image_slicer}")
		config.set("HARDWARE.Spectrometer","SpectrometerImageslicerSeparationPixels",f"{self.image_slicer_separation_px}")

		config.set("TRACER","TracerFitFunctionPolynomialOrder",f"{self.fit_function_poly_order}")
		config.set("TRACER","TracerFitFunctionUseXOffset",f"{self.fit_function_use_x_offset}")

		config.set("TRACER","TracerDistanceToImageEdgePx",f"{self.distance_to_image_edge_px}")
		config.set("TRACER","TracerSamplesPerOrder",f"{self.samples_per_order}")

		config.set("TRACER","TracerPeakDistancePx",f"{self.peak_distance_px}")
		config.set("TRACER","TracerPeakHeight",f"{self.peak_height}")
		config.set("TRACER","TracerPeakProminence",f"{self.peak_prominence}")
		config.set("TRACER","TracerPeakWidthPx",f"{self.peak_width_px}")

		config.set("TRACER","TracerSmoothingStiffness",f"{self.smoothing_stiffness}")
		config.set("TRACER","TracerSmoothingOrder",f"{self.smoothing_order}")

		with open(self.settings_path, 'w') as configfile:
			config.write(configfile)
		del config

		# Update QT_YETI.SETTINGS
		QT_YETI.readHardwareConfig()
		self.readTracerConfig()

	def get_effective_slit_height(self) -> int:

		odd_spot_size = int(2 * (self.spotsize_px//2) + 1)
		effective_slit_height = odd_spot_size

		if( self.image_slicer == True):
			effective_slit_height = odd_spot_size + (self.image_slicer_traces_per_order - 1) * self.image_slicer_separation_px

		return effective_slit_height

#%%
class QtYetiReferenceSpectrograms:
	_instance = None
	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)

			cls.create_spectrograms(cls)

		return cls._instance

	def create_spectrograms(self):
		self.Flatfield = -1
		self.Darkfield = -1
		self.Sciencedata = -1

""" Initialize Class for settings """
QT_YETI = QtYetiSettings()

#%%
class YetiCheckBox(QCheckBox):
	def __init__(self, *args, **kwargs):
		super(YetiCheckBox,self).__init__(*args, **kwargs)
		self.setChecked(False)

class YetiSpinBox(QSpinBox):
	def __init__(self, *args, **kwargs):
		super(YetiSpinBox,self).__init__(*args, **kwargs)
		self.setKeyboardTracking(False)
		self.setMaximum((2**31)-1)
		self.setMinimum(-2**31)
		self.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)

class YetiComboBox(QComboBox):
	def __init__(self, *args, **kwargs):
		super(YetiComboBox,self).__init__(*args, **kwargs)
		self.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)

class YetiDoubleSpinBox( QDoubleSpinBox ):
	def __init__(self, *args, **kwargs):
		super(YetiDoubleSpinBox, self).__init__(*args, **kwargs)
		self.setMaximum((2.0**31)-1)
		self.setMinimum(-2.0**31)
		self.setMinimumWidth(QT_YETI.SPIN_BOX_MIN_WIDTH)

class YetiDoubleSlider( QSlider ):
	""" Slider * 100 """
	def __init__(self):
		super(YetiDoubleSlider, self).__init__()
		pass

class TabYeti(QWidget):	
	def __init__(self, parent):
		super(QWidget,self).__init__(parent)
		
		self.layout = QVBoxLayout(self)

		self.yetis_image = QLabel(self)
		self.yetis_image.setPixmap(QPixmap(QT_YETI.IMAGE_PATH))

		self.layout.setAlignment(Qt.AlignCenter)

		self.layout.addStretch()
		self.layout.addWidget(QLabel("<i>We lit the streets but we cut off the magesty of the heavens.</i> - Joe Rogan (2024)"))
		self.layout.addWidget(self.yetis_image)
		self.layout.addWidget(QLabel("Yeti is a cute hamster sleeping during daytime and investigating at night."))
		self.layout.addStretch()
		self.setLayout(self.layout)

##############################################################################
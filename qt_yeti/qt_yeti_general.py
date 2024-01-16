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

QT_YETI = 0
class QtYetiSettings:
	_instance = None
	def __new__(cls):
		if cls._instance is None:
			cls._instance = super().__new__(cls)
			
			cls.initialize_variables(cls)
			cls.readHardwareConfig(cls)
			cls.readTracerConfig(cls)

		return cls._instance

	def initialize_variables(self):
		self.WINDOW_NAME = "QtYETI - Yeti\'s Extra-Terrestrial Investigations."
		self.WINDOW_WIDTH = 1344
		self.WINDOW_HEIGHT = 800

		self.TRACER_WINDOW_NAME = "QtYETI - Order Tracer Settings"
		self.TRACER_WINDOW_WIDTH = 256
		self.TRACER_WINDOW_HEIGHT = 256

		self.CALIBRATOR_WINDOW_NAME = "QtYETI - Geometric Calibrator"
		self.CALIBRATOR_WINDOW_WIDTH = 512
		self.CALIBRATOR_WINDOW_HEIGHT = 256

		self.MATPLOTLIB_DPI = 128
		self.MATPLOTLIB_XY_LIM_SCALE = 1.2
		self.MATPLOTLIB_CANVAS_WIDTH = 10 # times 100 or dpi
		self.MATPLOTLIB_CANVAS_HEIGHT = 8 # times 100 or dpi

		self.SPIN_BOX_MIN_WIDTH = 80
		self.SPIN_BOX_MAX_WIDTH = 256

		self.IMAGE_PATH = "./qt_yeti/yeti.png"
		self.SETTINGS_INI_PATH = "./qt_yeti_settings.ini"

		self.MESSAGE = 0
		self.WARNING = 1
		self.ERROR = -1

		self.ANNOTATION_X_COORDINATE = 42

		self.SUMMATIONS = {\
			"Simple Sum":"simple_sum",\
			"On Trace Center":"order_center_pixel",\
			"Square-root weighted":"sqrt_weighted_sum",\
			"Optimal Extraction":"optimal"\
		}

		self.TRACING_MODES = {\
			"Maximum": False,\
			"Fitted": True\
		}

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
			return 0
		
		except BaseException as be:
			print(f"Caught exception: {be} in file: {__file__}")
			return -666

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

			return 0
		
		except BaseException as be:
			print(f"Caught exception: {be} in file: {__file__}")
			return -666

	def readTracerConfig(self):
		...
	
	def writeTracerConfig(self):
		...


""" Initialize Class for settings """
QT_YETI = QtYetiSettings()

#%%
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
	YETI_WHITE = "#DDDDDD"

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
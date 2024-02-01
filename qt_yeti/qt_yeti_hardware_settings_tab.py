from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from dataclasses import dataclass

import sys
import os

from PyQt5.QtWidgets import QWidget

from qt_yeti.qt_yeti_general import *
from qt_yeti.qt_yeti_functions import *
#from qt_yeti.qt_yeti_hardware_settings_tab import *

import numpy as np
import scipy
#%%
###############################################################################################################################

class TabHardwareSettings(QWidget):

	def __init__(self, parent):
		super(QWidget, self).__init__(parent)

		self.setupTabStructure()
		self.customizeTab()
		
		self.read_settings_data()

		self.FloatingWindow = None

		self.connect_slots()
		self.setFocusPolicy(Qt.StrongFocus)
		self.setFocus()

	def setupTabStructure(self):
		# Top Level Tab layout
		self.tab_layout = QHBoxLayout()
		self.setLayout(self.tab_layout)

		# Add Control Panel
		self.control_panel = QWidget()
		self.control_panel.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Fixed)
		self.tab_layout.addWidget(self.control_panel,1,Qt.AlignmentFlag.AlignLeft)

	def customizeTab(self):
		# Fill Control Panel
		self.control_panel_layout = QGridLayout()
		self.control_panel_layout.setContentsMargins(0,0,0,0)
		self.control_panel.setLayout(self.control_panel_layout)

		# Form with Spectrometer and Detector parameters
		self.hardware_form = QWidget()
		self.hardware_form_layout = QFormLayout()
		self.hardware_form_layout.setContentsMargins(0,0,0,0)
		self.hardware_form.setLayout( self.hardware_form_layout )

		self.control_panel_layout.addWidget( self.hardware_form )

		# Create Spinboxes / Buttons
		self.spectrometer_grating_density_per_mm = YetiSpinBox()
		self.spectrometer_grating_incident_inplane_angle_deg = YetiDoubleSpinBox()
		self.spectrometer_grating_incident_outofplane_angle_deg = YetiDoubleSpinBox()
		self.spectrometer_grating_outgoing_angle_deg = YetiDoubleSpinBox()
		self.spectrometer_has_image_slicer = QCheckBox()
		self.spectrometer_image_slicer_separation_px = YetiSpinBox()
		#
		self.detector_focal_length_mm = YetiDoubleSpinBox()
		self.detector_inplane_angle_deg = YetiDoubleSpinBox()
		self.detector_outofplane_angle_deg = YetiDoubleSpinBox()
		self.detector_x_pixels = YetiSpinBox()
		self.detector_y_pixels = YetiSpinBox()
		self.detector_pixel_size_um = YetiDoubleSpinBox()
		self.detector_pixel_binning = YetiSpinBox()
		self.detector_spot_size_px = YetiSpinBox()
		self.detector_bit_depth = YetiSpinBox()
		#
		self.action_read_button = QPushButton("&Read Config")
		self.action_save_button = QPushButton("&Save Config")
		self.action_float_button = QPushButton("&Open Floating Window")

		# Add to form
		lines = [QFrame(),QFrame(),QFrame(),QFrame()]
		for line in lines:
			line.setFrameShape(QFrame.HLine)
			line.setFrameShadow(QFrame.Sunken)

		self.hardware_form_layout.addRow(QLabel("Add canvas and draw the grating, lines and other stuff"))
		self.hardware_form_layout.addRow(self.spectrometer_grating_density_per_mm, QLabel("Spectrometer: Grating Density (L/mm)"))
		self.hardware_form_layout.addRow(self.spectrometer_grating_incident_inplane_angle_deg, QLabel("Spectrometer: Grating Inplane Incident Angle α (°)"))
		self.hardware_form_layout.addRow(self.spectrometer_grating_incident_outofplane_angle_deg, QLabel("Spectrometer: Grating Outofplane Incident Angle γ (°)"))
		self.hardware_form_layout.addRow(self.spectrometer_grating_outgoing_angle_deg, QLabel("Spectrometer: Grating Outgoing Angle β<sub>0</sub> (°)"))
		self.hardware_form_layout.addRow(self.spectrometer_has_image_slicer, QLabel("Spectrometer: Has Imageslicer"))
		self.hardware_form_layout.addRow(self.spectrometer_image_slicer_separation_px, QLabel("Spectrometer: Imageslicer Spot Separation (px)"))
		self.hardware_form_layout.addRow(lines[0])
		self.hardware_form_layout.addRow(self.detector_focal_length_mm, QLabel("Detector: Focal Length f<sub>cam</sub> (mm)"))
		self.hardware_form_layout.addRow(self.detector_inplane_angle_deg, QLabel("Detector: Inplane Angle Φ (°)"))
		self.hardware_form_layout.addRow(self.detector_outofplane_angle_deg, QLabel("Detector: Outofplane Angle Θ (°)"))
		self.hardware_form_layout.addRow(self.detector_x_pixels, QLabel("Detector: Horizontal Pixel Count"))
		self.hardware_form_layout.addRow(self.detector_y_pixels, QLabel("Detector: Vertical Pixel Count"))
		self.hardware_form_layout.addRow(self.detector_pixel_size_um, QLabel("Detector: Pixel Size (μm)"))
		self.hardware_form_layout.addRow(self.detector_pixel_binning, QLabel("Detector: Binning"))
		self.hardware_form_layout.addRow(self.detector_spot_size_px, QLabel("Detector: Light Source Spot Size (px)"))
		self.hardware_form_layout.addRow(lines[1])
		self.hardware_form_layout.addRow(self.action_read_button, self.action_save_button)
		self.hardware_form_layout.addRow(self.action_float_button)

	def connect_slots(self):
		# Signals / Slots
		self.action_read_button.clicked.connect(self.read_settings_data)
		self.action_save_button.clicked.connect(self.save_settings_data)
		self.action_float_button.clicked.connect(self.open_floating_window)

	def read_settings_data(self):
		try:
			QT_YETI.readHardwareConfig() # See below
			QtYetiLogger(QT_YETI.MESSAGE,f"Hardware settings read.", True)
			self.spectrometer_grating_density_per_mm.setValue(					QT_YETI.SPECTROMETER_GRATING_DENSITY_PER_MM)
			self.spectrometer_grating_incident_inplane_angle_deg.setValue(		QT_YETI.SPECTROMETER_GRATING_INCIDENT_INPLANE_ANGLE_DEG)
			self.spectrometer_grating_incident_outofplane_angle_deg.setValue(	QT_YETI.SPECTROMETER_GRATING_INCIDENT_OUTOFPLANE_ANGLE_DEG)
			self.spectrometer_grating_outgoing_angle_deg.setValue(				QT_YETI.SPECTROMETER_GRATING_OUTGOING_ANGLE_DEG)
			self.spectrometer_has_image_slicer.setChecked(						QT_YETI.SPECTROMETER_HAS_IMAGE_SLICER)
			self.spectrometer_image_slicer_separation_px.setValue(				QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX)
			self.detector_focal_length_mm.setValue(								QT_YETI.DETECTOR_FOCAL_LENGTH_MM)
			self.detector_inplane_angle_deg.setValue(							QT_YETI.DETECTOR_INPLANE_ANGLE_DEG)
			self.detector_outofplane_angle_deg.setValue(						QT_YETI.DETECTOR_OUTOFPLANE_ANGLE_DEG)
			self.detector_x_pixels.setValue(									QT_YETI.DETECTOR_X_PIXELS)
			self.detector_y_pixels.setValue(									QT_YETI.DETECTOR_Y_PIXELS)
			self.detector_pixel_size_um.setValue(								QT_YETI.DETECTOR_PIXEL_SIZE_UM)
			self.detector_pixel_binning.setValue(								QT_YETI.DETECTOR_PIXEL_BINNING)
			self.detector_spot_size_px.setValue(								QT_YETI.DETECTOR_SPOT_SIZE_PX)
			self.detector_bit_depth.setValue(									QT_YETI.DETECTOR_BIT_DEPTH)
		except ValueError:
			QtYetiLogger(QT_YETI.ERROR,f"Failed to read hardware settings. {ValueError.name}",True)


	def save_settings_data(self):
		QT_YETI.SPECTROMETER_GRATING_DENSITY_PER_MM = 					self.spectrometer_grating_density_per_mm.value()
		QT_YETI.SPECTROMETER_GRATING_INCIDENT_INPLANE_ANGLE_DEG = 		self.spectrometer_grating_incident_inplane_angle_deg.value()
		QT_YETI.SPECTROMETER_GRATING_INCIDENT_OUTOFPLANE_ANGLE_DEG = 	self.spectrometer_grating_incident_outofplane_angle_deg.value()
		QT_YETI.SPECTROMETER_GRATING_OUTGOING_ANGLE_DEG = 				self.spectrometer_grating_outgoing_angle_deg.value()
		QT_YETI.SPECTROMETER_HAS_IMAGE_SLICER = 						self.spectrometer_has_image_slicer.isChecked()
		QT_YETI.SPECTROMETER_IMAGE_SLICER_SEPARATION_PX = 				self.spectrometer_image_slicer_separation_px.value()
		QT_YETI.DETECTOR_FOCAL_LENGTH_MM = 								self.detector_focal_length_mm.value()
		QT_YETI.DETECTOR_INPLANE_ANGLE_DEG = 							self.detector_inplane_angle_deg.value()
		QT_YETI.DETECTOR_OUTOFPLANE_ANGLE_DEG = 						self.detector_outofplane_angle_deg.value()
		QT_YETI.DETECTOR_X_PIXELS = 									self.detector_x_pixels.value()
		QT_YETI.DETECTOR_Y_PIXELS = 									self.detector_y_pixels.value()
		QT_YETI.DETECTOR_PIXEL_SIZE_UM = 								self.detector_pixel_size_um.value()
		QT_YETI.DETECTOR_PIXEL_BINNING = 								self.detector_pixel_binning.value()
		QT_YETI.DETECTOR_SPOT_SIZE_PX = 								self.detector_spot_size_px.value()
		QT_YETI.DETECTOR_BIT_DEPTH = 									self.detector_bit_depth.value()

		try:
			QT_YETI.writeHardwareConfig()
			QtYetiLogger(QT_YETI.MESSAGE,f"Hardware settings saved.",True)

		except ValueError:
			QtYetiLogger(QT_YETI.ERROR,f"Failed to save hardware settings. {ValueError.name}",True)

	def open_floating_window(self):
		self.FloatingWindow = FloatingHardwareSettingsWindow(self, self.control_panel, self.tab_layout)

class FloatingHardwareSettingsWindow(QWidget):
	def __init__(self, parent: TabHardwareSettings, widget: QWidget, original_container_layout: QLayout) -> None:
		super(FloatingHardwareSettingsWindow, self).__init__(parent=None)
	
		self.setWindowTitle("Floating Hardware")
		#self.resize(400,800)
		#self.setWindowFlags(Qt.WindowStaysOnTopHint)
		self.setWindowIcon(QIcon(QT_YETI.IMAGE_PATH))

		self.window_layout = QVBoxLayout()
		self.setLayout(self.window_layout)

		self.parent = parent
		self.loaded_widget = widget
		self.original_container_layout = original_container_layout

		QtYetiLogger(QT_YETI.MESSAGE,f"Floating Hardware Settings Window opened")

		self.show()

	def __del__(self):
		QtYetiLogger(QT_YETI.MESSAGE,f"Floating Hardware Settings Window closed")
	
	def show(self):

		# Grab the widget

		self.window_layout.addWidget(self.loaded_widget)

		super(FloatingHardwareSettingsWindow, self).show()

	def closeEvent(self, event: QCloseEvent) -> None:
		# Give the widget back
		# self.original_container_layout.addWidget(self.loaded_widget)
		self.original_container_layout.addWidget(self.loaded_widget,1,Qt.AlignmentFlag.AlignLeft)
		self.parent.FloatingWindow = None
		super(FloatingHardwareSettingsWindow, self).closeEvent(event)
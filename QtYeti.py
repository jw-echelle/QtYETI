from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import sys

from astropy.io import fits
import scipy
import numpy as np
import csv

from qt_yeti.qt_yeti_general import *
from qt_yeti.qt_yeti_functions import *
from qt_yeti.qt_yeti_tracer_tab import TabOrderTracer
from qt_yeti.qt_yeti_calibrator_tab import TabCalibrator
from qt_yeti.qt_yeti_spectrometer_tab import TabSpectrometer
from qt_yeti.qt_yeti_hardware_settings_tab import TabHardwareSettings

qt_yeti_suppress_qt_warnings()

class MainWindow(QMainWindow):
	pass
	def __init__(self, *argv, **kwargs):
		super(MainWindow, self).__init__(*argv, **kwargs)

		self.setWindowTitle(QT_YETI.WINDOW_NAME)
		self.resize(QSize(QT_YETI.WINDOW_WIDTH, QT_YETI.WINDOW_HEIGHT))
		self.setWindowIcon(QIcon(QT_YETI.IMAGE_PATH))

		#self.setFocusPolicy(Qt.NoFocus)
		#self.setWindowOpacity(0.75)

		self.setCentralWidget( MainWindowTabControl(self) )
		self.menuBar().addMenu("&File").addAction("Load Tracer Coeffs")
		#self.show()
		self.statusBar().showMessage("Yeti Yeti Yeti | Jakob Wierzbowski 2022-2023 | Contact via <jw.echelle@outlook.com>")

class MainWindowTabControl(QWidget):
	def __init__(self, parent):
		super(MainWindowTabControl, self).__init__(parent)
		layout = QVBoxLayout(self)
		
		tab_control = QTabWidget()
		tab_control.setFocusPolicy(Qt.NoFocus)
		#.setFocusPolicy(Qt.NoFocus)
		#tab_control.setWindowOpacity(0.1)

		tracer_tab = TabOrderTracer(self)
		calibrator_tab = TabCalibrator(self)
		spectrometer_tab = TabSpectrometer(self)
		#spectrometer_tab = SpecificTab(self) # Prepared. Use when program flow clear
		hardware_tab =  TabHardwareSettings(self)
		yetis_tab = TabYeti(self)

		tab_control.addTab(tracer_tab,"Order Tracer")
		tab_control.addTab(calibrator_tab, "Calibration")
		tab_control.addTab(spectrometer_tab,"Spectrometer")
		tab_control.addTab(hardware_tab, "Hardware && Settings")
		tab_control.addTab(yetis_tab, "Who is Yeti?")

		layout.addWidget(tab_control)
		self.setLayout(layout)

		tab_control.setCurrentIndex(0)

def main():
	background_application = QApplication(sys.argv)
	foreground_window = MainWindow()
	foreground_window.show() # --> can be done within class constructor
	background_application.exec_()

if __name__ == '__main__':

	print("\r\nYeti Yeti Yeti\r\n")
	
	todo_list_message = f"\r\n\
		• Handle FITS files with multiple images in one file\r\n\
		• [done] How to deal with positive/negative absolute orders and their direction\r\n\
		• How should the order_index behave: top to bottom or bottom to top?\r\n\
		• Load ThAr into Flatfield TAB, click on ThAr Peak and give this order the absolute m number via a window\r\n\
		• Think of single spectrum viewer via bintables or other means\r\n\
		• call it trace or order?\r\n\
		• ConfigParser & SafeConfigParser - check deprecation\r\n\
		• Log button: ranges are not correct. IntMax/IntMin\r\n\
		• TracerSettings to be read into QT_YETI_Settings\r\n \
		• Remove Point class and use Spot class\r\n \
		• ...\r\n"

	QtYetiLogger(QT_YETI.WARNING,todo_list_message,False)
	main()
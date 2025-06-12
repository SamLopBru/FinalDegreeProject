from PyQt6 import uic
from PyQt6.QtWidgets import (QMainWindow, QApplication, QPushButton, QLabel, QFileDialog, QScrollBar, QMessageBox, QProgressBar, QSlider,  QRadioButton, QDialog, QGroupBox, QComboBox, 
QDialogButtonBox, QFrame, QSizePolicy)
from PyQt6.QtGui import QPixmap, QImage, QMovie, QPainter, QPen, QColor, QIcon
from PyQt6 import QtCore as Qt
from PIL import Image
import numpy as np
import sys
import pydicom as py
import preprocesamiento
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from sigmoideyLoss import finalMIP, draw_box
import pydicom
from pydicom.dataset import Dataset
from pydicom.uid import generate_uid
from pydicom.sequence import Sequence
import datetime
import torch
from torchvision import transforms
from model.faster import CustomFaster
from utilFuncs import samplingSlices, get_original_index, unresizePascalVOC


class ProcessingWorker(Qt.QThread): # procesing thread

    results_ready = Qt.pyqtSignal(object, object, object, int) 
    progress = Qt.pyqtSignal(int)


    def __init__(self, parent, slices, prepro_applicado, model):
        super().__init__(parent)
        self.slices = slices
        self.prepro_applicado = prepro_applicado
        self.model = model

    def prepararImage(self):

        tensors = []

        slices = samplingSlices(self.slices)
        for idx, slice in enumerate(slices):
            transoform = transforms.ToTensor()
            nuevo_tamano = (256, 256)
            slice = Image.fromarray(slice).convert("RGB")
            slice_redimensionada = slice.resize(nuevo_tamano, Image.LANCZOS)
            slice_redimensionada = transoform(slice_redimensionada)
            tensors.append(slice_redimensionada)

        return torch.stack(tensors)

    def run(self):
        
        boxes = []
        scores = []
        classification = []
        model = self.model

        slices_tensor = self.prepararImage()
        self.progress.emit(33)  
        slices_tensor = slices_tensor.unsqueeze(0)  # batch dimension
        model.eval()
        self.progress.emit(50)
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                output, logits = model(slices_tensor) 
                if output['boxes'].shape == (0,4):
                    boxes = None
                    classification = 0
                    scores = 1
                else:
                    boxes = unresizePascalVOC([box.tolist() for box in output['boxes']], (self.slices.shape[1],self.slices.shape[2]) , (256,256))
                    classification = output['labels'].item()
                    scores = output['scores'].item()
                self.progress.emit(85)  

            mri_unsampled = torch.argmax(logits, dim=1)
            mri = get_original_index(mri_unsampled, self.slices.shape[0])
            self.progress.emit(95) 

        self.results_ready.emit(boxes, classification, scores, mri)


class LoadImageThread(Qt.QThread):  # load image thread
    finished = Qt.pyqtSignal(tuple)

    def __init__(self, filename, parent):
        super().__init__()
        self.filename = filename
        self.parent = parent

    def run(self):
        try:
            
            dicom_data = py.dcmread(self.filename)
            slices = dicom_data.pixel_array
            if slices.shape[0]<27:
                self.finished.emit((None, None))
                self.parent.loading_gif.stop()

            image_data = [getattr(dicom_data, "RescaleIntercept", 0),  #take information from the dicom file
                          getattr(dicom_data, "RescaleSlope", 1), 
                          slices.shape[1], slices.shape[2],
                          getattr(dicom_data, "PatientID", "Unknown"),
                          getattr(dicom_data, "PatientName", "Unknown"),
                          getattr(dicom_data, "PatientSex", "Unknown"),
                          getattr(dicom_data,"Laterality","Unknown")]
            self.finished.emit((slices, image_data))
        
        except Exception as e:
            print(f"Error loading image: {e}")
            self.finished.emit((None, None)) 


class PreproThread(Qt.QThread):  # proprocessing thread
    finished = Qt.pyqtSignal(np.ndarray)
    progress = Qt.pyqtSignal(int)

    def __init__(self, slices):
        super().__init__()
        self.slices = slices

    def run(self):
        num_cores = cpu_count()
        with ThreadPool(processes=num_cores) as pool:
            processed_slices = []
            total_slices = len(self.slices)
            for i, result in enumerate(pool.imap_unordered(preprocesamiento.procesar_slice, self.slices)):
                processed_slices.append(result)
                progress = int((i + 1) / total_slices * 100)
                self.progress.emit(progress)

        result_array = np.stack(processed_slices, axis=0)
        self.finished.emit(result_array)


class MIPDialog(QDialog): # SM options dialog to adjust the SM parameters
    def __init__(self,  pixel_array, clahe, imageData , mri ,parent=None):
        super().__init__(parent)

        uic.loadUi("ventanaMIP.ui", self)
        self.setWindowTitle("SM Options")

        self.pixel_array = pixel_array
        self.clahe = clahe
        self.imageData = imageData
        self.parent = parent
        self.mri = mri

        #Define widgets
        self.slider_centerSlice = self.findChild(QSlider, "slider_centerSlice")
        self.slider_numSlices = self.findChild(QSlider, "slider_numSlices")
        self.slider_leveling = self.findChild(QSlider, "slider_leveling")
        self.slider_windowing = self.findChild(QSlider, "slider_windowing")
        self.button_minMip = self.findChild(QRadioButton, "button_minMip")
        self.button_maxMip = self.findChild(QRadioButton, "button_maxMip")
        self.button_meanMip = self.findChild(QRadioButton, "button_meanMip")
        self.box_centerSlice = self.findChild(QGroupBox, "box_centerSlice")
        self.box_numSlices = self.findChild(QGroupBox, "box_numSlices")
        self.box_windowing = self.findChild(QGroupBox, "box_windowing")
        self.box_leveling = self.findChild(QGroupBox, "box_leveling")
        self.box_mip_function = self.findChild(QComboBox, "select_functions")
        self.button_box = self.findChild(QDialogButtonBox, "button_box")

        self.slider_centerSlice.setMinimum(0)
        self.slider_centerSlice.setPageStep(1)
        self.slider_centerSlice.setMaximum(len(self.pixel_array))
        self.slider_centerSlice.setValue(len(self.pixel_array)//2)
        self.box_centerSlice.setTitle(f"Slice: {self.mri}")

        self.slider_numSlices.setMinimum(1)
        self.slider_numSlices.setPageStep(1)
        self.slider_numSlices.setMaximum(len(self.pixel_array)//2-1)
        self.box_numSlices.setTitle(f"Number Slices: {self.slider_numSlices.value()}")

        self.slider_leveling.setMinimum(0)
        self.slider_leveling.setPageStep(1)
        self.slider_leveling.setMaximum(100)
        self.slider_leveling.setValue(62)
        self.box_leveling.setTitle(f"Level: {62}")

        self.slider_windowing.setMinimum(0)
        self.slider_windowing.setPageStep(1)
        self.slider_windowing.setMaximum(100)
        self.slider_windowing.setValue(70)
        self.box_windowing.setTitle(f"Window: {70}")
        

        self.slider_centerSlice.valueChanged.connect(self.update_centerSliceText)
        self.slider_numSlices.valueChanged.connect(self.update_numSlicesText)
        self.slider_leveling.valueChanged.connect(self.update_level_label)
        self.slider_windowing.valueChanged.connect(self.update_window_label)

        self.button_box.accepted.connect(self.apply_mip)

        self.show()

    # functions to update the text of the slider when changing the value
    def update_centerSliceText(self, value):
        self.box_centerSlice.setTitle(f"Slice: {value}")

    def update_numSlicesText(self,value):
        self.box_numSlices.setTitle(f"Number Slices: {value}")
    
    def update_window_label(self, value):
        self.box_windowing.setTitle(f"Window: {value}")

    def update_level_label(self, value):
        self.box_leveling.setTitle(f"Level: {value}")

    def apply_mip(self):
        mode = "mean"
        """Generate MIP image with the selected values and show it in a new window."""
        if self.button_maxMip.isChecked():
            mode = "max"
        elif self.button_minMip.isChecked():
            mode = "min"
        elif self.button_meanMip.isChecked():
            mode = "mean"

        window = self.slider_windowing.value() / 100.0
        level = self.slider_leveling.value() / 100.0

        mip_function_index = self.box_mip_function.currentIndex()
        if mip_function_index == 0:
            mip_function = "Sigmoid Normalize"
        elif mip_function_index == 1:
            mip_function = "Logaritmic Inverse"

        center_slice = self.slider_centerSlice.value()
        num_slices = self.slider_numSlices.value()

        #print(f"Generating MIP - Mode: {mode}, Window: {window}, Level: {level}, Type: {mip_function}, Center Slice: {center_slice}, Num Slices: {num_slices}")

        mip_image = finalMIP(self.pixel_array, mode, mip_function, window, level, center_slice, num_slices, self.clahe)
        
        self.parent.mip_image = mip_image
        self.close()


class UI(QMainWindow): # main class

    def __init__(self):
        super(UI, self).__init__()

        self.setWindowTitle("Visor de Tomosíntesis")
        self.setWindowIcon(QIcon("imagenes/visionado.png"))

        uic.loadUi("diseño.ui", self)

        self.filename = ""
        self.mip_image = None
        self.slices = None
        self.clahe = False  # if clahe is applied preprocessing is applied
        self.mri = None
        self.prepro_applicado = False
        self.show_boxes = False
        self.model = CustomFaster()
        self.boxes = None
        self.classification = None
        self.scores = None
        self.progress_dialog = None
        self.inforPaciente = ""
        self.infoProcesar = ""


        state_dic = torch.load("model/faster3Epoch.pth")
        self.model.load_state_dict(state_dic)

        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()  
        ])

       


        self.frame_images = self.findChild(QFrame, "frame_images")

        self.label_image = self.findChild(QLabel,"label_image")
        self.label_image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.label_infoImage = self.findChild(QLabel, "label_infoImage")
        self.label_mip = self.findChild(QLabel, "label_mip")
        self.label_mip.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        
        self.button_clear = self.findChild(QPushButton,"button_clear")
        self.button_prepro = self.findChild(QPushButton,"button_prepro")
        self.button_procesar = self.findChild(QPushButton, "button_procesar")
        self.button_leveling = self.findChild(QPushButton, "button_leveling")
        self.button_bbox = self.findChild(QPushButton, "button_bbox")

        self.slider_image = self.findChild(QScrollBar, "slider_image")
        self.slider_image.setMinimum(0)
        self.slider_image.setPageStep(1)
        self.slider_image.hide()  # initially hide the slider

        self.warning_NumSlices = QMessageBox()

        self.warning_prepro = QMessageBox()
        self.warning_prepro.setWindowTitle("Preprocessing Warning")
        self.warning_prepro.setIcon(QMessageBox.Icon.Warning)
        self.warning_prepro.setText("Esta acción es irreversible y podría tardar unos minutos en completarse. ¿Desea continuar?")
        self.warning_prepro.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        self.warning_mip = QMessageBox()
        self.warning_mip.setWindowTitle("Warning SM")
        self.warning_mip.setIcon(QMessageBox.Icon.Warning)
        self.warning_mip.setText("Esta acción podría tardar unos minutos. ¿Desea continuar?")
        self.warning_mip.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        self.warning_procesar = QMessageBox()
        self.warning_procesar.setWindowTitle("Warning Processing")
        self.warning_procesar.setIcon(QMessageBox.Icon.Information)
        self.warning_procesar.setText("Antes de procesar la imagen, es recomensable preprocesarla para mejorar los resultados. ¿Desea hacerlo?")
        self.warning_procesar.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        boton_extra_procesar = QPushButton("Continuar Igualmente")
        self.warning_procesar.addButton(boton_extra_procesar, QMessageBox.ButtonRole.ActionRole)

        self.warning_reset = QMessageBox()
        self.warning_reset.setWindowTitle("Warning Reset")
        self.warning_reset.setIcon(QMessageBox.Icon.Warning)
        self.warning_reset.setText("Si se realiza está acción, los cambios no se podrán revertir y se reseteará la interfaz")
        self.warning_reset.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)

        
        self.add_image.triggered.connect(self.openImage)
        self.guardar_mip.triggered.connect(self.saveMIPImage) 
        self.slider_image.valueChanged.connect(self.updateSlice)
        self.button_clear.clicked.connect(self.clear_image)
        self.button_prepro.clicked.connect(self.warningPrepro)
        self.button_leveling.clicked.connect(self.open_mip_options)
        self.button_procesar.clicked.connect(self.procesar)
        self.button_bbox.clicked.connect(self.toggle_boxes)

        self.show()

    def open_mip_options(self):
        res = self.warning_mip.exec()

        if res == QMessageBox.StandardButton.Ok:
            mip_dialog = MIPDialog(self.slices, self.clahe, self.imageData, self.mri , parent=self)
            mip_dialog.exec()

            self.horizontalLayout_3.setStretch(0,0)
            
            self.showMIP(self.mip_image)
        else:
            return
    
    def toggle_boxes(self):
        """Alteranates the visibility of the bounding boxes on the image."""
        self.show_boxes = not self.show_boxes
        self.update_image(self.slider_image.value())  # update the image to show or hide the boxes

    def resizeEvent(self, event):
        """Update the image when the window is resized."""
        if hasattr(self, "slices"):  # only update if the image is loaded
            self.update_image(self.slider_image.value())
        if hasattr(self, 'mip_image'):
            self.showMIP(self.mip_image)

        event.accept()
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        current_value = self.slider_image.value()
        
        if delta > 0:  
            new_value = current_value - 1
        elif delta < 0:  
            new_value = current_value + 1
        else:
            return  

        new_value = max(self.slider_image.minimum(), min(new_value, self.slider_image.maximum()))

        self.slider_image.setValue(new_value)

    def activarBotones(self):
        self.button_prepro.setEnabled(True)
        self.button_procesar.setEnabled(True)
        self.button_leveling.setEnabled(True)

    def desactivarBotones(self):
        self.button_prepro.setEnabled(False)
        self.button_procesar.setEnabled(False)
        self.button_leveling.setEnabled(False)
        self.button_clear.setEnabled(False)
        self.button_bbox.setEnabled(False)
        
    def clear_image(self): 
        """Reset the interface and clear the image."""
        res = self.warning_reset.exec()
        if res != QMessageBox.StandardButton.Ok:
            return
        else:
            self.label_image.clear()  
            self.filename = "" 
            self.slices = None  
            self.slider_image.setValue(0)  
            self.slider_image.setMaximum(0)  
            self.slider_image.hide()  
            self.label_infoImage.setText("")
            self.horizontalLayout_3.setStretch(9,0)
            self.mip_image = None
            self.label_mip.clear()
            self.clahe = None
            self.prepro_applicado = False
            self.boxes = None
            self.classification = None
            self.scores = None
            self.inforPaciente = ""
            self.infoProcesar = ""
            self.desactivarBotones()

    def openImage(self):
        if self.filename != "":
            QMessageBox.critical(self, "Cargar Imagen", "No se puede cargar la imagen si existe otra en patalla. Por favor, resetea la interfaz")
            return
        filename, _ = QFileDialog.getOpenFileName(self, "Open File", 
                                                r'C:\Users\samue\Documents\UNI\TFG\Código\Interfaz\imagenes', 
                                                "Dicom Files (*.dcm)")

        if filename:
            self.loading_gif = QMovie("imagenes/pato.gif")
            self.label_image.setMovie(self.loading_gif)  # Stop the GIF if it was running
            self.loading_gif.start() 
            self.load_thread = LoadImageThread(filename, self)
            self.load_thread.finished.connect(self.on_image_loaded)
            self.load_thread.start()
            
    def on_image_loaded(self, result):
        
        """Manejar la imagen cargada"""
        self.slices, self.imageData = result
        self.loading_gif.stop()
    
        if self.slices is not None and self.imageData is not None:
            if len(self.slices.shape) == 3:
                self.slider_image.setMaximum(self.slices.shape[0] - 1)
                self.slider_image.show()
                self.update_image(0)  
                self.mri = self.slices.shape[0] // 2
            else:
                QMessageBox.critical(self, "Error", "La imagen DICOM no tiene el formato esperado (debe ser 3D).")
        else:
            QMessageBox.critical(self, "Error", "No se pudo cargar la imagen DICOM.")
        self.button_clear.setEnabled(True)

    def updateSlice(self, value):
        self.update_image(value)
    
    def update_image(self, slice_idx):

        if self.slices is None:
            return
        
        image_data = self.slices[slice_idx]
       
        image_data = (image_data * self.imageData[1]) + self.imageData[0]

  
        height, width = self.imageData[2], self.imageData[3]
        image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 65535    
        image_data = image_data.astype(np.uint16)
        bytes_per_line = width * 2

        qimage = QImage(image_data.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale16)
        pixmap = QPixmap.fromImage(qimage)
        self.label_image.setStyleSheet("QLabel { background-color : black; }")
        self.label_image.setAutoFillBackground(True)

        if self.show_boxes and self.boxes:
            painter = QPainter(pixmap)    
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            pen = QPen()
            pen.setColor(QColor(255, 0, 0,255))
            pen.setWidth(15)
            pen.setStyle(Qt.Qt.PenStyle.SolidLine) 
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(45)
            painter.setFont(font)

            x1, y1, x2, y2 = self.boxes
            w, h = x2 - x1, y2 - y1
            painter.drawRect(int(x1), int(y1), int(w ), int(h))

            text = f"{self.classification}: {self.scores:.2f}"
            text_rect = painter.fontMetrics().boundingRect(text)
            text_x = int(x1)
            text_y = int(y1 - 12)
            
            # verify if text is out of the left side
            if text_x < 0:
                text_x = 0
            
            # verify if text is out of the rigth side
            if text_x + text_rect.width() > width:
                text_x = width - text_rect.width()
            
            # verify if text is out of the top side
            if text_y - text_rect.height() < 0:
                text_y = text_rect.height() + 10  # the text is out of the top side, so it is put below the rectangle
        
            # draw the text
            painter.drawText(text_x, text_y, text)
            painter.end()

        label_width = self.label_image.width()
        label_height = self.label_image.height()

        if pixmap.width() > label_width or pixmap.height() > label_height:
            pixmap = pixmap.scaled(label_width,label_height,
                                Qt.Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.Qt.TransformationMode.SmoothTransformation,
                                )

        self.inforPaciente = f"""<p>Patient Name:  {self.imageData[5]}</p>
            <p>Patient ID:  {self.imageData[4]}</p>
            <p>Sexo: {self.imageData[6]}</p>
            <p>Slice: {slice_idx+1}/{self.slices.shape[0]}</p>"""

        self.label_infoImage.setText(self.inforPaciente + self.infoProcesar)
        self.label_image.setPixmap(pixmap)

        self.activarBotones()

    def warningPrepro(self):
        res = self.warning_prepro.exec()

        if res == QMessageBox.StandardButton.Ok:
            self.applyPrepro()
        else:
            return

    def applyPrepro(self):
        if self.slices is None:
            return
        if self.prepro_applicado == True:
            QMessageBox.warning(self, "Propocessing Warning", "El preprocesamiento ya se ha realizado.")
            if hasattr(self, '_pending_process') and self._pending_process:
                self._pending_process = False
                self._start_processing()
            return
        self.clahe = True
        self.statusbar_prepro = self.statusBar()
        self.progress_bar = QProgressBar(self)  
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.texto_progressBar = QLabel("Preprocesamiento cargando: ")

        self.statusbar_prepro.addPermanentWidget(self.texto_progressBar)
        self.statusbar_prepro.addPermanentWidget(self.progress_bar) 
        self.button_prepro.setEnabled(False)  
        
        self.prepro_thread = PreproThread(self.slices)
        self.prepro_thread.finished.connect(self.on_finished)
        self.prepro_thread.progress.connect(self.progress_bar.setValue)
        self.prepro_thread.start()

    def on_finished(self, result):
        self.prepro_applicado = True
        self.slices = result 
        self.update_image(self.slider_image.value()) 
        self.button_prepro.setEnabled(True)  
        self.statusbar_prepro.removeWidget(self.progress_bar)
        self.statusbar_prepro.removeWidget(self.texto_progressBar)
        self.setStatusBar(None) 
        self.texto_progressBar.clear()
        
        # if there is a pending 'processing', start it
        if hasattr(self, '_pending_process') and self._pending_process:
            self._pending_process = False
            self._start_processing()
        
    def showMIP(self,image ):

        if image is None:
            return
        
        image = (image * self.imageData[1]) + self.imageData[0]

        image = (image - image.min()) / (image.max() - image.min()) * 65535
        image = image.astype(np.uint16)

        height, width = self.imageData[2], self.imageData[3]
        bytes_per_line = width * 2

        q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale16)

        pixmap = QPixmap.fromImage(q_image)        

        label_width = self.label_image.width()
        label_height = self.label_image.height()
        if pixmap.width() > label_width or pixmap.height() > label_height:
            pixmap = pixmap.scaled(label_width, label_height,
                                Qt.Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.Qt.TransformationMode.SmoothTransformation)

        self.label_mip.setPixmap(pixmap)
        
    def saveMIPImage(self):
        """Save the SM image within three formats: PNG, JPG and DICOM.
        It could be saved with or without the bounding box."""
        draw = False
        if self.mip_image is None:
            QMessageBox.warning(self, "Guardar Imagen", "No hay ninguna imagen MIP para guardar.")
            return
        
        if self.boxes != None:
            res = QMessageBox.warning(self, "Guardar MIP con boxes", "¿Desea guardar la imagen con la caja delimitadora?", 
                                      QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
            if res == QMessageBox.StandardButton.Ok:
                self.mip_image
                draw = True


        
        file_path, file_format = QFileDialog.getSaveFileName(
            self, "Guardar Imagen", "", "PNG (*.png);;JPG (*.jpg);;DICOM (*.dcm)"
        )

        image = (self.mip_image - self.mip_image.min()) / (self.mip_image.max() - self.mip_image.min()) * 65535
        image = image.astype(np.uint16)
        height, width = self.imageData[2], self.imageData[3]

        bytes_per_line = width * 2
        q_image = QImage(image.data.tobytes(), width, height, bytes_per_line, QImage.Format.Format_Grayscale16)
        
        if draw == True:
            painter = QPainter(q_image)
            pen = QPen(QColor("red"))
            pen.setWidth(15)
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(45)
            painter.setFont(font)
            x1, y1, x2, y2 = self.boxes
            w, h = x2 - x1, y2 - y1
            painter.drawRect(Qt.QRect(int(x1), int(y1), int(w), int(h)))
            painter.drawText(int(x1), int(y1) - 10 , f"{self.classification}: {self.scores:.2f}")

            painter.end()

        

        if not file_path:  # if users cancels the dialog, file_path will be empty
            return

        # if PNG or JPG
        if file_format in ["PNG (*.png)", "JPG (*.jpg)"]:
            pixmap = QPixmap.fromImage(q_image)  
            pixmap.save(file_path) 

        # if DICOM
        elif file_format == "DICOM (*.dcm)":
            self.saveAsDicom(file_path, draw)

        QMessageBox.information(self, "Guardar Imagen", f"Imagen guardada como {file_path}")

    def saveAsDicom(self, file_path, draw):
        """Transform the MIP image to DICOM format and save it."""
        
        dicom_image = np.array(self.mip_image, dtype=np.uint16)
        if draw:
            x1, y1, x2, y2 = self.boxes
            w, h = x2 - x1, y2 - y1
            dicom_image = draw_box(dicom_image, int(x1), int(y1), int(w), int(h))
        
        # create a new DICOM dataset
        ds = Dataset()
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.preamble = b"\x00" * 128 

        anatomic_region_item = Dataset()
        anatomic_region_item.CodeValue = "76752008"  # code for "Breast"
        anatomic_region_item.CodingSchemeDesignator = "SCT"
        anatomic_region_item.CodeMeaning = "Breast"

        # set the minimum DICOM attributes
        ds.PatientName = self.imageData[5]
        ds.PatientID = self.imageData[4]
        ds.PatientSex = self.imageData[6]
        ds.Modality = "MG"  
        ds.Rows, ds.Columns = dicom_image.shape
        ds.StudyInstanceUID = generate_uid()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = generate_uid()
        ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.SamplesPerPixel = 1
        ds.PixelRepresentation = 0
        ds.AnatomicRegionSequence = Sequence([anatomic_region_item])
        ds.FrameLaterality = self.imageData[7]
        ds.ResacleIntercept = self.imageData[0]
        ds.RescaleSlope = self.imageData[1]

        ds.PixelData = dicom_image.tobytes()

        ds.save_as(file_path, write_like_original=True)
    
    def procesar(self):
        if self.slices.shape[0] < 27:
            QMessageBox.critical(self, "Error", "La imagen no puede procesarse si no tiene más de 27 slices")
            return

        if not self.clahe:
            res = self.warning_procesar.exec()
            if res == QMessageBox.StandardButton.Ok:
                self._pending_process = True
                self.applyPrepro()
                return
            elif res == QMessageBox.StandardButton.Cancel:
                return
        else:
            QMessageBox.information(self, "Procesamiento", "Este proceso puede demorarse varios minutos. Por favor, tenga paciencia.")
        self._start_processing()
        
    def _start_processing(self):
        """Function to start the processing of the image after the prepro is applied."""
        self.statusbar_procesar = self.statusBar()
        self.progress_bar = QProgressBar(self)  
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.texto_progressBar = QLabel("Procesamiento cargando: ")

        self.statusbar_procesar.addPermanentWidget(self.texto_progressBar)
        self.statusbar_procesar.addPermanentWidget(self.progress_bar) 
        
        # launch the processing thread
        self.worker = ProcessingWorker(self, self.slices, self.prepro_applicado, self.model)
        self.worker.results_ready.connect(self.procesamiento_finalizado)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.start()
    
    def procesamiento_finalizado(self, boxes, classification, scores, mri):
        self.statusbar_procesar.removeWidget(self.progress_bar)
        self.setStatusBar(None) 
        self.texto_progressBar.clear()

        self.mri = mri

        self.show_boxes = True
        self.update_image(self.slider_image.value()) 
        self.button_bbox.setEnabled(True)
        self.showResults(boxes, classification, scores)
        
    def showResults(self, boxes, classification, scores):
        self.boxes = boxes
        if classification == 0:
            self.classification = "Normal"
        else:
            self.classification = "Anomalía detectada"
        
        self.scores = scores

        self.warning_results = QMessageBox()
        self.warning_results.setWindowTitle("Resultados procesamiento")
        self.warning_results.setIcon(QMessageBox.Icon.Warning)
        self.warning_results.setText(f"Los resultados son: {self.classification} con una puntuación de {round(self.scores,2)*100}%")
        self.warning_results.exec()
        
        self.infoProcesar = f"""<p>Resultados: {self.classification}</p> 
        <p>Puntuación: {round(self.scores,2)*100}%</p>
        <p>MRI: {self.mri}</p>"""
        
        self.label_infoImage.setText(self.inforPaciente + self.infoProcesar)

   
if __name__=="__main__":
    app = QApplication(sys.argv)
    UIWindow = UI()
    UIWindow.showMaximized()
    app.exec()

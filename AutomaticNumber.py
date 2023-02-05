import sys
import cv2
import os
import easyocr
import csv
import uuid
import numpy as np
import tensorflow as tf
import datetime
import pandas as pd
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QMessageBox
from AutomaticNumber_GUI import Ui_MainWindow
from VehicleLicensePlateSearch_GUI import Ui_FMainWindow
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore, QtWidgets, QtGui
from object_detection.utils import label_map_util, config_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
from matplotlib import pyplot as plt
  
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        self.setWindowIcon(QtGui.QIcon("bot.png"))
        self.uic.btn_Start.clicked.connect(self.start_capture_video)
        self.uic.btn_stop.clicked.connect(self.exit)
        self.uic.btn_search.clicked.connect(self.search_plate)
        CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
        TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
        LABEL_MAP_NAME = 'label_map.pbtxt'
        paths = {
        'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
        'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
        'APIMODEL_PATH': os.path.join('Tensorflow','models'),
        'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
        'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
        'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
        'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
        'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
        'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
        'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
        'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
        'PROTOC_PATH':os.path.join('Tensorflow','protoc')
        }
        files = {
        'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
        'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
        'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
        }
        # Load pipeline config and build a detection model
        configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
        self.detection_model = model_builder.build(model_config=configs['model'], is_training=False)
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()
        self.category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
        self.detection_threshold = 0.7
        self.region_threshold = 0.05
    @tf.function
    def detect_fn(self,image):
        image, shapes = self.detection_model.preprocess(image)
        prediction_dict = self.detection_model.predict(image, shapes)
        detections = self.detection_model.postprocess(prediction_dict, shapes)
        return detections
    
    def filter_text(self, region, ocr_result, region_threshold):
        rectangle_size = region.shape[0]*region.shape[1] 
        plate = [] 
        for result in ocr_result:
            length = np.sum(np.subtract(result[0][1], result[0][0]))
            height = np.sum(np.subtract(result[0][2], result[0][1]))        
            if length*height / rectangle_size > region_threshold:
                plate.append(result[1])
        return plate
    def ocr_it(self, image, detections, detection_threshold, region_threshold):
        # Scores, boxes and classes above threhold
        scores = list(filter(lambda x: x> detection_threshold, detections['detection_scores']))
        boxes = detections['detection_boxes'][:len(scores)]
        classes = detections['detection_classes'][:len(scores)]    
        # Full image dimensions
        width = image.shape[1]
        height = image.shape[0] 
        # Apply ROI filtering and OCR
        for idx, box in enumerate(boxes):
            roi = box*[height, width, height, width]
            region = image[int(roi[0]):int(roi[2]),int(roi[1]):int(roi[3])]
            reader = easyocr.Reader(['en'])
            ocr_result = reader.readtext(region)
            text = self.filter_text(region, ocr_result, region_threshold)
            plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            plt.show()
            print(text)
            return text, region
    def save_results(self ,text, region, csv_filename, folder_path):
        now = QDate.currentDate()
        cr_date = now.toString('dd/MM/yyyy')
        cr_time = datetime.datetime.now().strftime("%I:%M %p")
        img_name = '{}.jpg'.format(uuid.uuid1())
        cv2.imwrite(os.path.join(folder_path, img_name), region) 
        with open(csv_filename, mode='a', newline='') as f:
            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([img_name, text, cr_date,cr_time])
        self.uic.txt_auto.setText(''.join(text))
        px='./Detection_Images/'+img_name
        self.uic.lb_auto.setPixmap(QPixmap(px))
        self.uic.txt_date.setText(cr_date)
        self.uic.txt_time.setText(cr_time)
    def exit(self):
        message = QMessageBox.warning(self,"Warning","Do you really want to exit?",QMessageBox.Yes | QMessageBox.No)
        if message == QMessageBox.Yes:
            sys.exit()
    def start_capture_video(self):
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened(): 
            ret, frame = self.cap.read()
            image_np = np.array(frame)  
            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = self.detect_fn(input_tensor)
            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections
            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
            label_id_offset = 1
            image_np_with_detections = image_np.copy()
            viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        detections['detection_boxes'],
                        detections['detection_classes']+label_id_offset,
                        detections['detection_scores'],
                        self.category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.8,
                        agnostic_mode=False)
            try: 
                text, region = self.ocr_it(image_np_with_detections, detections, self.detection_threshold, self.region_threshold)
                self.save_results(text, region, 'realtimeresults.csv', 'Detection_Images')
            except:
                pass
            if ret:
                self.displayImage(frame)
                cv2.waitKey()
        self.cap.release()
        cv2.destroyAllWindows()
    def displayImage(self,img):
        qformat = QImage.Format_Indexed8
        if len(img.shape)==3:
            if(img.shape[2])==4:
                qformat = QImage.Format_RGBA888
            else:
                qformat= QImage.Format_RGB888
        img = QImage(img,img.shape[1],img.shape[0],qformat)
        img = img.rgbSwapped()
        self.uic.label.setPixmap(QPixmap.fromImage(img))
        self.uic.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
    def search_plate(self):
        self.Second_window = QtWidgets.QMainWindow()
        self.uicf = Ui_FMainWindow()
        self.uicf.setupUi(self.Second_window)
        self.Second_window.setWindowIcon(QtGui.QIcon("search.png"))
        self.workbook = pd.read_csv('realtimeresults.csv')
        self.load_data(self.workbook)
        self.Second_window.show()       
        self.uicf.btn_fSearch.clicked.connect(self.search_fauto)
        self.uicf.btn_freload.clicked.connect(self.reload_fauto)
        self.uicf.tbW_fauto.cellClicked.connect(self.txt_load)
        self.uicf.tbW_fauto.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.uicf.cb_plate.clicked.connect(self.check_plate)
        self.uicf.cb_date.clicked.connect(self.check_date)
        self.uicf.btn_fexit.clicked.connect(self.exit_fauto)
    def exit_fauto(self):
        message = QMessageBox.question(self.Second_window,"Question","Do you want to exit?",QMessageBox.Yes | QMessageBox.No)
        if message == QMessageBox.Yes:
            self.Second_window.close()
    def check_plate(self):
        if self.uicf.cb_plate.isChecked()==True:
            self.uicf.txt_fsearch.setEnabled(True)
        else:
            self.uicf.txt_fsearch.setEnabled(False)
    def check_date(self):
        if self.uicf.cb_date.isChecked()==True:
            self.uicf.date_fauto.setEnabled(True)
        else:
            self.uicf.date_fauto.setEnabled(False)
    def load_data(self,workbook):
        self.uicf.tbW_fauto.setRowCount(len(workbook.index))
        self.uicf.tbW_fauto.setColumnCount(len(workbook.columns))
        for i in range(len(workbook.index)):
            for j in range(len(workbook.columns)):
                self.uicf.tbW_fauto.setItem(i, j, QTableWidgetItem(str(workbook.iat[i, j])))
        self.uicf.tbW_fauto.resizeColumnsToContents()
        self.uicf.tbW_fauto.resizeRowsToContents()
    def reload_fauto(self):
        self.load_data(self.workbook)
    def search_fauto(self):
        self.uicf.tbW_fauto.clear()
        text = self.uicf.txt_fsearch.toPlainText()
        date = self.uicf.date_fauto.dateTime().toString('dd/MM/yyyy')
        if self.uicf.cb_plate.isChecked()==True and self.uicf.cb_date.isChecked()==False :
            filt = (self.workbook['Plate'] == text) 
            search = self.workbook.loc[filt]
            self.load_data(search)
        elif self.uicf.cb_plate.isChecked()==False and self.uicf.cb_date.isChecked()==True :
            filt_date = (self.workbook['Date'] == date) 
            search_date = self.workbook.loc[filt_date]
            self.load_data(search_date)
        elif self.uicf.cb_plate.isChecked()==True and self.uicf.cb_date.isChecked()==True :
            filt_dateplate = (self.workbook['Date'] == date) & (self.workbook['Plate']== text )
            search_dateplate = self.workbook.loc[filt_dateplate]
            self.load_data(search_dateplate)
        else:
            QMessageBox.about(self.Second_window,"Notify","Please select search method!")
    def txt_load(self, row, col) :      
        item=self.uicf.tbW_fauto.item(row,col)
        list=[]
        for _col in range(0,4):
            item = self.uicf.tbW_fauto.item(row,_col)
            list.append(item.text())
        px='./Detection_Images/'+list[0]
        self.uicf.lb_fauto.setPixmap(QPixmap(px))
        self.uicf.txt_fauto.setText(''.join(list[1]))
        self.uicf.txt_fdate.setText(list[2])
        self.uicf.txt_ftime.setText(list[3])
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
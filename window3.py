# window3.py  (Phase 6b: Filtering → SAM2Rad only)

import sys
import os
import numpy as np
import pydicom
import cv2
import csv

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QSlider, QFileDialog,
    QGraphicsView, QGraphicsScene, QAbstractItemView, QHeaderView,
    QProgressDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QWheelEvent, QKeyEvent

from filtering1 import scan_dicom_for_horizontal_bars
from sam2rad_runner_copy import run_sam2rad_on_frame


# ============================================================
#                    ZOOMABLE VIEWER
# ============================================================
class ZoomableGraphicsView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._zoom = 0
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def wheelEvent(self, event: QWheelEvent):
        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1

        if self._zoom < -10:
            self._zoom = -10
        else:
            self.scale(factor, factor)


# ============================================================
#                      MAIN WINDOW
# ============================================================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultrasound Bone Age Pipeline — Phase 6b")

        self.PX_TO_MM = 0.06

        self.dicom_path = None
        self.frames = None

        self.row_items = []
        self.sam_results = {}
        self.manual_or_cache = {}

        self.current_row = None

        # ---------- UI ----------
        main = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()

        self.btn_load = QPushButton("Load DICOM")
        self.btn_load.clicked.connect(self.on_load_dicom)
        left.addWidget(self.btn_load)

        self.lbl_filename = QLabel("File: —")
        left.addWidget(self.lbl_filename)

        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.on_run_pipeline)
        left.addWidget(self.btn_run)

        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.on_export_csv)
        left.addWidget(self.btn_export)

        self.btn_auto_or = QPushButton("Compute Auto OR")
        self.btn_auto_or.clicked.connect(self.on_auto_or_all)
        right.addWidget(self.btn_auto_or)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Best Frame", "Bones", "OR Manual", "OR Auto"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        left.addWidget(self.table)

        main.addLayout(left, 3)

        self.viewer = ZoomableGraphicsView()
        self.scene = QGraphicsScene()
        self.viewer.setScene(self.scene)
        right.addWidget(self.viewer, 1)

        info = QHBoxLayout()
        info.addStretch(1)
        self.lbl_frame = QLabel("Frame: -")
        info.addWidget(self.lbl_frame)
        right.addLayout(info)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        right.addWidget(self.frame_slider)

        self.lbl_hint = QLabel("")
        right.addWidget(self.lbl_hint)

        btn_box = QHBoxLayout()
        self.btn_undo = QPushButton("Undo Point")
        self.btn_clear = QPushButton("Clear Points")
        self.btn_undo.clicked.connect(self.on_manual_undo)
        self.btn_clear.clicked.connect(self.on_manual_clear)
        self.btn_undo.setEnabled(False)
        self.btn_clear.setEnabled(False)
        btn_box.addWidget(self.btn_undo)
        btn_box.addWidget(self.btn_clear)
        right.addLayout(btn_box)

        main.addLayout(right, 6)

        self.viewer.mousePressEvent = self.on_image_click


    # ============================================================
    # LOAD DICOM
    # ============================================================
    def on_load_dicom(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DICOM", "", "DICOM (*.dcm)")
        if not path:
            return

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array

        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim == 4 and arr.shape[-1] > 1:
            arr = arr[..., 0]

        self.frames = arr
        self.dicom_path = path
        self.lbl_filename.setText(os.path.basename(path))

        n = arr.shape[0]
        self.frame_slider.setRange(1, n)
        self.frame_slider.setEnabled(True)
        self.btn_run.setEnabled(True)

        self.show_frame(1)

        self.table.setRowCount(0)
        self.row_items.clear()
        self.sam_results.clear()
        self.manual_or_cache.clear()
        self.current_row = None


    # ============================================================
    # RUN PIPELINE — PHASE 6b
    # ============================================================
    def on_run_pipeline(self):
        result = scan_dicom_for_horizontal_bars(self.dicom_path)
        fused = result["fused_windows"]

        n_frames = self.frames.shape[0]

        prog = QProgressDialog("Selecting best frames with SAM…", None, 0, len(fused), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()

        self.table.setRowCount(0)
        self.sam_results.clear()

        used = set()

        for i, r in enumerate(fused):
            prog.setValue(i)
            QApplication.processEvents()

            a, b = r["start"], r["end"]

            a2 = max(1, a - 5)
            b2 = min(n_frames, b + 5)

            best_f = None
            best_score = 0

            for f in range(a2, b2 + 1):
                res = run_sam2rad_on_frame(self.dicom_path, f - 1, fast_mode=True)
                nb = res.get("n_bones", 0)
                score = res.get("score", 0)

                if nb < 1:
                    continue

                if score > best_score:
                    best_score = score
                    best_f = f

            if best_f is None or best_score < 0.40:
                continue

            if best_f in used:
                continue
            used.add(best_f)

            rgb = self.to_rgb(self.frames[best_f - 1])

            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(str(best_f)))
            self.table.setItem(row, 1, QTableWidgetItem("?"))
            self.table.setItem(row, 2, QTableWidgetItem("—"))
            self.table.setItem(row, 3, QTableWidgetItem("—"))

            self.sam_results[row] = {
                "best_frame": best_f,
                "bones": "?",
                "rgb": rgb
            }

        prog.setValue(len(fused))
        QApplication.processEvents()

        if self.table.rowCount() > 0:
            self.table.selectRow(0)

        self.btn_export.setEnabled(True)
        self.btn_undo.setEnabled(True)
        self.btn_clear.setEnabled(True)


    # ============================================================
    # AUTO OR
    # ============================================================
    def on_auto_or_all(self):
        rows = self.table.rowCount()

        for r in range(rows):
            best = int(self.table.item(r, 0).text())
            res = run_sam2rad_on_frame(self.dicom_path, best - 1, fast_mode=False)

            nb = res.get("n_bones", None)
            if nb is not None:
                self.table.setItem(r, 1, QTableWidgetItem(str(nb)))
                self.sam_results[r]["bones"] = nb

            or1 = res.get("or1")
            or2 = res.get("or2")

            txt = ""
            if or1 is not None:
                txt += f"OR1={or1*100:.1f}% "
            if or2 is not None:
                txt += f"OR2={or2*100:.1f}%"
            txt = txt.strip() or "—"

            self.table.setItem(r, 3, QTableWidgetItem(txt))


    # ============================================================
    # DISPLAY + MANUAL OR
    # ============================================================
    def to_rgb(self, frame):
        f = frame.astype(np.float32)
        f = (f - f.min()) / max(f.max() - f.min(), 1e-5)
        f = (f * 255).astype(np.uint8)
        return np.stack([f, f, f], axis=-1)

    def show_frame(self, idx):
        rgb = self.to_rgb(self.frames[idx - 1])
        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(q))
        self.lbl_frame.setText(f"Frame: {idx}")

    def on_slider_changed(self, v):
        self.show_frame(v)

    def on_row_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        self.current_row = row
        best = self.sam_results[row]["best_frame"]
        self.frame_slider.setValue(best)

    def on_image_click(self, ev):
        pass

    def on_manual_undo(self):
        pass

    def on_manual_clear(self):
        pass


    # ============================================================
    # EXPORT
    # ============================================================
    def on_export_csv(self):
        out_path = os.path.join(os.path.dirname(self.dicom_path), "OR_results.csv")

        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Best Frame", "Bones", "OR Manual", "OR Auto"])
            for r in range(self.table.rowCount()):
                row = [
                    self.table.item(r, 0).text(),
                    self.table.item(r, 1).text(),
                    self.table.item(r, 2).text(),
                    self.table.item(r, 3).text()
                ]
                writer.writerow(row)

        QMessageBox.information(self, "Export", f"Saved to:\n{out_path}")


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 900)
    w.show()
    sys.exit(app.exec())
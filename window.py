#window.py
import sys
import os

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTableWidget, QTableWidgetItem, QSlider, QFileDialog,
    QGraphicsView, QGraphicsScene, QAbstractItemView, QHeaderView,
    QProgressDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import (
    QPixmap, QImage, QWheelEvent, QKeyEvent
)

import numpy as np
import pydicom
import cv2
import csv

from run_unet_filter_intervals import get_kept_frames
from sam2rad_runner import run_sam2rad_on_frame


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
        self.setWindowTitle("Ultrasound Bone Age Pipeline")

        # pixel → mm scale
        self.PX_TO_MM = 0.06

        self.dicom_path = None
        self.frames = None

        # row_items[i] = {"best_frame": int, "bones": int}
        self.row_items = []

        # sam_results[row] = {"best_frame": int, "bones": int, "rgb": np.ndarray}
        self.sam_results = {}

        # manual_or_cache[row] = {"pts":[(x,y), ...], "OR1": float, "OR2": float}
        self.manual_or_cache = {}

        self.current_row = None

        # ---------- UI ----------
        main = QHBoxLayout(self)
        left = QVBoxLayout()
        right = QVBoxLayout()

        # LEFT SIDE ------------------------------------------------
        self.btn_load = QPushButton("Load DICOM")
        self.btn_load.clicked.connect(self.on_load_dicom)
        left.addWidget(self.btn_load)

        self.lbl_filename = QLabel("File: —")
        left.addWidget(self.lbl_filename)

        self.btn_run = QPushButton("Run Pipeline")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.on_run_pipeline)
        left.addWidget(self.btn_run)
        # Export button
        self.btn_export = QPushButton("Export to CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.on_export_csv)
        left.addWidget(self.btn_export)
        self.btn_auto_or = QPushButton("Compute Auto OR")
        self.btn_auto_or.clicked.connect(self.on_auto_or_all)
        right.addWidget(self.btn_auto_or)


        # table: Best Frame | Bones | OR Manual
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Best Frame", "Bones", "OR Manual", "OR Auto"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.on_row_selected)
        left.addWidget(self.table)

        main.addLayout(left, 3)

        # RIGHT SIDE -----------------------------------------------
        self.viewer = ZoomableGraphicsView()
        self.scene = QGraphicsScene()
        self.viewer.setScene(self.scene)
        right.addWidget(self.viewer, 1)

        # frame label
        info = QHBoxLayout()
        info.addStretch(1)
        self.lbl_frame = QLabel("Frame: -")
        info.addWidget(self.lbl_frame)
        right.addLayout(info)

        # slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        right.addWidget(self.frame_slider)

        # hint text
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet("color: Black;")
        right.addWidget(self.lbl_hint)

        # undo / clear
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

        # enable manual point clicks
        self.viewer.mousePressEvent = self.on_image_click

    # ============================================================
    # LOAD DICOM
    # ============================================================
    def on_load_dicom(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select DICOM", "", "DICOM (*.dcm)"
        )
        if not path:
            return

        try:
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

            # show first frame
            self.show_frame(1)

            # reset state
            self.table.setRowCount(0)
            self.row_items.clear()
            self.sam_results.clear()
            self.manual_or_cache.clear()
            self.current_row = None
            self.lbl_hint.setText("")

        except Exception as e:
            QMessageBox.critical(self, "DICOM Load Error", repr(e))

    # ============================================================
    # RUN PIPELINE
    # ============================================================
    def on_run_pipeline(self):
        if not self.dicom_path:
            return

        # 1. UNet picks best frames per fused window
        kept_list = get_kept_frames(self.dicom_path)

        prog = QProgressDialog("Running Pipeline…", None, 0, len(kept_list), self)
        prog.setWindowTitle("Processing")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.show()

        # reset UI state
        self.table.setRowCount(0)
        self.row_items.clear()
        self.sam_results.clear()
        self.manual_or_cache.clear()
        self.current_row = None

        accepted = 0
        used_best_frames = set()   # ensured uniqueness

        for k, item in enumerate(kept_list):
            prog.setValue(k)
            QApplication.processEvents()

            kept = item["kept_frames"]
            if not kept:
                continue

            # choose strongest frame from UNet results
            best_frame = kept[0]

            # avoid duplicates
            if best_frame in used_best_frames:
                continue
            used_best_frames.add(best_frame)

            # Load raw frame (no mask, no overlay)
            frame_raw = self.frames[best_frame - 1]
            rgb = self.to_rgb(frame_raw)

            # Since SAM isn't run yet, bone count is unknown
            # Using placeholder until Auto OR
            n_bones = "?"

            row = self.table.rowCount()
            self.table.insertRow(row)

            self.table.setItem(row, 0, QTableWidgetItem(str(best_frame)))
            self.table.setItem(row, 1, QTableWidgetItem(str(n_bones)))
            self.table.setItem(row, 2, QTableWidgetItem("—"))
            self.table.setItem(row, 3, QTableWidgetItem("—"))  # OR Auto empty initially

            self.row_items.append({
                "best_frame": best_frame,
                "bones": n_bones
            })

            self.sam_results[row] = {
                "best_frame": best_frame,
                "bones": n_bones,
                "rgb": rgb
            }

            accepted += 1

        prog.setValue(len(kept_list))
        QApplication.processEvents()

        if accepted > 0:
            self.table.selectRow(0)

        self.lbl_hint.setText(
            "Select a row. Use slider to inspect frame. Press 'Compute Auto OR' to run SAM2Rad."
        )

        # allow manual plotting regardless
        self.enable_manual_buttons(True)

        # export CSV only if we have result rows
        self.btn_export.setEnabled(accepted > 0)

    def on_auto_or_all(self):
        if not self.dicom_path:
            return

        from sam2rad_runner import run_sam2rad_on_frame

        rows = self.table.rowCount()

        for r in range(rows):
            # read best frame
            best = int(self.table.item(r, 0).text())

            # run SAM2Rad
            res = run_sam2rad_on_frame(self.dicom_path, best - 1)

            # --- overwrite Bones with SAM result
            sam_bones = res.get("n_bones", None)
            if sam_bones is not None:
                self.table.setItem(r, 1, QTableWidgetItem(str(sam_bones)))
                self.sam_results[r]["bones"] = sam_bones

            # --- OR values
            or1 = res.get("or1", None)
            or2 = res.get("or2", None)

            txt = ""
            if or1 is not None:
                txt += f"OR1={or1*100:.1f}% "
            if or2 is not None:
                txt += f"OR2={or2*100:.1f}%"
            txt = txt.strip() or "—"

            # write into OR Auto column (col 3)
            self.table.setItem(r, 3, QTableWidgetItem(txt))



    # ============================================================
    def enable_manual_buttons(self, flag: bool):
        self.btn_undo.setEnabled(flag)
        self.btn_clear.setEnabled(flag)

    # ============================================================
    # TABLE SELECTION
    # ============================================================
    def on_row_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return

        self.current_row = row

        if row not in self.sam_results:
            return

        data = self.sam_results[row]
        best = data["best_frame"]
        n_b = data["bones"]

        self.frame_slider.setValue(best)  # triggers on_slider_changed → show_frame

        allowed = self.required_points(n_b)

        # If bone count is unknown (because we haven't run SAM yet)
        if isinstance(n_b, str):
            msg = f"Bone count pending. Manual OR: click {allowed[-1]} points (3 or 6)."
        else:
            msg = f"Manual OR allowed. Click {allowed[-1]} points (3 or 6)."

        self.lbl_hint.setText(msg)

        pts = []
        if row in self.manual_or_cache:
            pts = self.manual_or_cache[row].get("pts", [])
        self.overlay(pts)

    # ============================================================
    # RGB HELPER
    # ============================================================
    def to_rgb(self, frame):
        f = frame.astype(np.float32)
        mn, mx = f.min(), f.max()
        if mx > mn:
            f = (f - mn) / (mx - mn)
        f = (f * 255).astype(np.uint8)
        return np.stack([f, f, f], axis=-1)

    # ============================================================
    # DISPLAY
    # ============================================================
    def display_rgb(self, rgb, idx, n):
        q = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.scene.clear()
        self.scene.addPixmap(QPixmap.fromImage(q))
        self.lbl_frame.setText(f"Frame: {idx} / {n}")
        self.scene.update()

    def show_frame(self, idx):
        """Show a frame; if it's the current best-frame, overlay points."""
        if self.frames is None:
            return

        n = self.frames.shape[0]
        idx = max(1, min(idx, n))

        # If this is the current row's best frame, show with overlay
        if self.current_row is not None and self.current_row in self.sam_results:
            data = self.sam_results[self.current_row]
            best = data["best_frame"]
            if idx == best:
                pts = []
                if self.current_row in self.manual_or_cache:
                    pts = self.manual_or_cache[self.current_row].get("pts", [])
                self.overlay(pts)
                return

        # otherwise, just show raw frame
        frame_raw = self.frames[idx - 1]
        rgb = self.to_rgb(frame_raw)
        self.display_rgb(rgb, idx, n)

    def on_slider_changed(self, v):
        """Slider → show corresponding frame, with overlay if applicable."""
        if self.frames is None:
            return
        n = self.frames.shape[0]
        idx = max(1, min(v, n))
        self.show_frame(idx)

    # ============================================================
    # OVERLAY POINTS ON BEST FRAME
    # ============================================================
    def overlay(self, pts):
        if self.current_row is None:
            return
        if self.current_row not in self.sam_results:
            return

        data = self.sam_results[self.current_row]
        best = data["best_frame"]
        rgb = data["rgb"].copy()

        for (x, y) in pts:
            cv2.circle(rgb, (int(x), int(y)), 6, (0, 0, 255), -1)

        n = self.frames.shape[0]
        self.display_rgb(rgb, best, n)

    # ============================================================
    # IMAGE CLICK → COLLECT POINTS
    # ============================================================
    def on_image_click(self, ev):
        if self.current_row is None:
            return
        if self.current_row not in self.sam_results:
            return

        data = self.sam_results[self.current_row]
        best = data["best_frame"]

        # only allow clicking on that row's best frame
        if self.frame_slider.value() != best:
            return

        pos = self.viewer.mapToScene(ev.pos())
        x, y = pos.x(), pos.y()

        rgb = data["rgb"]
        H, W = rgb.shape[:2]
        if not (0 <= x < W and 0 <= y < H):
            return

        rec = self.manual_or_cache.get(self.current_row, {"pts": []})
        pts = rec.get("pts", [])

        allowed = self.required_points(data["bones"])
        pts.append((x, y))
        self.manual_or_cache[self.current_row] = {"pts": pts}
        self.overlay(pts)

        if len(pts) in allowed:
            self.compute_manual_OR(self.current_row)
        elif len(pts) > max(allowed):
            # too many → ignore extra
            pts.pop()
            self.manual_or_cache[self.current_row]["pts"] = pts
            self.overlay(pts)


    # ============================================================
    # REQUIRED POINTS
    # ============================================================
    def required_points(self, n_bones):
        
        return [3,6]

    # ============================================================
    # UNDO / CLEAR
    # ============================================================
    def on_manual_undo(self):
        if self.current_row is None:
            return
        if self.current_row not in self.manual_or_cache:
            return

        pts = self.manual_or_cache[self.current_row].get("pts", [])
        if not pts:
            return

        pts.pop()
        self.manual_or_cache[self.current_row] = {"pts": pts}
        self.table.setItem(self.current_row, 2, QTableWidgetItem("—"))
        self.overlay(pts)

    def on_manual_clear(self):
        if self.current_row is None:
            return
        if self.current_row not in self.manual_or_cache:
            return

        self.manual_or_cache[self.current_row] = {"pts": []}
        self.table.setItem(self.current_row, 2, QTableWidgetItem("—"))
        self.overlay([])

    # ============================================================
    # MANUAL OR COMPUTE
    # ============================================================
    def compute_manual_OR(self, row):
        pts = self.manual_or_cache[row].get("pts", [])
        if len(pts) not in (3, 6):
            return

        # px → mm
        def dist(a, b):
            dx = a[0] - b[0]
            dy = a[1] - b[1]
            return ((dx*dx + dy*dy)**0.5) * self.PX_TO_MM

        OR1 = None
        OR2 = None

        if len(pts) == 3:
            h = dist(pts[0], pts[1])
            H = dist(pts[0], pts[2])
            if H > 0:
                OR1 = h/H

        elif len(pts) == 6:
            h1 = dist(pts[0], pts[1])
            H1 = dist(pts[0], pts[2])
            h2 = dist(pts[3], pts[4])
            H2 = dist(pts[3], pts[5])
            if H1 > 0:
                OR1 = h1/H1
            if H2 > 0:
                OR2 = h2/H2

        txt = ""
        if OR1 is not None:
            txt += f"OR1={OR1*100:.1f}% "
        if OR2 is not None:
            txt += f"OR2={OR2*100:.1f}%"
        txt = txt.strip() or "—"

        self.table.setItem(row, 2, QTableWidgetItem(txt))
        self.manual_or_cache[row]["OR1"] = OR1
        self.manual_or_cache[row]["OR2"] = OR2

        print(f"[MANUAL] Row {row} → {txt}")


    # ============================================================
    # EXPORT CSV
    # ============================================================
    def on_export_csv(self):
        if not self.dicom_path:
            QMessageBox.warning(self, "No File", "Load a DICOM first.")
            return

        # determine output path
        dcm_dir = os.path.dirname(self.dicom_path)
        out_path = os.path.join(dcm_dir, "OR_results.csv")

        # collect rows
        rows = self.collect_csv_rows()

        # write CSV
        try:
            with open(out_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Best Frame", "Bones", "OR1 (%)", "OR2 (%)"])
                for r in rows:
                    writer.writerow(r)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Results saved to:\n{out_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


    # ============================================================
    # COLLECT CSV ROWS
    # ============================================================
    def collect_csv_rows(self):
        rows = []
        nrows = self.table.rowCount()

        for r in range(nrows):
            # best frame
            best = self.table.item(r, 0).text().strip()

            # bone count
            bones = self.table.item(r, 1).text().strip()

            # OR text from table
            or_txt = self.table.item(r, 2).text().strip()

            OR1 = ""
            OR2 = ""

            if r in self.manual_or_cache:
                o1 = self.manual_or_cache[r].get("OR1", None)
                o2 = self.manual_or_cache[r].get("OR2", None)

                if o1 is not None:
                    OR1 = f"{o1*100:.1f}%"
                if o2 is not None:
                    OR2 = f"{o2*100:.1f}%"

            else:
                # if OR text formatted in the table but not cache
                if "OR1" in or_txt:
                    try:
                        # quick parse if already in table
                        for token in or_txt.split():
                            if token.startswith("OR1="):
                                OR1 = token.split("=")[1]
                            if token.startswith("OR2="):
                                OR2 = token.split("=")[1]
                    except:
                        pass

            rows.append([best, bones, OR1, OR2])

        return rows

    
    # ============================================================
    # KEYBOARD LEFT/RIGHT
    # ============================================================
    def keyPressEvent(self, ev: QKeyEvent):
        if not self.frame_slider.isEnabled():
            return
        v = self.frame_slider.value()
        if ev.key() == Qt.Key_Left:
            self.frame_slider.setValue(max(1, v - 1))
        elif ev.key() == Qt.Key_Right:
            self.frame_slider.setValue(min(self.frame_slider.maximum(), v + 1))


# ============================================================
# ENTRY
# ============================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1600, 900)
    w.show()
    sys.exit(app.exec())
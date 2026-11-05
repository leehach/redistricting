"""Plan Metrics window. run with: python main.py"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

import metrics  # noqa: F401 (registers the metrics)
from metrics_dialog_ui import Ui_MetricsDialog  # the layout from Qt Designer
from registry import (FIELDS, GEOMETRY, REGISTRY, PlanContext, auto_match, has_shapes,
                      needed_fields, run_metrics, table_columns)

SPATIAL = {".shp", ".gpkg", ".geojson", ".json", ".zip"}  # read with geopandas, everything else as csv
FILE_FILTER = ("Data files (*.csv *.txt *.tsv *.shp *.gpkg *.geojson *.json *.zip);;"
               "All files (*)")
NOT_SET = "(choose a field)"  # first option in every field dropdown


def read_input(path):
    if Path(path).suffix.lower() in SPATIAL:
        return gpd.read_file(path)
    # sep=None lets pandas figure out comma vs tab vs semicolon
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


class Worker(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, ctx, chosen, name, out):
        super().__init__()
        self.ctx, self.chosen, self.name, self.out = ctx, chosen, name, out

    def run(self):
        try:
            table = run_metrics(self.ctx, self.chosen, self.name)
            table.to_csv(self.out, index=False)
            self.done.emit(table)
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class MetricsDialog(QtWidgets.QDialog, Ui_MetricsDialog):
    def __init__(self, *args, data=None, name="plan", **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.data, self.name = None, ""
        self.columns = {}  # field -> which column in the user's file
        self.worker, self.running, self.last_dir = None, False, "."
        self.build_tree()

        # signals and slots
        self.inputBrowseButton.clicked.connect(self.open_input)
        self.outputBrowseButton.clicked.connect(self.save_file)
        self.selectAllButton.clicked.connect(lambda: self.check_all(Qt.Checked))
        self.selectNoneButton.clicked.connect(lambda: self.check_all(Qt.Unchecked))
        self.metricTree.itemChanged.connect(self.refresh_fields)
        self.outputLineEdit.textChanged.connect(self.update_ready)
        self.runButton.clicked.connect(self.calculate)

        if data is not None:
            self.set_data(data, name)
        self.refresh_fields()

    # --- metric checkboxes ---

    def build_tree(self):
        #one "Group (All)" box per group with its metrics under it,
        #groups with only 1 metric just get a single box
        for group, items in REGISTRY.items():
            if len(items) == 1:
                self.add_metric_item(self.metricTree, items[0])
                continue
            parent = QtWidgets.QTreeWidgetItem(self.metricTree, [f"{group} (All)"])
            parent.setFlags(parent.flags() | Qt.ItemIsAutoTristate)  # parent checks/unchecks its children
            parent.setCheckState(0, Qt.Unchecked)
            for m in items:
                self.add_metric_item(parent, m)

    @staticmethod
    def add_metric_item(parent, m):
        item = QtWidgets.QTreeWidgetItem(parent, [m.name])
        item.setCheckState(0, Qt.Unchecked)
        item.setData(0, Qt.UserRole, m)  # stash the metric on the checkbox so we can get it back later

    def check_all(self, state):
        for i in range(self.metricTree.topLevelItemCount()):
            self.metricTree.topLevelItem(i).setCheckState(0, state)

    def chosen_metrics(self):
        it = QtWidgets.QTreeWidgetItemIterator(self.metricTree, QtWidgets.QTreeWidgetItemIterator.Checked)
        chosen = []
        while it.value():
            m = it.value().data(0, Qt.UserRole)
            if m is not None:  # group boxes don't have a metric attached
                chosen.append(m)
            it += 1
        return chosen

    # --- field matching ---

    def refresh_fields(self):
        # rebuilds the dropdowns every time the checked metrics change
        while self.fieldForm.rowCount():
            self.fieldForm.removeRow(0)
        chosen = self.chosen_metrics()
        if self.data is None or not chosen:
            self.fieldHintLabel.setText("Load an input file first." if self.data is None
                                        else "Tick a metric to see the fields it needs.")
            self.update_ready()
            return

        keys = needed_fields(chosen)
        kept = {k: c for k, c in self.columns.items() if k in keys}  # keep anything the user already picked
        self.columns.update(auto_match(self.data, keys, kept))
        self.fieldHintLabel.setText("Fields the ticked metrics need. Check the guesses "
                                    "and choose any that are missing.")
        names = table_columns(self.data)
        for key in keys:
            combo = QtWidgets.QComboBox()
            combo.addItems([NOT_SET, *map(str, names)])
            if key in self.columns:
                combo.setCurrentText(str(self.columns[key]))
            # index 0 is NOT_SET, so column i is names[i - 1]
            combo.currentIndexChanged.connect(
                lambda i, key=key: self.set_column(key, names[i - 1] if i else None))
            label = QtWidgets.QLabel(FIELDS[key].label)
            label.setToolTip("Needed by: " + ", ".join(m.name for m in chosen if key in m.needs))
            self.fieldForm.addRow(label, combo)
        if any(GEOMETRY in m.needs for m in chosen):
            self.fieldForm.addRow("Shapes", QtWidgets.QLabel(
                "from the file" if has_shapes(self.data) else "none in this file"))
        self.update_ready()

    def set_column(self, key, column):
        if column is None:
            self.columns.pop(key, None)
        else:
            self.columns[key] = column
        self.update_ready()

    # --- files ---

    def open_input(self):
        fn = QtWidgets.QFileDialog.getOpenFileName(self, "Open input file", self.last_dir, FILE_FILTER)[0]
        if not fn:  # user hit cancel
            return
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            data = read_input(fn)
        except Exception as e:  # show the error instead of crashing the app
            QtWidgets.QMessageBox.warning(self, "Couldn't read the file", f"{Path(fn).name}\n\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.last_dir = str(Path(fn).parent)
        self.inputLineEdit.setText(fn)
        self.outputLineEdit.setText(str(Path(fn).with_name(f"{Path(fn).stem}_metrics.csv")))
        self.set_data(data, Path(fn).stem)

    def set_data(self, data, name):
        # also works for data already in memory: MetricsDialog(data=gdf, name="x")
        self.data, self.name, self.columns = data, name, {}
        shapes = ", with shapes" if has_shapes(data) else ", no shapes"
        self.inputInfoLabel.setText(f"{name}: {len(data):,} rows, "
                                    f"{len(table_columns(data))} fields{shapes}")
        if not self.outputLineEdit.text():
            self.outputLineEdit.setText(str(Path.cwd() / f"{name}_metrics.csv"))
        self.refresh_fields()

    def save_file(self):
        fn = QtWidgets.QFileDialog.getSaveFileName(self, "Save metrics", self.outputLineEdit.text() or ".",
                                                   "CSV (*.csv)")[0]
        if fn:
            self.outputLineEdit.setText(fn)

    # --- running ---

    def problem(self):
        # whatever is still missing before Calculate can run, None if good to go
        chosen = self.chosen_metrics()
        if self.data is None:
            return "Load an input file."
        if not chosen:
            return "Tick at least one metric."
        keys = needed_fields(chosen)
        missing = [FIELDS[k].label for k in keys if k not in self.columns]
        if missing:
            return f"Choose a field for: {', '.join(missing)}."
        picked = [self.columns[k] for k in keys]
        if len(set(picked)) < len(picked):
            return "The same column is chosen for two fields."
        need_shapes = sorted({m.group for m in chosen if GEOMETRY in m.needs})
        if need_shapes and not has_shapes(self.data):
            return (f"{', '.join(need_shapes)} needs shapes: load a shapefile, GeoPackage "
                    "or GeoJSON, or untick it.")
        if not self.outputLineEdit.text().strip():
            return "Choose where to save the CSV."
        return None

    def update_ready(self):
        # greys out Calculate and shows what's missing at the bottom
        if self.running:
            return
        problem = self.problem()
        self.runButton.setEnabled(problem is None)
        self.statusLabel.setText(problem or "Ready.")

    def calculate(self):
        chosen, out = self.chosen_metrics(), self.outputLineEdit.text().strip()
        ctx = PlanContext(self.data, {k: self.columns[k] for k in needed_fields(chosen)})
        self.running = True
        self.runButton.setEnabled(False)
        self.statusLabel.setText("Calculating...")
        self.worker = Worker(ctx, chosen, self.name, out)
        self.worker.done.connect(lambda table: self.finished(table, out))
        self.worker.failed.connect(self.failed)
        self.worker.start()

    def finished(self, table, out):
        self.running = False
        self.update_ready()
        self.statusLabel.setText(f"Saved {len(table)} districts to {out}")

    def failed(self, message):
        self.running = False
        self.update_ready()
        QtWidgets.QMessageBox.critical(self, "Calculation failed", message)

    def reject(self):
        # Close, Esc and the X all end up here. don't close while it's still running
        if self.running:
            self.statusLabel.setText("Still calculating; close once it has saved.")
            return
        super().reject()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = MetricsDialog()
    dialog.show()
    app.exec()

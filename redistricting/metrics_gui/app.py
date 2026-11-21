"""Main app window. run with: python app.py

Plans are read in bulk from PAFs/BAFs (column 1 GEOID, column 2 district, see plan_files.py) and kept
in self.plans (a SessionPlans) for as long as the window is open. Precinct data (votes, demographics,
shapes) is loaded separately and only joined onto a plan when an analysis opens.
"""

import sys
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import theme
from main import FILE_FILTER as DATA_FILE_FILTER
from main import MetricsDialog
from main_window_ui import Ui_MainWindow  # the layout from Qt Designer
from plan_files import FILE_FILTER as PLAN_FILE_FILTER
from plan_files import PlanFileError, read_plan_files
from reconcile import ReconcileError, find_key, has_shapes, read_table, reconcile
from session import SessionPlans
from state_data import LABELS, SOURCES, STATES, geography, has_census_key, unit_name

MAX_LISTED = 10  # problems shown in the message box, the rest go under "Show Details"
FIPS_TO_ABBR = {fips: abbr for abbr, (_, fips) in STATES.items()}
def info_text(title, rows):
    """a bold title over 'Label: value' rows, for the (i) bubbles"""
    lines = "".join(f"<tr><td><b>{label}:</b></td><td>{value}</td></tr>" for label, value in rows)
    return f"<p><b>{title}</b></p><table cellspacing='0' cellpadding='2'>{lines}</table>"


# text for the (i) buttons. click one to see it
INFO = {
    "plans": info_text("Plan file format", [
        ("Column 1", "GEOID"),
        ("Column 2", "District number (integer)"),
        ("Other columns", "Ignored"),
        ("Header row", "Optional"),
        ("11-character GEOID", "Precinct assignment file (PAF)"),
        ("15-character GEOID", "Block assignment file (BAF)"),
        ("Folder import", "Skips files ending in _metrics"),
    ]),
    "data": info_text("Precinct data", [
        ("File types", "CSV, TSV, TXT, shapefile, GeoPackage, GeoJSON, zipped shapefile"),
        ("Join key", "GEOID column (GEOID20, GEOID, VTD or precinct ID)"),
        ("Joined", "When Metrics opens"),
        ("State downloads", "Joined only to plans from the same state"),
        ("Added files", "Joined to any plan they share GEOIDs with"),
    ]),
    "state": info_text("Data sources", [
        ("Shapes", "2020 Census TIGER/Line voting districts; blocks for CA, HI, ME and OR"),
        ("VEST", "2020 precinct election results and demographics (ALARM Project)"),
        ("Census", "2020 PL 94-171 redistricting data"),
        ("Cache", "~/.redistricting_cache"),
    ]),
}


def info_button(text, parent=None):
    """a small (i) button that shows text in a bubble when clicked"""
    button = QtWidgets.QToolButton(parent)
    button.setIcon(button.style().standardIcon(QtWidgets.QStyle.SP_MessageBoxInformation))
    button.setIconSize(QtCore.QSize(16, 16))
    button.setAutoRaise(True)
    button.setCursor(Qt.PointingHandCursor)
    button.setAccessibleName("Info")
    button.clicked.connect(lambda: QtWidgets.QWhatsThis.showText(
        button.mapToGlobal(button.rect().bottomLeft()), text, button))
    return button


class EmptyHint(QtWidgets.QLabel):
    """grey text in the middle of an empty table or list saying what to do first"""

    def __init__(self, view, text):
        super().__init__(text, view.viewport())
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setForegroundRole(QtGui.QPalette.PlaceholderText)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)  # clicks still reach the view
        view.viewport().installEventFilter(self)  # stay the size of the view

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Resize:
            self.setGeometry(obj.rect().adjusted(16, 0, -16, 0))
        return False


class Task(QtCore.QThread):
    """runs func(say) off the UI thread. say(text) updates the status bar"""
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def run(self):
        try:
            self.done.emit(self.func(self.progress.emit))
        except (ReconcileError, PlanFileError) as e:  # these are already written for the user
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


class StateDataDialog(QtWidgets.QDialog):
    """pick a state and which of its data sets to download"""

    def __init__(self, loaded, state="PA", parent=None):
        super().__init__(parent)
        self.loaded = loaded  # labels already in the session, so they aren't downloaded twice
        self.setWindowTitle("Download State Data")
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.stateCombo = QtWidgets.QComboBox()
        for abbr, (name, _) in sorted(STATES.items(), key=lambda kv: kv[1][0]):
            self.stateCombo.addItem(f"{name} ({abbr})", abbr)
        self.stateCombo.setCurrentIndex(self.stateCombo.findData(state))
        form.addRow("State *", self.stateCombo)
        layout.addLayout(form)

        box = QtWidgets.QGroupBox("Data *")
        box_layout = QtWidgets.QVBoxLayout(box)
        self.checks = {}
        for key, (text, _) in SOURCES.items():
            self.checks[key] = QtWidgets.QCheckBox(text)
            box_layout.addWidget(self.checks[key])
        self.unitLabel = QtWidgets.QLabel(wordWrap=True)
        unit_row = QtWidgets.QHBoxLayout()
        unit_row.addWidget(self.unitLabel, 1)
        unit_row.addWidget(info_button(INFO["state"], self))
        box_layout.addLayout(unit_row)
        layout.addWidget(box)

        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setText("Download")
        layout.addWidget(self.buttonBox)

        self.has_key = has_census_key()
        self.checks["vest"].setChecked(True)  # the usual one; shapes are big, Census needs a key
        self.stateCombo.currentIndexChanged.connect(self.refresh)
        for check in self.checks.values():
            check.toggled.connect(self.update_ready)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.refresh()

    def state(self):
        return self.stateCombo.currentData()

    def refresh(self):
        abbr = self.state()
        for key, check in self.checks.items():
            text = SOURCES[key][0]
            loaded = LABELS[key].format(abbr=abbr) in self.loaded
            no_key = key == "census" and not self.has_key
            check.setEnabled(not loaded and not no_key)
            if not check.isEnabled():
                check.setChecked(False)
            check.setText(text + (" - loaded" if loaded else ""))
            check.setToolTip("Set CENSUS_API_KEY in the conda environment to use this."
                             if no_key else "")
        large = " Block downloads are large." if geography(abbr) == "block" else ""
        self.unitLabel.setText(f"{STATES[abbr][0]} data is by {unit_name(abbr)}.{large}")
        self.update_ready()

    def chosen(self):
        return [k for k, c in self.checks.items() if c.isEnabled() and c.isChecked()]

    def update_ready(self):
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(bool(self.chosen()))


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, plans=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.plans = SessionPlans() if plans is None else plans  # the in-memory plans object
        self.precinct_data = {}  # label -> table keyed by GEOID, joined onto plans for analyses
        self.data_state = {}     # label -> state FIPS for downloads, so a plan only gets its state's
        self.task, self.last_dir, self.last_state = None, ".", "PA"
        self.setAcceptDrops(True)
        # (i) buttons right after each box's add buttons
        self.planButtonLayout.insertWidget(2, info_button(INFO["plans"], self))
        self.dataButtonLayout.insertWidget(2, info_button(INFO["data"], self))
        self.planHint = EmptyHint(self.planTable, "No plans loaded. Click Import Plans or "
                                  "Import Folder, or drag plan files onto this window.")
        self.dataHint = EmptyHint(self.dataList, "No precinct data loaded. Add vote, demographic "
                                  "or shape files keyed by GEOID, or download state data.")

        header = self.planTable.horizontalHeader()
        for col in range(4):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        # Del removes plans only while the plan table has focus
        self.planTable.addAction(self.actionRemovePlans)
        self.actionRemovePlans.setShortcutContext(Qt.WidgetWithChildrenShortcut)

        # signals and slots: each button does the same thing as its menu item
        for slot, button, action in [
            (self.import_plans, self.importButton, self.actionImportPlans),
            (self.import_folder, self.importFolderButton, self.actionImportFolder),
            (self.open_metrics, self.metricsButton, self.actionMetrics),
            (self.remove_plans, self.removeButton, self.actionRemovePlans),
            (self.add_data_file, self.addDataButton, self.actionAddData),
            (self.download_state, self.stateDataButton, self.actionDownloadState),
            (self.remove_data, self.removeDataButton, self.actionRemoveData),
        ]:
            button.clicked.connect(slot)
            action.triggered.connect(slot)
        self.actionExit.triggered.connect(self.close)
        self.planTable.itemSelectionChanged.connect(self.show_selected)
        self.planTable.doubleClicked.connect(lambda index: self.open_metrics())
        self.dataList.itemSelectionChanged.connect(self.update_ready)

        self.build_theme_menu()
        self.refresh_plans()
        self.refresh_data()

    def build_theme_menu(self):
        """View > Theme: one checkable item per theme, only one ticked at a time"""
        group = QtWidgets.QActionGroup(self)
        current = theme.saved()
        for name in theme.THEMES:
            action = self.menuTheme.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == current)
            action.triggered.connect(lambda _, n=name: theme.apply(QtWidgets.QApplication.instance(), n))
            group.addAction(action)

    # --- plan table ---

    def selected_names(self):
        rows = sorted({i.row() for i in self.planTable.selectedIndexes()})
        return [self.planTable.item(r, 0).data(Qt.UserRole) for r in rows]

    def current_plan(self):
        names = self.selected_names()
        return self.plans.plan(names[0]) if len(names) == 1 else None

    def refresh_plans(self, select=()):
        # the table is always just a copy of what's in self.plans
        table = self.planTable
        table.blockSignals(True)
        table.setSortingEnabled(False)  # or rows jump around while they're filled in
        table.setRowCount(len(self.plans))
        for row, plan in enumerate(self.plans):
            cells = [plan.name, plan.kind, plan.n_units, plan.n_districts, plan.source]
            for col, value in enumerate(cells):
                item = QtWidgets.QTableWidgetItem()
                item.setData(Qt.DisplayRole, value)  # numbers as numbers so they sort right
                if col == 0:
                    item.setData(Qt.UserRole, plan.name)
                table.setItem(row, col, item)
        table.setSortingEnabled(True)
        table.clearSelection()
        rows = QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows
        for row in range(table.rowCount()):
            if table.item(row, 0).data(Qt.UserRole) in select:
                table.selectionModel().select(table.model().index(row, 0), rows)
        table.blockSignals(False)
        self.planHint.setVisible(not len(self.plans))
        self.show_selected()

    def show_selected(self):
        names = self.selected_names()
        if len(names) == 1:
            plan = self.plans.plan(names[0])
            self.planInfoLabel.setText("\n".join([f"{plan.name}: {plan.summary()}", *plan.notes]))
        elif names:
            self.planInfoLabel.setText(f"{len(names)} plans selected")
        else:
            self.planInfoLabel.setText(f"{len(self.plans)} plans loaded" if len(self.plans) else "")
        self.update_ready()

    def remove_plans(self):
        names = self.selected_names()
        if not names:
            return
        what = names[0] if len(names) == 1 else f"these {len(names)} plans"
        answer = QtWidgets.QMessageBox.question(self, "Remove Plans",
                                                f"Remove {what} from this session?")
        if answer == QtWidgets.QMessageBox.Yes:
            for name in names:
                self.plans.remove(name)
            self.refresh_plans()

    # --- importing plans ---

    def import_plans(self):
        paths = QtWidgets.QFileDialog.getOpenFileNames(self, "Import Plans", self.last_dir,
                                                       PLAN_FILE_FILTER)[0]
        if paths:
            self.last_dir = str(Path(paths[0]).parent)
            self.import_paths(paths)

    def import_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Import Folder", self.last_dir)
        if folder:
            self.last_dir = folder
            self.import_paths([folder])

    def import_paths(self, paths):
        if self.task is not None:
            return
        self.start(lambda say: (say("Reading plans..."), read_plan_files(paths))[1], self.plans_read)

    def clash_choice(self, clashes):
        """what to do with plans whose names are already loaded: 'replace', 'both', 'skip' or None"""
        shown = ", ".join(clashes[:MAX_LISTED]) + (" ..." if len(clashes) > MAX_LISTED else "")
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Plans Already Loaded",
                                    f"{len(clashes)} of these plans are already loaded: {shown}",
                                    parent=self)
        buttons = {box.addButton("Replace", QtWidgets.QMessageBox.AcceptRole): "replace",
                   box.addButton("Keep Both", QtWidgets.QMessageBox.ActionRole): "both",
                   box.addButton("Skip", QtWidgets.QMessageBox.ActionRole): "skip"}
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        return buttons.get(box.clickedButton())

    def plans_read(self, result):
        plans, problems = result
        clashes = [p.name for p in plans if p.name in self.plans]
        choice = self.clash_choice(clashes) if clashes else "both"
        if choice is None:
            self.statusBar.showMessage("Import cancelled.")
            return

        added, seen = [], set()
        for plan in plans:
            clash = plan.name in self.plans
            if clash and choice == "skip":
                continue
            # same name twice in one import, or "keep both": the newer one gets a _2
            if plan.name in seen or (clash and choice == "both"):
                plan.name = self.plans.free_name(plan.name)
            self.plans.add(plan, replace=True)
            seen.add(plan.name)
            added.append(plan.name)
        self.refresh_plans(select=added[-1:])
        self.statusBar.showMessage(f"Imported {len(added)} plan{'s' if len(added) != 1 else ''}.")
        if problems:
            self.show_problems(len(added), problems)

    def show_problems(self, n_added, problems):
        lines = [f"{Path(path).name}: {why}" for path, why in problems]
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, "Some Files Were Skipped",
                                    f"Imported {n_added} plan{'s' if n_added != 1 else ''}. "
                                    f"{len(problems)} file{'s' if len(problems) != 1 else ''} "
                                    "couldn't be read as plans.", parent=self)
        box.setInformativeText("\n".join(lines[:MAX_LISTED])
                               + (f"\n... and {len(lines) - MAX_LISTED} more"
                                  if len(lines) > MAX_LISTED else ""))
        if len(lines) > MAX_LISTED:
            box.setDetailedText("\n".join(f"{path}: {why}" for path, why in problems))
        box.exec()

    # drag files or folders onto the window to import them
    def dragEnterEvent(self, event):
        if self.task is None and any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.import_paths(paths)

    # --- precinct data ---

    def refresh_data(self):
        self.dataList.clear()
        for label, table in self.precinct_data.items():
            shapes = ", with shapes" if has_shapes(table) else ""
            item = QtWidgets.QListWidgetItem(
                f"{label}: {len(table):,} rows, {len(table.columns)} columns{shapes}", self.dataList)
            item.setData(Qt.UserRole, label)
        self.dataHint.setVisible(not self.precinct_data)
        self.update_ready()

    def add_data(self, label, table, fips=None):
        replaced = label in self.precinct_data
        self.precinct_data[label] = table
        self.data_state[label] = fips
        self.refresh_data()
        self.statusBar.showMessage(f"{'Replaced' if replaced else 'Added'} {label}.")

    def add_data_file(self):
        if self.task is not None:
            return
        paths = QtWidgets.QFileDialog.getOpenFileNames(self, "Add Precinct Data Files",
                                                       self.last_dir, DATA_FILE_FILTER)[0]
        if not paths:
            return
        self.last_dir = str(Path(paths[0]).parent)

        def read_all(say):  # one bad file doesn't stop the rest
            tables, problems = [], []
            for path in paths:
                say(f"Reading {Path(path).name}...")
                try:
                    table = read_table(path)
                    find_key(table, Path(path).name)  # no GEOID column: say so now, not in Metrics
                    tables.append((Path(path).stem, table))
                except ReconcileError as e:
                    problems.append((path, str(e).split(": ", 1)[1]))
                except Exception as e:
                    problems.append((path, f"{type(e).__name__}: {e}"))
            return tables, problems

        def loaded(result):
            tables, problems = result
            for label, table in tables:
                self.add_data(label, table)
            self.statusBar.showMessage(f"Added {len(tables)} data file"
                                       f"{'s' if len(tables) != 1 else ''}.")
            if problems:
                lines = [f"{Path(path).name}: {why}" for path, why in problems]
                QtWidgets.QMessageBox.warning(self, "Some Files Were Skipped",
                                              f"{len(problems)} file{'s' if len(problems) != 1 else ''}"
                                              " couldn't be read:\n\n" + "\n".join(lines[:MAX_LISTED]))

        self.start(read_all, loaded)

    def download_state(self):
        if self.task is not None:
            return
        dialog = StateDataDialog(self.precinct_data, self.last_state, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        abbr, keys = dialog.state(), dialog.chosen()
        self.last_state = abbr

        def fetch(say):  # each one is cached on disk, so a retry after a failure is quick
            return [(key, SOURCES[key][1](abbr, say=say)) for key in keys]

        def loaded(results):
            for key, table in results:
                self.add_data(LABELS[key].format(abbr=abbr), table, STATES[abbr][1])
            self.statusBar.showMessage(f"Added {len(results)} {STATES[abbr][0]} data set"
                                       f"{'s' if len(results) != 1 else ''}.")

        self.start(fetch, loaded)

    def remove_data(self):
        for item in self.dataList.selectedItems():
            self.precinct_data.pop(item.data(Qt.UserRole), None)
            self.data_state.pop(item.data(Qt.UserRole), None)
        self.refresh_data()

    # --- analyses (new ones get a button next to Metrics and a method like this) ---

    def analysis_table(self, plan):
        """the plan joined to every loaded precinct data set, or None if it can't be.
        the plan itself is left as it is"""
        # downloads only join plans from their own state; your own files always join
        fips = plan.assignment["GEOID"].str[:2].mode()[0]
        data = {label: table for label, table in self.precinct_data.items()
                if self.data_state.get(label) in (None, fips)}
        if not data:
            where = STATES[FIPS_TO_ABBR[fips]][0] if fips in FIPS_TO_ABBR else "this plan's state"
            self.statusBar.showMessage(f"No precinct data loaded for {where}: only the plan's "
                                       "districts are available to the analysis.")
            return plan.assignment.copy()
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # your own files can be from another state
            own = {label for label in data if self.data_state.get(label) is None}
            result = reconcile(plan.assignment, extras=data, optional=own)
        except ReconcileError as e:
            hint = ("\n\nThis plan is a BAF (Census blocks); the precinct data may be by voting "
                    "district." if plan.kind == "BAF" else "")
            QtWidgets.QMessageBox.critical(self, "Couldn't Join Precinct Data", f"{e}{hint}")
            return None
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar.showMessage("; ".join(result.notes))
        return result.data

    def open_metrics(self):
        plan = self.current_plan()
        if plan is None:
            return
        table = self.analysis_table(plan)
        if table is None:
            return
        dialog = MetricsDialog(self, data=table, name=plan.name)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.show()

    # --- background work ---

    def start(self, func, on_done):
        self.task = Task(func, self)
        self.task.progress.connect(self.statusBar.showMessage)
        self.task.done.connect(lambda result: self.finish(on_done, result))
        self.task.failed.connect(lambda message: self.finish(self.failed, message))
        self.update_ready()
        self.task.start()

    def finish(self, then, value):
        self.task.wait()  # the thread is done emitting, let it finish before we drop it
        self.task = None
        self.statusBar.clearMessage()
        self.update_ready()
        then(value)

    def failed(self, message):
        QtWidgets.QMessageBox.critical(self, "Something Went Wrong", message)

    def update_ready(self):
        idle = self.task is None
        n_selected = len(self.selected_names())
        states = [
            (idle, self.importButton, self.actionImportPlans),
            (idle, self.importFolderButton, self.actionImportFolder),
            (n_selected == 1, self.metricsButton, self.actionMetrics),
            (n_selected > 0, self.removeButton, self.actionRemovePlans),
            (idle, self.addDataButton, self.actionAddData),
            (idle, self.stateDataButton, self.actionDownloadState),
            (bool(self.dataList.selectedItems()), self.removeDataButton, self.actionRemoveData),
        ]
        for enabled, button, action in states:
            button.setEnabled(enabled)
            action.setEnabled(enabled)

    def closeEvent(self, event):
        if self.task is not None:
            self.statusBar.showMessage("Still working; close once it has finished.")
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    theme.apply(app)
    window = MainWindow()
    window.show()
    app.exec()

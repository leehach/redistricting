"""Main app window. run with: python app.py

Plans get imported from disk or from the repo, reconciled into one table each, and kept in
self.plans (a SessionPlans) for as long as the window is open. The list in the middle shows
what's in there, the buttons under it run things on the selected plan.
"""

import os
import sys
from pathlib import Path

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from pa_redistricting.census import fetch_census
from pa_redistricting.config import PLANS_FOLDER, PLANS_OWNER, PLANS_REPO
from pa_redistricting.election_results import fetch_elections
from pa_redistricting.plan_assignments import (download_text, list_plan_files, plan_name,
                                               read_assignment)
from pa_redistricting.voting_districts import load_vtds

from main import FILE_FILTER, MetricsDialog
from main_window_ui import Ui_MainWindow  # the layout from Qt Designer
from reconcile import ReconcileError, read_table, reconcile
from session import SessionPlans

ROLES = {"assignments": "district assignments", "voting": "voting data",
         "demographics": "demographic data"}


# --- the slow parts, run on a background thread. say(text) updates the status bar ---

def load_from_disk(paths, say):
    """paths = {"assignments": file, "voting": file, "demographics": file}, last two optional"""
    tables = {}
    for role, path in paths.items():
        say(f"Reading {Path(path).name}...")
        try:
            tables[role] = read_table(path)
        except Exception as e:
            raise ReconcileError(f"Couldn't read the {ROLES[role]} file {Path(path).name}:\n{e}")
    say("Reconciling...")
    return reconcile(tables["assignments"], voting=tables.get("voting"),
                     demographics=tables.get("demographics"))


def load_from_repo(repo_paths, with_census, cache, say):
    """one reconciled plan per repo file. shapes/votes/census are downloaded once and kept in cache"""
    if "shapes" not in cache:
        say("Downloading voting district shapes (big file, takes a bit)...")
        cache["shapes"] = load_vtds()
    if "votes" not in cache:
        say("Downloading election results...")
        cache["votes"] = fetch_elections()
    if with_census and "census" not in cache:
        say("Downloading Census data, one county at a time...")
        cache["census"] = fetch_census()
    plans = []
    for i, path in enumerate(repo_paths, 1):
        say(f"Downloading plan {i} of {len(repo_paths)}: {plan_name(path)}...")
        result = reconcile(read_assignment(download_text(path)), voting=cache["votes"],
                           demographics=cache["census"] if with_census else None,
                           shapes=cache["shapes"])
        if not with_census:
            result.notes.append("demographic data: not included")
        plans.append((path, result))
    return plans


class Task(QtCore.QThread):
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(str)

    def __init__(self, func, parent=None):
        super().__init__(parent)
        self.func = func

    def run(self):
        try:
            self.done.emit(self.func(self.progress.emit))
        except ReconcileError as e:  # these are already written for the user
            self.failed.emit(str(e))
        except Exception as e:
            self.failed.emit(f"{type(e).__name__}: {e}")


# --- picking plans out of the repo ---

class RepoDialog(QtWidgets.QDialog):
    def __init__(self, repo_paths, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import from repo")
        self.resize(520, 480)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            f"{len(repo_paths)} plans in {PLANS_OWNER}/{PLANS_REPO}/{PLANS_FOLDER}. "
            "Ctrl or Shift click to pick more than one."))
        self.filterLineEdit = QtWidgets.QLineEdit(placeholderText="type to filter")
        layout.addWidget(self.filterLineEdit)
        self.planList = QtWidgets.QListWidget()
        self.planList.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for path in repo_paths:
            item = QtWidgets.QListWidgetItem(plan_name(path), self.planList)
            item.setData(Qt.UserRole, path)
        layout.addWidget(self.planList)

        has_key = bool(os.environ.get("CENSUS_API_KEY", "").strip())
        self.censusCheckBox = QtWidgets.QCheckBox("Include Census demographics (slow the first time)")
        self.censusCheckBox.setEnabled(has_key)
        self.censusCheckBox.setChecked(has_key)
        if not has_key:
            self.censusCheckBox.setText("Include Census demographics (needs CENSUS_API_KEY set)")
        layout.addWidget(self.censusCheckBox)

        self.buttonBox = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setText("Import")
        layout.addWidget(self.buttonBox)

        self.filterLineEdit.textChanged.connect(self.apply_filter)
        self.planList.itemSelectionChanged.connect(self.update_ready)
        self.planList.itemDoubleClicked.connect(lambda item: self.accept())
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.update_ready()

    def apply_filter(self, text):
        for i in range(self.planList.count()):
            item = self.planList.item(i)
            item.setHidden(text.strip().lower() not in item.text().lower())

    def update_ready(self):
        self.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(bool(self.chosen()))

    def chosen(self):
        return [i.data(Qt.UserRole) for i in self.planList.selectedItems() if not i.isHidden()]


# --- the window ---

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, plans=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.plans = SessionPlans() if plans is None else plans  # the in-memory plans object
        self.task, self.last_dir = None, "."
        self.repo_paths, self.repo_cache = None, {}  # repo downloads, kept for the session

        # signals and slots
        self.path_edits = {"assignments": self.assignmentsLineEdit, "voting": self.votingLineEdit,
                           "demographics": self.demographicsLineEdit}
        browse = {"assignments": self.assignmentsBrowseButton, "voting": self.votingBrowseButton,
                  "demographics": self.demographicsBrowseButton}
        for role, edit in self.path_edits.items():
            browse[role].clicked.connect(lambda _, role=role: self.browse(role))
            edit.returnPressed.connect(self.add_from_disk)
        self.nameLineEdit.returnPressed.connect(self.add_from_disk)
        self.assignmentsLineEdit.textChanged.connect(self.update_ready)
        self.addPlanButton.clicked.connect(self.add_from_disk)
        self.repoImportButton.clicked.connect(self.import_from_repo)
        self.planList.itemSelectionChanged.connect(self.show_selected)
        self.planList.itemDoubleClicked.connect(lambda item: self.open_metrics())
        self.metricsButton.clicked.connect(self.open_metrics)
        self.removeButton.clicked.connect(self.remove_selected)

        self.refresh_list()
        self.update_ready()

    # --- plan list ---

    def selected(self):
        item = self.planList.currentItem()
        return self.plans.plan(item.text()) if item is not None and item.text() in self.plans else None

    def refresh_list(self, select=None):
        # the list is always just a copy of what's in self.plans
        if select is None and self.planList.currentItem() is not None:
            select = self.planList.currentItem().text()
        self.planList.blockSignals(True)
        self.planList.clear()
        self.planList.addItems(self.plans.names)
        self.planList.blockSignals(False)
        if select in self.plans:
            self.planList.setCurrentRow(self.plans.names.index(select))
        self.show_selected()

    def show_selected(self):
        plan = self.selected()
        for button in (self.metricsButton, self.removeButton):  # analysis buttons need a plan
            button.setEnabled(plan is not None)
        if plan is None:
            self.planInfoLabel.setText("Select a plan to work with it." if len(self.plans)
                                       else "No plans yet. Import one above.")
            return
        lines = [f"{plan.name} (from {plan.source}): {plan.summary()}", *plan.notes]
        self.planInfoLabel.setText("\n".join(lines))

    def remove_selected(self):
        plan = self.selected()
        if plan is None:
            return
        answer = QtWidgets.QMessageBox.question(self, "Remove plan",
                                                f"Remove {plan.name} from this session?")
        if answer == QtWidgets.QMessageBox.Yes:
            self.plans.remove(plan.name)
            self.refresh_list()

    def pick_name(self, name):
        """what to save a plan as if the name is taken. None = user cancelled"""
        if name not in self.plans:
            return name
        box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Question, "Plan already loaded",
                                    f"There is already a plan named {name}.", parent=self)
        replace = box.addButton("Replace it", QtWidgets.QMessageBox.AcceptRole)
        both = box.addButton("Keep both", QtWidgets.QMessageBox.ActionRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is replace:
            return name
        if box.clickedButton() is both:
            return self.plans.free_name(name)
        return None

    # --- analyses (new ones get a button next to Metrics and a method like this) ---

    def open_metrics(self):
        plan = self.selected()
        if plan is None:
            return
        dialog = MetricsDialog(self, data=plan.data, name=plan.name)
        dialog.setAttribute(Qt.WA_DeleteOnClose)
        dialog.show()

    # --- import from disk ---

    def browse(self, role):
        fn = QtWidgets.QFileDialog.getOpenFileName(self, f"Open {ROLES[role]}", self.last_dir,
                                                   FILE_FILTER)[0]
        if fn:
            self.last_dir = str(Path(fn).parent)
            self.path_edits[role].setText(fn)

    def typed_paths(self):
        # strip the quotes windows adds with "Copy as path"
        paths = {role: edit.text().strip().strip('"') for role, edit in self.path_edits.items()}
        return {role: path for role, path in paths.items() if path}

    def add_from_disk(self):
        if self.task is not None:
            return
        paths = self.typed_paths()
        if "assignments" not in paths:
            self.statusBar.showMessage("Choose a district assignments file first.")
            return
        missing = [f"{ROLES[role]}: {path}" for role, path in paths.items() if not Path(path).is_file()]
        if missing:
            QtWidgets.QMessageBox.warning(self, "File not found", "\n".join(missing))
            return
        name = self.pick_name(self.nameLineEdit.text().strip() or Path(paths["assignments"]).stem)
        if name is None:
            return
        self.start(lambda say: load_from_disk(paths, say),
                   lambda result: self.disk_loaded(name, result))

    def disk_loaded(self, name, result):
        self.plans.add(name, result.data, source="disk", notes=result.notes, replace=True)
        for edit in (*self.path_edits.values(), self.nameLineEdit):
            edit.clear()
        self.refresh_list(select=name)
        self.statusBar.showMessage(f"Added {name}.")

    # --- import from repo ---

    def import_from_repo(self):
        if self.task is not None:
            return
        if self.repo_paths is None:  # get the list of plans first, then come back here
            self.start(lambda say: (say("Getting the list of plans..."), list_plan_files())[1],
                       self.repo_listed)
            return
        dialog = RepoDialog(self.repo_paths, self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        names = {}  # repo path -> name to save it as
        for path in dialog.chosen():
            name = self.pick_name(plan_name(path))
            if name is not None:
                names[path] = name
        if not names:
            return
        with_census, cache = dialog.censusCheckBox.isChecked(), self.repo_cache
        self.start(lambda say: load_from_repo(list(names), with_census, cache, say),
                   lambda loaded: self.repo_loaded(names, loaded))

    def repo_listed(self, repo_paths):
        self.repo_paths = repo_paths
        self.import_from_repo()

    def repo_loaded(self, names, loaded):
        for path, result in loaded:
            self.plans.add(names[path], result.data, source="repo", notes=result.notes, replace=True)
        self.refresh_list(select=names[loaded[-1][0]])
        self.statusBar.showMessage(f"Added {len(loaded)} plan{'s' if len(loaded) != 1 else ''} "
                                   "from the repo.")

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
        QtWidgets.QMessageBox.critical(self, "Import failed", message)

    def update_ready(self):
        # no imports while one is running
        busy = self.task is not None
        self.diskGroupBox.setEnabled(not busy)
        self.repoGroupBox.setEnabled(not busy)
        self.addPlanButton.setEnabled(bool(self.assignmentsLineEdit.text().strip()))

    def closeEvent(self, event):
        if self.task is not None:
            self.statusBar.showMessage("Still importing; close once it has finished.")
            event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

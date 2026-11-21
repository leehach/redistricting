"""App themes, picked from View > Theme. Colors follow VS Code's built-in themes: window = side bar,
base = editor/inputs, surface = buttons and headers. call apply(app) right after making the QApplication
"""

from dataclasses import dataclass

from PyQt5 import QtCore, QtGui, QtWidgets


@dataclass(frozen=True)
class Theme:
    window: str       # background
    base: str         # inputs, lists, tables
    surface: str      # buttons, table headers, tooltips
    text: str
    border: str
    accent: str       # ticked boxes, focus
    select: str       # selected rows
    select_text: str


THEMES = {
    "Light Modern": Theme("#F8F8F8", "#FFFFFF", "#E5E5E5", "#3B3B3B", "#CECECE", "#005FB8",
                          "#E4E6F1", "#000000"),
    "Dark Modern": Theme("#181818", "#1F1F1F", "#313131", "#CCCCCC", "#3C3C3C", "#0078D4",
                         "#04395E", "#FFFFFF"),
    "Light+": Theme("#F3F3F3", "#FFFFFF", "#E7E7E7", "#333333", "#CECECE", "#007ACC",
                    "#0060C0", "#FFFFFF"),
    "Dark+": Theme("#252526", "#1E1E1E", "#3C3C3C", "#CCCCCC", "#454545", "#007ACC",
                   "#04395E", "#FFFFFF"),
    "Quiet Light": Theme("#F2F2F2", "#FFFFFF", "#E7E7E7", "#333333", "#CCCCCC", "#705697",
                         "#C4D9B1", "#333333"),
    "Solarized Light": Theme("#EEE8D5", "#FDF6E3", "#DDD6C1", "#586E75", "#D3CBB7", "#B58900",
                             "#DFCA88", "#3E4B50"),
    "Solarized Dark": Theme("#00212B", "#002B36", "#003847", "#93A1A1", "#0D4A59", "#2AA198",
                            "#005A6F", "#FFFFFF"),
    "Monokai": Theme("#1E1F1C", "#272822", "#414339", "#F8F8F2", "#3B3C35", "#A6E22E",
                     "#75715E", "#F8F8F2"),
    "Abyss": Theme("#060621", "#000C18", "#181F2F", "#6688CC", "#2B3C5D", "#225588",
                   "#08286B", "#FFFFFF"),
}
DEFAULT = "Light Modern"
SETTING = "theme"


def _mix(a: str, b: str, t: float) -> QtGui.QColor:
    """t of the way from color a to color b"""
    a, b = QtGui.QColor(a), QtGui.QColor(b)
    return QtGui.QColor(*(round(x + (y - x) * t) for x, y in
                          zip(a.getRgb()[:3], b.getRgb()[:3])))


def palette(t: Theme) -> QtGui.QPalette:
    p = QtGui.QPalette()
    roles = {
        QtGui.QPalette.Window: t.window, QtGui.QPalette.WindowText: t.text,
        QtGui.QPalette.Base: t.base, QtGui.QPalette.AlternateBase: _mix(t.base, t.text, 0.04),
        QtGui.QPalette.Text: t.text, QtGui.QPalette.Button: t.surface,
        QtGui.QPalette.ButtonText: t.text, QtGui.QPalette.BrightText: t.select_text,
        QtGui.QPalette.Highlight: t.select, QtGui.QPalette.HighlightedText: t.select_text,
        QtGui.QPalette.ToolTipBase: t.surface, QtGui.QPalette.ToolTipText: t.text,
        QtGui.QPalette.PlaceholderText: _mix(t.text, t.base, 0.45), QtGui.QPalette.Link: t.accent,
        # Fusion draws bevels and frames from these
        QtGui.QPalette.Light: _mix(t.surface, "#FFFFFF", 0.15), QtGui.QPalette.Midlight: t.surface,
        QtGui.QPalette.Mid: t.border, QtGui.QPalette.Dark: _mix(t.border, "#000000", 0.25),
        QtGui.QPalette.Shadow: _mix(t.border, "#000000", 0.5),
    }
    for role, color in roles.items():
        p.setColor(role, QtGui.QColor(color))
    disabled = _mix(t.text, t.window, 0.55)
    for role in (QtGui.QPalette.WindowText, QtGui.QPalette.Text, QtGui.QPalette.ButtonText):
        p.setColor(QtGui.QPalette.Disabled, role, disabled)
    return p


def style(t: Theme) -> str:
    part = _mix(t.accent, t.base, 0.5).name()
    return f"""
QHeaderView::section {{ background: {t.surface}; color: {t.text}; border: none;
    border-right: 1px solid {t.border}; border-bottom: 1px solid {t.border}; padding: 4px 6px; }}
QToolTip {{ background: {t.surface}; color: {t.text}; border: 1px solid {t.border}; }}
QAbstractItemView::indicator {{ width: 12px; height: 12px; border: 1px solid
    {_mix(t.text, t.base, 0.4).name()}; border-radius: 2px; background: {t.base}; }}
QAbstractItemView::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}
QAbstractItemView::indicator:indeterminate {{ background: {part}; border-color: {t.accent}; }}
"""


def _settings():
    return QtCore.QSettings("redistricting", "metrics_gui")


def saved() -> str:
    """the theme picked last time, or the default"""
    try:
        name = _settings().value(SETTING, DEFAULT)
    except Exception:
        name = DEFAULT
    return name if name in THEMES else DEFAULT


def apply(app: QtWidgets.QApplication, name: str | None = None):
    """switch the whole app to a theme (the saved one if no name) and remember it"""
    name = name if name in THEMES else saved()
    t = THEMES[name]
    app.setStyle("Fusion")  # the Windows style ignores palette colors on buttons and headers
    app.setPalette(palette(t))
    QtWidgets.QToolTip.setPalette(palette(t))
    app.setStyleSheet(style(t))
    try:
        _settings().setValue(SETTING, name)
    except Exception:
        pass
    return name

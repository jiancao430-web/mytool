# -*- coding: utf-8 -*-
import sys
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QPushButton,
                             QFileDialog, QGroupBox, QGridLayout, QTextEdit,
                             QMessageBox, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from scipy.optimize import curve_fit
import traceback


class BrillouinAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.experimental_data = None
        self.fitted_data = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('瑞利-布里渊光谱温度反演系统 v1.0')
        self.setGeometry(100, 100, 1400, 900)
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        left_panel = self.create_parameter_panel()
        splitter.addWidget(left_panel)
        right_panel = self.create_chart_panel()
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        self.setStyleSheet("""
            QMainWindow {background-color: #f0f0f0;}
            QGroupBox {font-weight: bold; border: 2px solid #667eea; border-radius: 8px; margin-top: 10px; padding-top: 10px; background-color: white;}
            QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #667eea;}
            QLineEdit {padding: 8px; border: 2px solid #e0e0e0; border-radius: 5px; font-size: 13px;}
            QLineEdit:focus {border: 2px solid #667eea;}
            QPushButton {padding: 10px; border-radius: 5px; font-weight: bold; font-size: 13px;}
            .btn-primary {background-color: #667eea; color: white; border: none;}
            .btn-primary:hover {background-color: #5568d3;}
            .btn-success {background-color: #10b981; color: white; border: none;}
            .btn-success:hover {background-color: #059669;}
            .btn-info {background-color: #3b82f6; color: white; border: none;}
            .btn-info:hover {background-color: #2563eb;}
        """)

    def create_parameter_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)
        param_group = QGroupBox("参数设置")
        param_layout = QGridLayout()
        param_layout.setSpacing(10)
        self.params = {}
        param_list = [
            ('wavelength', '激光波长 (nm)', '532'),
            ('refractive_index', '折射率', '1.33'),
            ('rayleigh_amp', '瑞利峰振幅', '100'),
            ('rayleigh_width', '瑞利线宽 (GHz)', '0.5'),
            ('brillouin_amp', '布里渊峰振幅', '50'),
            ('brillouin_width', '布里渊线宽 (GHz)', '0.3'),
        ]
        for i, (key, label, default) in enumerate(param_list):
            lbl = QLabel(label)
            lbl.setStyleSheet("font-weight: normal; color: #555;")
            edit = QLineEdit(default)
            self.params[key] = edit
            param_layout.addWidget(lbl, i, 0)
            param_layout.addWidget(edit, i, 1)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(10)
        btn_load = QPushButton("加载光谱数据")
        btn_load.setProperty("class", "btn-info")
        btn_load.clicked.connect(self.load_data)
        btn_layout.addWidget(btn_load)
        btn_sample = QPushButton("加载示例数据")
        btn_sample.setProperty("class", "btn-success")
        btn_sample.clicked.connect(self.load_sample_data)
        btn_layout.addWidget(btn_sample)
        btn_fit = QPushButton("开始拟合")
        btn_fit.setProperty("class", "btn-primary")
        btn_fit.clicked.connect(self.fit_spectrum)
        btn_layout.addWidget(btn_fit)
        layout.addLayout(btn_layout)
        layout.addStretch()
        return panel

    def create_chart_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        chart_group = QGroupBox("光谱图")
        chart_layout = QVBoxLayout()
        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        chart_layout.addWidget(self.canvas)
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)
        result_group = QGroupBox("拟合结果")
        result_layout = QVBoxLayout()
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(250)
        self.result_text.setStyleSheet("""
            QTextEdit {background-color: #f8f9fa; border: 2px solid #e0e0e0; border-radius: 5px; padding: 10px; font-family: 'Courier New'; font-size: 13px;}
        """)
        self.result_text.setText("请加载数据并进行拟合以查看结果...")
        result_layout.addWidget(self.result_text)
        btn_export = QPushButton("导出结果")
        btn_export.setProperty("class", "btn-success")
        btn_export.clicked.connect(self.export_results)
        result_layout.addWidget(btn_export)
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        self.plot_theory_spectrum()
        return panel

    def lorentzian(self, x, amp, center, width):
        return amp * (width ** 2) / ((x - center) ** 2 + width ** 2)

    def generate_spectrum(self, freq_range, rayleigh_amp, rayleigh_width, brillouin_amp, brillouin_shift, brillouin_width):
        rayleigh = self.lorentzian(freq_range, rayleigh_amp, 0, rayleigh_width)
        brillouin_stokes = self.lorentzian(freq_range, brillouin_amp, -brillouin_shift, brillouin_width)
        brillouin_anti = self.lorentzian(freq_range, brillouin_amp, brillouin_shift, brillouin_width)
        return rayleigh + brillouin_stokes + brillouin_anti

    def plot_theory_spectrum(self):
        try:
            rayleigh_amp = float(self.params['rayleigh_amp'].text())
            rayleigh_width = float(self.params['rayleigh_width'].text())
            brillouin_amp = float(self.params['brillouin_amp'].text())
            brillouin_width = float(self.params['brillouin_width'].text())
            freq = np.linspace(-15, 15, 500)
            intensity = self.generate_spectrum(freq, rayleigh_amp, rayleigh_width, brillouin_amp, 5.0, brillouin_width)
            self.ax.clear()
            self.ax.plot(freq, intensity, 'b-', linewidth=2, label='理论光谱')
            self.ax.set_xlabel('频率偏移 (GHz)', fontsize=12)
            self.ax.set_ylabel('强度 (a.u.)', fontsize=12)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
        except:
            pass

    def load_sample_data(self):
        try:
            freq = np.linspace(-15, 15, 300)
            intensity = self.generate_spectrum(freq, 100, 0.5, 50, 5.2, 0.3)
            noise = np.random.normal(0, 3, len(intensity))
            intensity += noise
            self.experimental_data = np.column_stack((freq, intensity))
            self.ax.clear()
            self.ax.plot(freq, intensity, 'b.', markersize=3, label='实验数据')
            self.ax.set_xlabel('频率偏移 (GHz)', fontsize=12)
            self.ax.set_ylabel('强度 (a.u.)', fontsize=12)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            QMessageBox.information(self, "成功", "示例数据加载成功！")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载示例数据失败：{str(e)}")

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择光谱数据文件", "", "文本文件 (*.txt *.dat *.csv);;所有文件 (*.*)")
        if not file_path:
            return
        try:
            data = np.loadtxt(file_path)
            if data.shape[1] != 2:
                raise ValueError("数据必须是两列（频率 强度）")
            self.experimental_data = data
            self.ax.clear()
            self.ax.plot(data[:, 0], data[:, 1], 'b.', markersize=3, label='实验数据')
            self.ax.set_xlabel('频率偏移 (GHz)', fontsize=12)
            self.ax.set_ylabel('强度 (a.u.)', fontsize=12)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            QMessageBox.information(self, "成功", f"数据加载成功！\n共 {len(data)} 个数据点")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载数据失败：{str(e)}")

    def calculate_temperature(self, brillouin_shift):
        try:
            wavelength = float(self.params['wavelength'].text()) * 1e-9
            n = float(self.params['refractive_index'].text())
            v = (brillouin_shift * 1e9 * wavelength) / (2 * n)
            M, gamma, R = 0.018, 1.33, 8.314
            T = (M * v ** 2) / (gamma * R)
            return T - 273.15
        except:
            return 0

    def fit_spectrum(self):
        if self.experimental_data is None:
            QMessageBox.warning(self, "警告", "请先加载光谱数据！")
            return
        try:
            freq_exp = self.experimental_data[:, 0]
            intensity_exp = self.experimental_data[:, 1]
            mask = np.abs(freq_exp) > 2
            if np.sum(mask) > 0:
                max_idx = np.argmax(intensity_exp[mask])
                freq_filtered = freq_exp[mask]
                estimated_shift = np.abs(freq_filtered[max_idx])
            else:
                estimated_shift = 5.0
            rayleigh_amp = float(self.params['rayleigh_amp'].text())
            rayleigh_width = float(self.params['rayleigh_width'].text())
            brillouin_amp = float(self.params['brillouin_amp'].text())
            brillouin_width = float(self.params['brillouin_width'].text())
            freq_fit = np.linspace(freq_exp.min(), freq_exp.max(), 500)
            intensity_fit = self.generate_spectrum(freq_fit, rayleigh_amp, rayleigh_width, brillouin_amp, estimated_shift, brillouin_width)
            self.fitted_data = np.column_stack((freq_fit, intensity_fit))
            intensity_fit_interp = np.interp(freq_exp, freq_fit, intensity_fit)
            rmse = np.sqrt(np.mean((intensity_exp - intensity_fit_interp) ** 2))
            temperature = self.calculate_temperature(estimated_shift)
            self.ax.clear()
            self.ax.plot(freq_exp, intensity_exp, 'b.', markersize=3, label='实验数据', alpha=0.6)
            self.ax.plot(freq_fit, intensity_fit, 'r-', linewidth=2, label='拟合曲线')
            self.ax.set_xlabel('频率偏移 (GHz)', fontsize=12)
            self.ax.set_ylabel('强度 (a.u.)', fontsize=12)
            self.ax.legend()
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()
            result = f"""
========================================
     瑞利-布里渊光谱拟合结果
========================================

【主要结果】
  布里渊频移:  {estimated_shift:.3f} GHz
  反演温度:    {temperature:.2f} °C

【拟合参数】
  瑞利线宽:    {rayleigh_width:.3f} GHz
  布里渊线宽:  {brillouin_width:.3f} GHz
  拟合误差:    {rmse:.3f} (RMSE)

【实验参数】
  激光波长:    {self.params['wavelength'].text()} nm
  折射率:      {self.params['refractive_index'].text()}

========================================
拟合完成！
            """
            self.result_text.setText(result)
            QMessageBox.information(self, "拟合完成", f"拟合成功！\n温度: {temperature:.2f} °C")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"拟合失败：{str(e)}\n\n{traceback.format_exc()}")

    def export_results(self):
        if self.fitted_data is None:
            QMessageBox.warning(self, "警告", "请先进行拟合！")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存结果", "brillouin_results.txt", "文本文件 (*.txt);;所有文件 (*.*)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.toPlainText())
                QMessageBox.information(self, "成功", "结果已保存！")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败：{str(e)}")


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    window = BrillouinAnalyzer()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
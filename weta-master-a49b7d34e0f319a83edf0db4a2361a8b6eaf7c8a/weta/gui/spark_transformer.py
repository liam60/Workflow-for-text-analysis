import pyspark.sql
import pyspark.ml
from Orange.widgets import widget, gui
from PyQt5 import QtWidgets, QtGui
from .spark_base import SparkBase, Parameter


class SparkTransformer(SparkBase):

    # -----------Inputs / Outputs ---------------------
    DataFrame = None

    class Inputs:
        DataFrame = widget.Input("DataFrame", pyspark.sql.DataFrame)

    class Outputs:
        DataFrame = widget.Output("DataFrame", pyspark.sql.DataFrame)
        Transformer = widget.Output("Transformer", pyspark.ml.Transformer)

    learner = None  # type: type
    input_dtype = None

    # -------------- Layout config ---------------
    want_main_area = False
    resizing_enabled = True

    def __init__(self):
        super(SparkTransformer, self).__init__()

        doc = self.learner.__doc__ if self.learner is not None else ''

        self.v_help_box = gui.widgetBox(self.v_main_box, 'Documentation', addSpace=True)
        self.v_help_box.setMinimumWidth(400)

        # Create doc info.
        self.v_doc_text = QtWidgets.QTextEdit('<pre>' + doc + '</pre>', self.v_help_box)
        self.v_doc_text.setAcceptRichText(True)
        self.v_doc_text.setReadOnly(True)
        self.v_doc_text.autoFormatting()
        self.v_doc_text.setFont(QtGui.QFont('Menlo, Consolas, Courier', 11))
        self.v_doc_text.setReadOnly(True)

        self.v_help_box.layout().addWidget(self.v_doc_text)

    @Inputs.DataFrame
    def set_input_data_frame(self, data_frame):
        self.DataFrame = data_frame

    def _validate_input(self):
        if not super(SparkTransformer, self)._validate_input():
            return False

        if self.DataFrame is None:
            self.v_apply_button.setEnabled(False)
            self.error('Input data frame does not exist')
            for name, parameter in self.parameters_meta().items():
                if parameter.input_column:
                    combo = getattr(self.controls, name)
                    combo.setEditable = True
            return False
        else:
            self.v_apply_button.setEnabled(True)
            # update data column combobox
            # types = dict(self.input_data_frame.dtypes)
            columns = self.DataFrame.columns
            for name, parameter in self.parameters_meta().items():
                if parameter.input_column:
                    if not parameter.input_multiple:
                        saved_value = getattr(self, name)
                        saved_value = saved_value if saved_value in columns else columns[0]
                        combo = getattr(self.controls, name)
                        combo.setEditable = False
                        combo.clear()
                        combo.addItems(columns)
                        # combo.setCurrentIndex(columns.index(saved_value))
                        setattr(self, name, saved_value)
                    else:
                        if len(self.columns_list) == 0:
                            self.columns_list = columns
            return True

    def _validate_parameters(self):
        if not super(SparkTransformer, self)._validate_parameters():
            return False

        df = self.DataFrame
        types = dict(df.dtypes)
        for name, parameter in self.parameters_meta().items():
            if parameter.input_column and parameter.input_dtype is not None:
                input_column = getattr(self, name)
                if types[input_column] != parameter.input_dtype:
                    self.error('Input column must be %s type' % parameter.input_dtype)
                    return False
            if parameter.output_column:
                output_column = getattr(self, name)
                if output_column in df.columns:
                    self.error('Output column must not override an existing one')
                    return False

        return True

from PyQt5 import QtWidgets
from functools import partial

from Orange.widgets import widget, gui
from Orange.widgets.settings import Setting
import weta.core
from weta.gui.async_task import AsyncTask
from weta.gui.spark_environment import SparkEnvironment


class Parameter:
    T_STRING = 'string'
    T_ARRAY_STRING = 'array<string>'
    T_VECTOR = 'vector'

    def __init__(self, type, default_value, label, description='',
                 items=None,
                 input_column=False, input_dtype=None, input_multiple=False,
                 output_column=False,):
        self.type = type
        self.default_value = default_value
        self.label = label
        self.description = description
        self.items = items
        self.input_column = input_column
        self.input_dtype = input_dtype
        self.input_multiple = input_multiple
        self.output_column = output_column


class SparkBase(SparkEnvironment, AsyncTask):
    """
    Base Widget: mainly handle parameter settings, async task
    """

    want_main_area = False
    resizing_enabled = True

    box_text = ''

    class Inputs:
        pass

    class Outputs:
        pass

    class Parameters:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # convert parameters as settings dynamically
        for name, parameter in cls.parameters_meta0().items():
            setattr(cls, name, Setting(parameter.default_value, name=name, tag=parameter))

    def __init__(self):
        super(SparkBase, self).__init__()
        super(AsyncTask, self).__init__()

        # Create parameters Box.
        self.v_main_box = gui.widgetBox(self.controlArea, orientation='horizontal', addSpace=True)
        self.v_setting_box = gui.widgetBox(self.v_main_box, self.box_text if self.box_text != '' else self.name,
                                           addSpace=True)

        self.v_main_box.setMinimumHeight(500)
        self.v_setting_box.setMaximumWidth(250)

        # info area
        self.v_info_box = gui.widgetBox(self.v_setting_box, 'Info:', addSpace=True)

        # setting area
        self.v_parameters_box = gui.widgetBox(self.v_setting_box, 'Parameters:', addSpace=True)

        self.initParametersUI()

        self.v_apply_button = gui.button(self.v_setting_box, self, 'Apply', self.apply)
        self.v_apply_button.setEnabled(False)

    def initParametersUI(self):

        for name, parameter in self.parameters_meta().items():
            if parameter.items is not None:
                gui.comboBox(self.v_parameters_box, self, name, label=parameter.label, labelWidth=300,
                             valueType=parameter.type, items=parameter.items)
            elif parameter.type == bool:
                gui.checkBox(self.v_parameters_box, self, name, label=parameter.label, labelWidth=300)
            elif parameter.input_column:
                items = tuple([parameter.default_value])
                label = parameter.label
                if parameter.input_dtype is not None:
                    label += ' (%s)' % parameter.input_dtype
                if not parameter.input_multiple:
                    gui.comboBox(self.v_parameters_box, self, name, label=label, labelWidth=300,
                                 valueType=parameter.type, items=items, editable=True, sendSelectedValue=True)
                else:
                    self.columns_list = []
                    gui.widgetLabel(self.v_parameters_box, label=label, labelWidth=300)
                    gui.listBox(self.v_parameters_box, self, name, labels='columns_list',
                                 selectionMode=QtWidgets.QListWidget.MultiSelection)

            else:
                gui.lineEdit(self.v_parameters_box, self, name, parameter.label, labelWidth=300,
                             valueType=parameter.type)

    def handleNewSignals(self):
        self.apply()

    def _validate_input(self):
        return True

    def _validate_parameters(self):
        return True

    @classmethod
    def parameters_meta0(cls):
        return inner_class_variables(cls, 'Parameters', Parameter)

    def parameters_meta(self):
        return inner_class_variables(self, 'Parameters', Parameter)

    def inputs_meta(self):
        return inner_class_variables(self, 'Inputs', widget.Input)

    def outputs_meta(self):
        return inner_class_variables(self, 'Outputs', widget.Output)

    def apply(self):
        if self._task is not None:
            # First make sure any pending tasks are cancelled.
            self.cancel()
        assert self._task is None

        self.clear_messages()
        if not self._validate_input() or not self._validate_parameters():
            return

        self.v_apply_button.setEnabled(True)
        # hide window first
        self.hide()

        # environment
        env = {'sqlContext': self.sqlContext, 'sc': self.sc, 'ui': self}

        # collect params
        params = {name: getattr(self, name) for name, parameter in self.parameters_meta().items()}
        # self._apply(params)

        # collect inputs
        inputs = {}
        for input_name, input_var in self.inputs_meta().items():
            assert input_name == input_var.name
            input_value = getattr(self, input_name) if hasattr(self, input_name) else None
            inputs[input_name] = input_value

        func = weta.core.find_func(self.__module__.split('.')[-1])

        self.async_call(partial(func, env, inputs, params))

    def on_finish(self, results):
        # collect outputs
        outputs = {}
        for output_name, output_var in self.outputs_meta().items():
            assert output_name == output_var.name
            outputs[output_name] = output_var
        for output_name, output_value in results.items():
            outputs[output_name].send(output_value)


def inner_class_variables(ctx, inner_cls_name, type):
    if not hasattr(ctx, inner_cls_name):
        return {}
    else:
        cls = getattr(ctx, inner_cls_name)
        return {name: getattr(cls, name) for name in dir(cls) if isinstance(getattr(cls, name), type)}

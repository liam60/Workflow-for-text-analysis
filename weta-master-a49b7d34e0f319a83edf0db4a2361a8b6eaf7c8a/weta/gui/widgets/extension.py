from Orange.widgets.widget import Input


class InputChannel(Input):
    """
    Description of an input signal.

    The class is used to declare input signals for a widget as follows
    (the example is taken from the widget Test & Score)::

        class Inputs:
            train_data = Input("Data", Table, default=True)
            test_data = Input("Test Data", Table)
            learner = Input("Learner", Learner, multiple=True)
            preprocessor = Input("Preprocessor", Preprocess)

    Every input signal must be used to decorate exactly one method that
    serves as the input handler, for instance::

        @Inputs.train_data
        def set_train_data(self, data):
            ...

    Parameters
    ----------
    name (str):
        signal name
    type (type):
        signal type
    id (str):
        a unique id of the signal
    doc (str, optional):
        signal documentation
    replaces (list of str):
        a list with names of signals replaced by this signal
    multiple (bool, optional):
        if set, multiple signals can be connected to this output
        (default: `False`)
    default (bool, optional):
        when the widget accepts multiple signals of the same type, one of them
        can set this flag to act as the default (default: `False`)
    explicit (bool, optional):
        if set, this signal is only used when it is the only option or when
        explicitly connected in the dialog (default: `False`)
    """
    def __init__(self, name, type, id=None, doc=None, replaces=None, *,
                 multiple=False, default=False, explicit=False):
        super(InputChannel, self).__init__(name, type, id, doc, replaces, multiple, default, explicit)
        self.handler = self._default_handler

    def _default_handler(self, data):
        self.data = data

    def has_data(self):
        return self.data is not None

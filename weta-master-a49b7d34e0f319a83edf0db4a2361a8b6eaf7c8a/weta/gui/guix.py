from Orange.widgets import gui

def comboBox(widget, master, value, box=None, label=None, labelWidth=None,
             orientation=Qt.Vertical, items=(), callback=None,
             sendSelectedValue=False, valueType=str,
             emptyString=None, editable=False,
             contentsLength=None, maximumContentsLength=25,
             **misc):
    """
    Construct a combo box.

    The `value` attribute of the `master` contains either the index of the
    selected row (if `sendSelected` is left at default, `False`) or a value
    converted to `valueType` (`str` by default).

    :param widget: the widget into which the box is inserted
    :type widget: QWidget or None
    :param master: master widget
    :type master: OWWidget or OWComponent
    :param value: the master's attribute with which the value is synchronized
    :type value:  str
    :param box: tells whether the widget has a border, and its label
    :type box: int or str or None
    :param orientation: tells whether to put the label above or to the left
    :type orientation: `Qt.Horizontal` (default), `Qt.Vertical` or
        instance of `QLayout`
    :param label: a label that is inserted into the box
    :type label: str
    :param labelWidth: the width of the label
    :type labelWidth: int
    :param callback: a function that is called when the value is changed
    :type callback: function
    :param items: items (optionally with data) that are put into the box
    :type items: tuple of str or tuples
    :param sendSelectedValue: flag telling whether to store/retrieve indices
        or string values from `value`
    :type sendSelectedValue: bool
    :param valueType: the type into which the selected value is converted
        if sentSelectedValue is `False`
    :type valueType: type
    :param emptyString: the string value in the combo box that gets stored as
        an empty string in `value`
    :type emptyString: str
    :param editable: a flag telling whether the combo is editable
    :type editable: bool
    :param int contentsLength: Contents character length to use as a
        fixed size hint. When not None, equivalent to::

            combo.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(contentsLength)
    :param int maximumContentsLength: Specifies the upper bound on the
        `sizeHint` and `minimumSizeHint` width specified in character
        length (default: 25, use 0 to disable)
    :rtype: QComboBox
    """

    # Local import to avoid circular imports
    from Orange.widgets.utils.itemmodels import VariableListModel

    if box or label:
        hb = widgetBox(widget, box, orientation, addToLayout=False)
        if label is not None:
            widgetLabel(hb, label, labelWidth)
    else:
        hb = widget

    combo = OrangeComboBox(
        hb, maximumContentsLength=maximumContentsLength,
        editable=editable)

    if contentsLength is not None:
        combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(contentsLength)

    combo.box = hb
    for item in items:
        if isinstance(item, (tuple, list)):
            combo.addItem(*item)
        else:
            combo.addItem(str(item))

    if value:
        cindex = getdeepattr(master, value)
        model = misc.pop("model", None)
        if model is not None:
            combo.setModel(model)
        if isinstance(model, VariableListModel):
            callfront = CallFrontComboBoxModel(combo, model)
            callfront.action(cindex)
        else:
            if isinstance(cindex, str):
                if items and cindex in items:
                    cindex = items.index(cindex)
                else:
                    cindex = 0
            if cindex > combo.count() - 1:
                cindex = 0
            combo.setCurrentIndex(cindex)

        if isinstance(model, VariableListModel):
            connectControl(
                master, value, callback, combo.activated[int],
                callfront,
                ValueCallbackComboModel(master, value, model)
            )
        elif sendSelectedValue:
            connectControl(
                master, value, callback, combo.activated[str],
                CallFrontComboBox(combo, valueType, emptyString),
                ValueCallbackCombo(master, value, valueType, emptyString))
        else:
            connectControl(
                master, value, callback, combo.activated[int],
                CallFrontComboBox(combo, None, emptyString))
    miscellanea(combo, hb, widget, **misc)
    combo.emptyString = emptyString
    return combo
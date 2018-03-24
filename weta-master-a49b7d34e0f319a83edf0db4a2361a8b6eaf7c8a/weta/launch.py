import sys
import os
import Orange.canvas.__main__ as orange
from Orange import widgets
import logging
import Orange.canvas.registry.discovery as discovery
import nltk.data

def dummy_widget_discovery(discovery):
    pass

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.ERROR)

    setattr(widgets, 'widget_discovery', dummy_widget_discovery)

    import os
    # os.remove("/Users/Chao/Library/Caches/Orange/3.4.5/canvas/widget-registry.pck")
    sys.exit(orange.main())

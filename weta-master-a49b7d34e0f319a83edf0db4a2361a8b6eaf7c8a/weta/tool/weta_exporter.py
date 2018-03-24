import xml.etree.ElementTree as ET
import ast
import pickle
import base64
import re

class Exporter:
    def export(self, ows_file, export_path):
        tree = ET.parse(ows_file)
        schema = tree.getroot()
        node_elements = schema.findall("./nodes/node")
        link_elements = schema.findall("./links/link")
        node_properties_elements = schema.findall("./node_properties/properties")

        self.nodes = {}
        for ne in node_elements:
            node = ne.attrib
            node['inputs'] = {}
            node['outputs'] = {}
            self.nodes[ne.attrib['id']] = node

        self.links = [le.attrib for le in link_elements]
        self.node_properties = {np.attrib['node_id']: self.parse_properties(np) for np in node_properties_elements}

        for link in self.links:
            sink_node_id = link['sink_node_id']
            sink_channel = link['sink_channel']
            source_node_id = link['source_node_id']
            source_channel = link['source_channel']

            self.nodes[sink_node_id]['inputs'][sink_channel] = {'node_id': source_node_id, 'channel': source_channel}

            if source_channel not in self.nodes[source_node_id]['outputs']:
                self.nodes[source_node_id]['outputs'][source_channel] = []
            self.nodes[source_node_id]['outputs'][source_channel].append({'node_id': sink_node_id, 'channel': sink_channel})

        for node_id in self.node_properties.keys():
            self.nodes[node_id]['settings'] = self.node_properties[node_id]['settings']
            self.nodes[node_id]['settings_format'] = self.node_properties[node_id]['settings_format']

        # find root nodes
        roots = []
        for node in self.nodes.values():
            if len(node['inputs'].keys()) == 0:
                roots.append(node)

        code = []
        code.append('from weta.core.weta_lib import * \n')
        # start from roots, generate code
        generated_nodes = set()

        # for defined scripts, generate them firstly
        for node in self.nodes.values():
            func_name = node['qualified_name'].split('.')[-2]
            if func_name == 'pyspark_script_console':
                func_name = func_name + node['id']
                script = node['settings']['_script'].replace('\n', '  \n', -1)
                code.append('def %s(inputs, settings):' % func_name)
                code.append("    data = inputs.get('data', None)")
                code.append("    df = inputs.get('df', None)")
                code.append("    df1 = inputs.get('df1', None)")
                code.append("    df2 = inputs.get('df2', None)")
                code.append("    df3 = inputs.get('df3', None)")
                code.append("    transformer = inputs.get('transformer', None)")
                code.append("    estimator = inputs.get('estimator', None)")
                code.append("    model = inputs.get('model', None) \n")
                code.append(re.sub( '^',' '*4, script ,flags=re.MULTILINE ))
                code.append("")
                code.append("    return {'data': data, 'df': df, 'df1': df1, 'df2': df2, 'df3': df3, 'transformer': transformer, 'estimator': estimator, 'model': model} \n")

        for root in roots:
            stack = [root]

            while len(stack) > 0:
                node = stack.pop()

                if node['id'] in generated_nodes:
                    continue

                unsatisfied_input = False
                for input in node['inputs'].keys():
                    input_node_id = node['inputs'][input]['node_id']
                    if input_node_id not in generated_nodes:
                        unsatisfied_input = True
                        stack.append(node)
                        stack.append(self.nodes[input_node_id])
                        break

                if unsatisfied_input:
                    continue

                # start generating
                code.append('\n# %% ' + node['title'])  # for run cell in VS code

                input_entries = []
                for input in node['inputs'].keys():
                    input_node_id = node['inputs'][input]['node_id']
                    input_node_channel = node['inputs'][input]['channel']
                    input_entries.append("'%s': outputs%s['%s']" % (input, input_node_id, input_node_channel) )
                code.append('inputs%s = {%s}' % (node['id'], ", ".join(input_entries)))
                qualified_name: str = node['qualified_name']
                func_name = qualified_name.split('.')[-2]

                if func_name == 'pyspark_script_console':
                    func_name = func_name + node['id']
                    code.append('settings%s = {}' % node['id'])
                else:
                    code.append('settings%s = %s' % (node['id'], node['settings']))

                code.append('outputs%s = %s(inputs%s, settings%s)' % (node['id'], func_name, node['id'], node['id']))

                generated_nodes.add(node['id'])

                for output in node['outputs'].keys():
                    output_nodes = node['outputs'][output]
                    for output_node in output_nodes:
                        output_node_id = output_node['node_id']
                        stack.append(self.nodes[output_node_id])

        print('\n'.join(code))



    @staticmethod
    def parse_properties(np):
        format = np.get('format')
        node_id = np.get('node_id')
        data: str = np.text

        properties = None
        if format == 'literal':
            properties = ast.literal_eval(data)
        if format == 'pickle':
            properties = pickle.loads(base64.decodebytes(data.encode('ascii')))

        excludes = ['savedWidgetGeometry', '__version__']
        for prop in excludes:
            if prop in properties:
                properties.pop(prop)

        return {'node_id': node_id, 'settings': properties, 'settings_format': format}


if __name__ == '__main__':
    exporter = Exporter()
    exporter.export('/Users/Chao/nzpolice/summer/weta.ows', '')
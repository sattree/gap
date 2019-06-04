from itertools import chain
import subprocess
import json

import nltk
import networkx as nx

class CoreNLPServer():
    def __init__(self, classpath=None, corenlp_options=None, java_options=['-Xmx5g']):
        self.classpath = classpath
        self.corenlp_options = corenlp_options
        self.java_options = java_options
        
    def start(self):
        corenlp_options = [('-'+k, str(v)) for k,v in self.corenlp_options.items()]
        corenlp_options = list(chain(*corenlp_options))
        cmd = ['java']\
            + self.java_options \
            + ['-cp'] \
            + [self.classpath+'*'] \
            + ['edu.stanford.nlp.pipeline.StanfordCoreNLPServer'] \
            + corenlp_options
        self.popen = subprocess.Popen(cmd, 
                                        # shell=True,
                                        stdout=subprocess.PIPE, 
                                        stderr=subprocess.STDOUT, 
                                        universal_newlines=True,
                                        close_fds=True)

        # for line in iter(self.popen.stdout.readline, ''):
        #     print(line)

        # self.popen.wait()
        
        self.url = 'http://localhost:{}/'.format(self.corenlp_options.port)
        
    def stop(self):
        self.popen.terminate()
        self.popen.wait()

# timeout is hardcoded to 60s in nltk implementation
def api_call(self, data, properties=None, timeout=600):
        default_properties = {
            'outputFormat': 'json',
            'annotators': 'tokenize,pos,lemma,ssplit,{parser_annotator}'.format(
                parser_annotator=self.parser_annotator
            ),
        }

        default_properties.update(properties or {})

        response = self.session.post(
            self.url,
            params={'properties': json.dumps(default_properties)},
            data=data.encode(self.encoding),
            timeout=timeout,
        )

        response.raise_for_status()

        return response.json()

def map_chars_to_tokens(doc, char_offset):
    #### Convert character level offsets to token level
    # tokenization or mention labelling may not be perfect
    # Identify the token that contains first character of mention
    # Identify the token that contains last character of mention
    # example - token: 'Delia-', mention: 'Delia'
    #           token: 'N.J.Parvathy', mention: 'Parvathy'
    # Token starts before the last character of mention and ends after the last character of mention
    # Remember character offset end here is the character immediately after the token
    return next(filter(lambda token: char_offset in range(token.idx, token.idx+len(token.text)), doc), None).i

def parse_tree_to_graph(sent_trees, doc, tokens=None, **kwargs):
    graph = nx.Graph() 
    leaves = []
    edges = []
    for sent_tree in sent_trees:
        edges, leaves = get_edges_in_tree(sent_tree, leaves=leaves, path='', edges=edges, **kwargs)
    graph.add_edges_from(edges)
    
    if tokens is None:
        tokens = [token.word for token in doc]
    assert tokens == leaves, 'Tokens in parse tree and input sentence don\'t match.'
    
    return graph
    
#DFS
# trace path to create unique names for all nodes
def get_edges_in_tree(parent, leaves=[], path='', edges=[], lrb_rrb_fix=False):
    for i, node in enumerate(parent):
        if type(node) is nltk.Tree:
            from_node = path
            to_node = '{}-{}-{}'.format(path, node.label(), i)
            edges.append((from_node, to_node))

            if lrb_rrb_fix:
                if node.label() == '-LRB-':
                    leaves.append('(')
                if node.label() == '-RRB-':
                    leaves.append(')')

            edges, leaves = get_edges_in_tree(node, leaves, to_node, edges)
        else:
            from_node = path
            to_node = '{}-{}'.format(node, len(leaves))
            edges.append((from_node, to_node))
            leaves.append(node)
    return edges, leaves

def get_syntactical_distance_from_graph(graph, token_a, token_b, debug=False):
       return nx.shortest_path_length(graph, 
                                   source='{}-{}'.format(token_a.word if hasattr(token_a, 'word') else token_a.text, token_a.i),
                                   target='{}-{}'.format(token_b.word if hasattr(token_b, 'word') else token_b.text, token_b.i))

def get_normalized_tag(token):
        tag = token.dep_
        tag = 'subj' if 'subj' in tag else tag
        tag = 'dobj' if 'dobj' in tag else tag
        return tag
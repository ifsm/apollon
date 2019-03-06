"""
grapher.py -- Plot graphs from HMMs.

Copyright (C) 2017 Michael Blaß

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from matplotlib import cm
from matplotlib.patches import ArrowStyle
from matplotlib.patches import Circle
from matplotlib.patches import ConnectionStyle
from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt

import numpy as np
import networkx as nx
from scipy.spatial import distance

from apollon import tools


def _prepare_fig(pos):
    """Prepare a figure with the correct size.

    Params:
        pos    (dict) with structur {node_name_i: np.array([pos_x, pos_y])}
                      as return by nx.layout methods.
    Return:
        (Figure, AxesSubplot)
    """
    pos_data = np.array(list(pos.values()))
    diameter = distance.pdist(pos_data).max()
    dd = diameter / 2 + 1

    fig = plt.figure(figsize=(7, 7), frameon=False)
    ax = fig.add_subplot(111)
    r = 1.5
    ax.axis([-(dd+r), (dd+r), -(dd+r), (dd+r)])
    ax.set_axis_off()

    return fig, ax


def _draw_nodes(G, pos, ax):
    """Draw the nodes of a (small) networkx graph.

    Params:
        G    (nx.classes.*) a networkx graph.
        pos  (dict)         returned by nx.layout methods.
        ax   (AxesSubplot)  mpl axe.

    Return:
        (dict) of Circle patches.
    """
    #degree = np.array([deg for node, deg in G.degree], dtype=float)
    #degree /= degree.sum()

    flare_kwargs = {'alpha'    : 0.2,
                    'edgecolor': (0, 0, 0, 1),
                    'facecolor': None}

    node_kwargs = {'alpha'    : 0.8,
                   'edgecolor': (0, 0, 0, 1),
                   'facecolor': None}

    nodes = {}
    node_params = zip(pos.items())

    for i, (label, xy) in enumerate(pos.items()):
        size = G.nodes[label]['size']
        fsize = G.nodes[label]['fsize']
        flare_kwargs['facecolor'] = 'C{}'.format(i)
        flare = Circle(xy, size+fsize, **flare_kwargs)

        node_kwargs['facecolor'] = 'C{}'.format(i)
        node = Circle(xy, size, **node_kwargs)

        ax.add_patch(flare)
        ax.add_patch(node)

        font_style = {'size':15, 'weight':'bold'}
        text_kwargs = {'color': (0, 0, 0, .8),
                       'verticalalignment': 'center',
                       'horizontalalignment': 'center',
                       'fontdict': font_style}
        ax.text(*xy, i+1, **text_kwargs)

        nodes[label] = node

    return nodes


def _draw_edges(G, pos, nodes, ax):
    """Draw the edges of a (small) networkx graph.

    Params:
        G       (nx.classes.*)  a networkx graph.
        pos     (dict)          returned by nx.layout methods.
        nodes   (dict)          of Circle patches.
        ax      (AxesSubplot)   mpl axe.

    Return:
        (dict) of Circle patches.
    """
    pointer = ArrowStyle.Fancy(head_width=10, head_length=15)
    curved_edge = ConnectionStyle('arc3', rad=.2)

    arrow_kwargs = {'arrowstyle': pointer,
                    'antialiased': True,
                    'connectionstyle': curved_edge,
                    'edgecolor': None,
                    'facecolor': None,
                    'linewidth': None}

    edges = {}
    for i, (a, b, attr) in enumerate(G.edges.data()):
        arrow_kwargs['edgecolor'] = attr['color']
        arrow_kwargs['facecolor'] = attr['color']
        arrow_kwargs['linewidth'] = 1.0

        edge = FancyArrowPatch(pos[a], pos[b],
                               patchA=nodes[a], patchB=nodes[b],
                               shrinkA=5, shrinkB=5,
                               **arrow_kwargs)
        ax.add_patch(edge)
        edges[(a, b)] = edge

    return edges


def _legend(G, nodes, ax):
    """Draw the legend for a (small) nx graph.

    Params:
        G       (nx.classes.*) a networkx graph.
        nodes   (list)         of Circle patches.
        ax      (AxesSubplot)  mpl axe.

    Return:
        (AxesSubplot)
    """
    legend_kwargs = {'fancybox': True,
                     'fontsize': 14,
                     'bbox_to_anchor': (1.02, 1.0)}

    labels = [r'$f_c = {:>9.3f}$ Hz'.format(k) for k in G.nodes.keys()]
    legend = ax.legend(nodes.values(), labels, **legend_kwargs, borderaxespad=0)

    return legend


def draw_network(labels, tpm, delta):
    """Draw the graph of a HMM's transition probability matrix.

    Params:
        lables  (iterable)      Labels for each state.
        tpm     (np.ndarray)    A two-dimensional (row) stochastic matrix.
        delta   (iterable)

    Return:
        (Figure, AxesSubplot)
    """
    G = nx.MultiDiGraph()
    #scaled_tpm = np.exp(tools.scale(tpm, 0, 1.5))

    for i, from_state in enumerate(labels):
        G.add_node(from_state, fsize=np.exp(delta[i]))

        for j, to_state in enumerate(labels):
            if not np.isclose(tpm[i, j], 0.0):
                G.add_edge(from_state, to_state,
                           weight=tpm[i, j],
                           color='k')

    sd = np.sum([np.exp(degree) for node, degree in G.degree()])

    for node, degree in G.degree():
        G.node[node]['size'] = .5 + np.exp(degree) / sd

    #pos = nx.layout.circular_layout(G, center=(0., 0.), scale=4)
    pos = nx.layout.spring_layout(G, center=(0.0, 0.0), scale=4)

    fig, ax = _prepare_fig(pos)
    nodes = _draw_nodes(G, pos, ax)
    edges = _draw_edges(G, pos, nodes, ax)
    legend = _legend(G, nodes, ax)

    return fig, ax, G


def save_hmmfig(fig, path, **kwargs):
    """Save the figure to file.

    This saves the figure and ensures that the out-of-axes legend
    is completely visible in the saved version.

    All kwargs are passed on to plt.savefig.

    Params:
        fig     (Figure)    Figure of HMM tpm.
        path    (str)       Path to save file.
    """
    fig.savefig(fname=path,
                bbox_extra_artists=(fig.axes[0].legend_,),
                bbox_inches='tight', **kwargs)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from itertools import chain, combinations
from functools import partial
from matplotlib.patches import Rectangle, Circle

class UpSetPlot():
    def __init__(self, n_bases, n_data_points, space=3, log=False, filtered=False):
        """
        Generates figures and axes.

        :param n_bases: The number of unique instances

        :param n_data_points: The number of histogram bars (nb of unique sets)
        """
        # map queries to graphic properties
        self.query = []
        qu_col = plt.cm.rainbow(np.linspace(.01, .99, len(self.query)))
        self.query2color = dict(zip([frozenset(q) for q in self.query], qu_col))
        self.query2zorder = dict(zip([frozenset(q) for q in self.query], np.arange(len(self.query)) + 1))

        # set standard colors
        self.greys = plt.cm.Greys([.22, .8])

        # figure details
        self.space = space
        self.log = log
        self.filtered = filtered

        # set figure properties
        self.rows = n_bases
        self.cols = n_data_points
        self.x_values, self.y_values = self._create_coordinates(n_bases, n_data_points)
        self.fig, self.ax_intbars, self.ax_intmatrix, \
        self.ax_setsize, self.ax_tablenames, = self._prepare_figure()

        self.standard_graph_settings = {
            'scatter': {
                'alpha': .3,
                'edgecolor': None
            },
            'hist': {
                'histtype': 'stepfilled',
                'alpha': .3,
                'lw': 0
            }
        }

    def _create_coordinates(self, rows, cols):
        """
        Creates the x, y coordinates shared by the main plots.

        :param rows: number of rows of intersection matrix
        :param cols: number of columns of intersection matrix
        :return: arrays with x and y coordinates
        """
        x_values = (np.arange(cols) + 1)
        y_values = (np.arange(rows) + 1)
        return x_values, y_values

    def _prepare_figure(self):
        """
        Prepares the figure, axes (and their grid)

        :return: references to the newly created figure and axes
        """
        fig = plt.figure(figsize=(25, 12.5))
        topgs = gridspec.GridSpec(1, 1)[0, 0]
        fig_cols = self.cols + 0
        fig_rows = self.rows + self.rows * 4

        gs_top = gridspec.GridSpecFromSubplotSpec(fig_rows, fig_cols, subplot_spec=topgs, wspace=.1, hspace=.2)
        # extend below plots to go higher:
        extend_to_top = 4
        setsize_w, setsize_h = 3, self.rows + extend_to_top
        #tablesize_w, tablesize_h = setsize_w + 2, self.rows
        # move hbar plot more to the left (default=2) 3...for small, 7...for large (is now self.space)
        extend_hbar_to_left = self.space
        tablesize_w, tablesize_h = setsize_w + extend_hbar_to_left, self.rows + extend_to_top
        intmatrix_w, intmatrix_h = tablesize_w + self.cols, self.rows + extend_to_top
        intbars_w, intbars_h = tablesize_w + self.cols, self.rows * 3

        ax_setsize = plt.subplot(gs_top[-setsize_h:-1, 0:setsize_w])
        ax_tablenames = plt.subplot(gs_top[-tablesize_h:-1, setsize_w:tablesize_w])
        ax_intmatrix = plt.subplot(gs_top[-intmatrix_h:-1, tablesize_w:intmatrix_w])
        # move lower part of upper bar plot to top
        extend_bar_to_top = 4
        ax_intbars = plt.subplot(gs_top[:self.rows * 3 + extend_bar_to_top, tablesize_w:intbars_w])

        return fig, ax_intbars, ax_intmatrix, ax_setsize, ax_tablenames

    def _color_for_query(self, query):
        """
        Helper function that returns the standard dark grey for non-queried intersections, and the color assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :return: color as length 4 array.
        """
        query_color = self.query2color.get(query, self.greys[1])
        return query_color

    def _zorder_for_query(self, query):
        """
        Helper function that returns 0 for non-queried intersections, and the zorder assigned to
        a query when the class was instantiated otherwise
        :param query: frozenset.
        :return: zorder as int.
        """
        query_zorder = self.query2zorder.get(query, 0)
        return query_zorder

    def main_plot(self, set_names, occurance_list, in_sets, out_sets, set_sizes, unique_bases):
        """
        Creates the main graph comprising bar plot of base set sizes, bar plot of intersection sizes and intersection
        matrix.

        :param set_names: array of names of input data frames, sorted (as above)

        :param in_sets: list of lists. Each list represents the available base sets in a that number of instances.

        :param out_sets: list of lists. Complementary set to in_sets.

        :param set_sizes: array of ints. Contains the intersection sizes.

        :return: dictionary containing figure and axes references.
        """
        # calculate sum of elements for lower plot representing the percentage of missing values
        total_nb_elements = sum(occurance_list)
        # Plots horizontal bar plot for base set sizes (lower left plot)
        ylim = self._base_sets_plot(set_sizes, total_nb_elements)
        self._table_names_plot(set_names, ylim)
        # bar plot showing the set sizes (main plot)
        xlim = self._inters_sizes_plot(occurance_list)
        set_row_map = dict(zip(unique_bases, self.y_values))
        # TODO: adapt _inters_matrix to the available data structures
        self._inters_matrix(in_sets, out_sets, xlim, ylim, set_row_map)
        return {'figure': self.fig,
                'intersection_bars': self.ax_intbars,
                'intersection_matrix': self.ax_intmatrix,
                'base_set_size': self.ax_setsize,
                'names': self.ax_tablenames}

    def _table_names_plot(self, sorted_set_names, ylim):
        ax = self.ax_tablenames
        ax.set_ylim(ylim)
        xlim = ax.get_xlim()
        tr = ax.transData.transform
        for i, name in enumerate(sorted_set_names):
            ax.text(x=1,  # (xlim[1]-xlim[0]/2),
                    y=self.y_values[i],
                    s=name,
                    fontsize=12,
                    clip_on=True,
                    va='center',
                    ha='right',
                    transform=ax.transData,
                    family='monospace')

        # if len(self.x_values) > 1:
        # row_width = self.x_values[1] - self.x_values[0]
        # else:
        #     row_width = self.x_values[0]
        #
        # background = plt.cm.Greys([.09])[0]
        #
        # for r, y in enumerate(self.y_values):
        #     if r % 2 == 0:
        #         ax.add_patch(Rectangle((xlim[0], y - row_width / 2), height=row_width,
        #                                width=xlim[1],
        #                                color=background, zorder=0))
        ax.axis('off')


    def _base_sets_plot(self, sorted_set_sizes, total_nb_elements):
        """
        Plots horizontal bar plot for base set sizes.

        :param sorted_sets: list of data frames, sorted according to user's directives.
        :return: Axes.
        """
        ax = self.ax_setsize
        ax.invert_xaxis()
        height = .6
        bar_bottoms = self.y_values

        sorted_set_sizes_ratio = [elem/total_nb_elements  for elem in sorted_set_sizes]

        bars = ax.barh(bar_bottoms, sorted_set_sizes_ratio, height=height, color=self.greys[1])
        for i, v in enumerate(sorted_set_sizes_ratio):
            ax.text(v+0.1, (i+1), '{:.1f}'.format(v*100), color=self.greys[1], size=11, va='center', ha='right')


        self._strip_axes(ax, keep_spines=['bottom'])#, keep_ticklabels=['bottom'])

        ax.set_ylim((height / 2, (ax.get_ylim()[1] + height / 2)))
        xlim = ax.get_xlim()
        gap = max(xlim) / 500.0 * 20
        ax.set_xlim(xlim[0] + gap, xlim[1] - gap)
        xlim = ax.get_xlim()
        ax.spines['bottom'].set_bounds(xlim[0], xlim[1])

        # bracket_height = ax.transData.inverted().transform([(0, 0), (0, ax.get_ylim()[1])])
        # bracket_height = np.abs(bracket_height[1, 1] - bracket_height[0, 1])
        # for i, (x, y) in enumerate(zip([len(x) for x in sorted_sets], self.y_values)):
        # ax.annotate(sorted_set_names[i], rotation=90, ha='right', va='bottom', fontsize=15,
        #                 xy=(x, y), xycoords='data',
        #                 xytext=(-30, 0), textcoords='offset points',
        #                 arrowprops=dict(arrowstyle="-[, widthB=%s"%(bracket_height,),
        #                                 shrinkA=1,
        #                                 shrinkB=3,
        #                                 connectionstyle='arc,angleA=-180, angleB=180, armB=30',
        #                                 ),
        #                 )

        ax.set_xlabel("Percent missing", fontweight='bold', fontsize=10, labelpad=8)

        return ax.get_ylim()

    def _strip_axes(self, ax, keep_spines=None, keep_ticklabels=None):
        """
        Removes spines and tick labels from ax, except those specified by the user.

        :param ax: Axes on which to operate.
        :param keep_spines: Names of spines to keep.
        :param keep_ticklabels: Names of tick labels to keep.

        Possible names are 'left'|'right'|'top'|'bottom'.
        """
        tick_params_dict = {'which': 'both',
                            'bottom': False,
                            'top': False,
                            'left': False,
                            'right': False,
                            'labelbottom': False,
                            'labeltop': False,
                            'labelleft': False,
                            'labelright': False}
        if keep_ticklabels is None:
            keep_ticklabels = []
        if keep_spines is None:
            keep_spines = []
        lab_keys = [(k, "".join(["label", k])) for k in keep_ticklabels]
        for k in lab_keys:
            tick_params_dict[k[0]] = False
            tick_params_dict[k[1]] = False
        ax.tick_params(**tick_params_dict)
        for sname, spine in ax.spines.items():
            if sname not in keep_spines:
                spine.set_visible(False)

    def _inters_sizes_plot(self, set_sizes):
        """
        Plots bar plot for intersection sizes.
        to the user's directives

        :param set_sizes: array of ints. Sorted, likewise.

        :return: Axes
        """
        ax = self.ax_intbars
        width = .5
        self._strip_axes(ax, keep_spines=['left']) #, keep_ticklabels=['left'])

        bar_bottom_left = self.x_values

        # bars should all have the same color (for now)
        bar_colors = [self.greys[1] for val in set_sizes]

        ax.bar(bar_bottom_left, set_sizes, width=width, color=bar_colors, linewidth=0)
        if self.log:
            ax.set_yscale('log')

        ylim = ax.get_ylim()
        label_vertical_gap = (ylim[1] - ylim[0]) / 60

        for x, y in zip(self.x_values, set_sizes):
            if self.log:
                ax.text(x, 1.3*y, "%.5g" % y,
                        rotation=90, ha='center', va='bottom')
            else:
                ax.text(x, y + label_vertical_gap, "%.5g" % y,
                        rotation=90, ha='center', va='bottom')

        if not self.log:
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 15))
        else:
            xlim = ax.get_xlim()
            range_labels = range(5,10) if self.filtered else range(1,10,2)
            for i in range_labels:
                ax.text(xlim[1]*0.99, 10**i * 1.3, r'$10^{%i}$' % i, ha='center', va='bottom', color='grey')


        gap = max(ylim) / 500.0 * 20
        ax.set_ylim(ylim[0] - gap, ylim[1] + gap)
        ylim = ax.get_ylim()
        ax.spines['left'].set_bounds(ylim[0], ylim[1])

        if self.log:
            ax.yaxis.grid(True, lw=.35, color='grey', ls=':')
        ax.set_axisbelow(True)
        ax.set_ylabel("Intersection size", labelpad=8, fontweight='bold', fontsize=13)

        return ax.get_xlim()

    def _inters_matrix(self, in_sets, out_sets, xlims, ylims, set_row_map):
        """
        Plots intersection matrix.

        :param in_sets
        :param out_sets

        :param xlims: tuple. x limits for the intersection matrix plot.

        :param ylims: tuple. y limits for the intersection matrix plot.

        :param set_row_map: dict. Maps data frames (base sets) names to a row of the intersection matrix

        :return: Axes
        """
        ax = self.ax_intmatrix
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        if len(self.x_values) > 1:
            row_width = self.x_values[1] - self.x_values[0]
        else:
            row_width = self.x_values[0]

        self._strip_axes(ax)

        background = plt.cm.Greys([.09])[0]

        for r, y in enumerate(self.y_values):
            if r % 2 == 0:
                ax.add_patch(Rectangle((xlims[0], y - row_width / 2), height=row_width,
                                       width=xlims[1],
                                       color=background, zorder=0))

        for col_num, (in_s, out_s) in enumerate(zip(in_sets, out_sets)):
            in_y = [set_row_map[s] for s in in_s]
            out_y = [set_row_map[s] for s in out_s]
            # in_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[1]) for y in in_y]
            # out_circles = [Circle((self.x_values[col_num], y), radius=dot_size, color=self.greys[0]) for y in out_y]
            # for c in chain.from_iterable([in_circles, out_circles]):
            # ax.add_patch(c)
            ax.scatter(np.repeat(self.x_values[col_num], len(in_y)), in_y, color=np.tile(self.greys[1], (len(in_y), 1)),s=200)
            ax.scatter(np.repeat(self.x_values[col_num], len(out_y)), out_y, color=self.greys[0], s=200)
            if len(in_y)>0:
                ax.vlines(self.x_values[col_num], min(in_y), max(in_y), lw=3.5, color=self._color_for_query(frozenset(in_s)))



# ################################### own functions ###########################################
# function which seperates the differnt bases
# - input: list of lists containing nan column names 
def get_base_sets(nan_list):
    list_of_lists = nan_list.copy()
    list_of_lists.sort(key=len)
    base_sets = []
    for lst in list_of_lists:
        x = set(lst)
        for base in base_sets:
            # some instances are in two base sets: 
            # if statement ensures that the sets are represented as a whole 
            #    -> one value can therefore exist in multiple base sets (which is what we want)
            if len(set(x)-set(base))==(len(x)-len(base)):
                x -= set(base)
        if len(list(x))>0:
            base_sets.append(list(x))
    # remove duplicates in bases
    new_intersects = []
    for i in range(len(base_sets)):
        for j in range(i+1, len(base_sets)):
            # if there exists an intersection
            intersect = list(set(base_sets[i]).intersection(base_sets[j]))
            if len(intersect)>0:
                # save it in new_intersects
                new_intersects.append(intersect)
                # then remove it from list_of_list[i&j]
                for item in intersect:
                    base_sets[i].remove(item)
                    base_sets[j].remove(item)
    # keep only unique elements       
    new_intersects = [list(x) for x in set(tuple(x) for x in new_intersects)]
    new_intersects.sort(key=len, reverse=True)
    # append to main list
    for base in new_intersects:
        base_sets.append(base)
    # return only not empty lists
    return [x for x in base_sets if x != []]


def get_bases(column_list, base_sets):
    """create list of missing bases (bases turned into numbers for easier comparability)"""
    x = set(column_list)
    base_lst = []
    for i, base in enumerate(base_sets):
        if x != x-set(base):
            base_lst.append(i)
    return base_lst



# -------- main --------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--linear', help='Main histogram has linear instead of logarithmic scale', default=False, action='store_true')
    parser.add_argument('--filter', help='Display only the 99.99% most occuring sub sets.', default=False, action='store_true')
    args = parser.parse_args()
    # now the 'real' main can begin
    import sys
    path_missing_dict = '/home/sebastian/Documents/astonomy-python/data-storage/dr2_storage_finished_list_full.npy'
    mis_list = np.load(path_missing_dict)
    mis_list = sorted(mis_list, key=lambda k: k['occurance'], reverse=True)
    # 1) prepare input for further usage 
    n_mis     = [dic['nan-columns'] for dic in mis_list]
    occu_list = [dic['occurance'] for dic in mis_list]
    base_list = [get_bases(dic['nan-columns'], get_base_sets(n_mis)) for dic in mis_list]
    # filter out the top 99.99% most frequently occuring base-set combinations
    if args.filter:
        total_nb_entries = sum(occu_list)
        print(total_nb_entries)
        sum_occ = 0
        break_idx = 0
        for i, occ in enumerate(occu_list):
            print(occ)
            sum_occ += occ
            if sum_occ>total_nb_entries*0.9999:
                break_idx = i+1
                break

        print('break_idx: {}'.format(break_idx))
        base_list = base_list[:break_idx]
        occu_list = occu_list[:break_idx]

    # 2) unique bases
    unique_bases = list(set([item for sublist in base_list for item in sublist]))
    # count how often a certain base is missing (each time a base is in a base list we 
    # add up that total_base_occurance for that base...histogram for bases) 
    total_base_occurance = {}
    none_missing = 0
    for base in unique_bases:
        total_base_occurance[base] = 0
        for (base_l, occ) in zip(base_list, occu_list):
            if base in base_l:
                total_base_occurance[base] += occ
            elif len(base_l) == 0:
                none_missing = occ

    # 3) invert the base list
    invert_base_list = []
    for base_l in base_list:
        invert_base_list.append(list(set(unique_bases)-set(base_l)))

    print('\nAvailable bases: {}\n'.format(get_base_sets(n_mis)))
    print('Base list(len={}): {}'.format(len(base_list),base_list))
    print('Inverse base list: {}'.format(invert_base_list))
    # print('Occurance list: {}'.format(occu_list))
    # after one pass we now have to by hand name the base sets:
    base_set_names = ['Astrometric pseudo-color', 'Radial velocity', 'Extinction/Redening',
            'Radius/Luminiosity', 'Effective Temperature', 'Flux RP band', 'Parallax/Proper motion', 'Flux BP band']
    print('Base names: {}'.format(base_set_names))

    # remove feature 8 from the pool of features
    remove_nb = 8
    if remove_nb in unique_bases:
        unique_bases.remove(remove_nb)
    for i in range(len(base_list)):
        if remove_nb in base_list[i]:
            base_list[i].remove(remove_nb)
        else:
            invert_base_list[i].remove(remove_nb)
    total_base_occurance.pop(remove_nb, None)

    base_occurances = [total_base_occurance[k] for k in total_base_occurance.keys()]
    sort=True
    if sort:
        base_occurances_sorted = [base_occurances[idx] for idx in sorted(range(len(base_occurances)), key=lambda k: base_occurances[k], reverse=True)]
        base_set_names_sorted = [base_set_names[idx] for idx in sorted(range(len(base_occurances)), key=lambda k: base_occurances[k], reverse=True)]
        unique_bases_sorted = [unique_bases[idx] for idx in sorted(range(len(base_occurances)), key=lambda k: base_occurances[k], reverse=True)]
        base_occurances = base_occurances_sorted
        base_set_names = base_set_names_sorted
        unique_bases = unique_bases_sorted

    upset = UpSetPlot(n_bases=len(unique_bases), n_data_points=len(base_list), space=3, log=(not args.linear), filtered=args.filter)
    fig_dict = upset.main_plot(set_names=base_set_names, occurance_list=occu_list, in_sets=base_list, out_sets=invert_base_list,
            set_sizes=base_occurances, unique_bases=unique_bases)

    plt.savefig('./savefig.png')

